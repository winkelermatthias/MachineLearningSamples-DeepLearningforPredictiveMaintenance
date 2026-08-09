/* POSIX port of the relspec sensor firmware: a faithful device simulator.
 *
 * Runs the same core (rs_proto, rs_ring) an MCU port runs, but samples a
 * synthetic bearing+shaft vibration signal instead of an ADC. Speaks the
 * full RSP/1 surface: UDP beacons, TCP HELLO/DATA/ACQ_END stream, and
 * responds to gateway control frames (START/STOP/SET_FS/IDENT).
 *
 *   ./device-sim --id RS-000A01 --gateway 127.0.0.1:47701 \
 *       --fs 8192 --acq-ms 1000 --count 4 --speed 50 --growth 0.4
 *
 * --speed N paces the stream N× faster than real time (test rigs);
 * --speed 0 streams flat out.
 */
#define _POSIX_C_SOURCE 200809L
#define _USE_MATH_DEFINES
#include <arpa/inet.h>
#include <errno.h>
#include <math.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
#include "../../core/rs_proto.h"
#include "../../core/rs_ring.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FW_VER 0x0100
#define CHUNK 1024
#define FULLSCALE_G 8.0

typedef struct {
    const char *id;
    const char *gw_host; int gw_port;
    const char *bc_host; int bc_port;
    uint32_t fs;
    int acq_ms, period_ms, count;
    double speed, growth;
    unsigned seed;
    int wait_start;
} cfg_t;

static uint64_t now_us(void)
{
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
}

static void sleep_us(uint64_t us)
{
    struct timespec ts = { (time_t)(us / 1000000ull),
                           (long)(us % 1000000ull) * 1000L };
    nanosleep(&ts, NULL);
}

/* ---- synthetic machine: shaft tone + vane tone + BPFO impacts ---- */
typedef struct {
    double fr, ph1, ph3, res_ph, env;
    double next_impact_t, t;
    double severity;
    unsigned rng;
} synth_t;

static double frand(unsigned *s)
{   /* xorshift; uniform [-1,1) */
    unsigned x = *s; x ^= x << 13; x ^= x >> 17; x ^= x << 5; *s = x;
    return (double)(int)x / 2147483648.0;
}

static void synth_init(synth_t *sy, unsigned seed, double severity)
{
    memset(sy, 0, sizeof *sy);
    sy->fr = 29.5;
    sy->severity = severity;
    sy->rng = seed ? seed : 1;
    sy->next_impact_t = 0.01;
}

static double synth_sample(synth_t *sy, double fs)
{
    const double dt = 1.0 / fs;
    double x = 0.30 * sin(sy->ph1) + 0.10 * sin(sy->ph3);
    sy->ph1 += 2 * M_PI * sy->fr * dt;
    sy->ph3 += 2 * M_PI * 3.2 * sy->fr * dt;
    /* BPFO impact train: each impact re-excites a 3.1 kHz resonance */
    if (sy->t >= sy->next_impact_t) {
        sy->env += sy->severity * (0.9 + 0.2 * frand(&sy->rng));
        double bpfo = 3.58 * sy->fr;
        sy->next_impact_t += (1.0 / bpfo) * (1.0 + 0.01 * frand(&sy->rng));
    }
    x += sy->env * sin(sy->res_ph);
    sy->res_ph += 2 * M_PI * 3100.0 * dt;
    sy->env *= exp(-dt / 0.0015);             /* ~1.5 ms ring-down */
    x += 0.05 * frand(&sy->rng);
    sy->t += dt;
    return x;
}

/* ---- sockets ---- */
static int udp_sock(void)
{
    int s = socket(AF_INET, SOCK_DGRAM, 0);
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_BROADCAST, &one, sizeof one);
    return s;
}

static int resolve4(const char *host, struct in_addr *out)
{
    if (inet_pton(AF_INET, host, out) == 1) return 0;
    struct addrinfo hints = {0}, *res = NULL;
    hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, NULL, &hints, &res) != 0 || !res) return -1;
    *out = ((struct sockaddr_in *)res->ai_addr)->sin_addr;
    freeaddrinfo(res);
    return 0;
}

static int tcp_connect(const char *host, int port)
{
    struct sockaddr_in a = {0};
    a.sin_family = AF_INET; a.sin_port = htons((uint16_t)port);
    if (resolve4(host, &a.sin_addr) != 0) return -1;
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (connect(s, (struct sockaddr *)&a, sizeof a) < 0) {
        close(s); return -1;
    }
    int one = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    return s;
}

static int send_all(int fd, const uint8_t *buf, size_t len)
{
    while (len) {
        ssize_t w = send(fd, buf, len, 0);
        if (w < 0) { if (errno == EINTR) continue; return -1; }
        buf += w; len -= (size_t)w;
    }
    return 0;
}

typedef struct { int streaming; uint32_t fs_next; } devstate_t;

static void handle_ctrl(int fd, devstate_t *st, const cfg_t *cfg)
{
    struct pollfd p = { fd, POLLIN, 0 };
    while (poll(&p, 1, 0) > 0 && (p.revents & POLLIN)) {
        uint8_t buf[RS_CTRL_LEN];
        ssize_t r = recv(fd, buf, sizeof buf, MSG_DONTWAIT);
        if (r <= 0) return;
        rs_ctrl_t c;
        if (r == RS_CTRL_LEN && rs_ctrl_parse(buf, (size_t)r, &c) == 0) {
            switch (c.cmd) {
            case RS_C_START: st->streaming = 1; break;
            case RS_C_STOP:  st->streaming = 0; break;
            case RS_C_SET_FS: st->fs_next = c.arg; break;
            case RS_C_IDENT:
                fprintf(stderr, "[%s] IDENT blink\n", cfg->id); break;
            default: break;
            }
        }
    }
}

static void send_beacon(int us, const cfg_t *cfg, uint32_t fs,
                        int streaming, uint32_t drops, int uptime)
{
    rs_beacon_t b = {0};
    b.flags = streaming ? RS_BF_STREAMING : 0;
    b.fw_ver = FW_VER;
    snprintf(b.dev_id, sizeof b.dev_id, "%s", cfg->id);
    b.fs_hz = fs; b.channels = 1;
    b.battery_mv = 3650; b.uptime_s = (uint32_t)uptime; b.drop_total = drops;
    uint8_t buf[RS_BEACON_LEN];
    rs_beacon_encode(&b, buf);
    struct sockaddr_in a = {0};
    a.sin_family = AF_INET; a.sin_port = htons((uint16_t)cfg->bc_port);
    if (resolve4(cfg->bc_host, &a.sin_addr) != 0) return;
    sendto(us, buf, sizeof buf, 0, (struct sockaddr *)&a, sizeof a);
}

int main(int argc, char **argv)
{
    cfg_t cfg = { "RS-000A01", "127.0.0.1", RS_PORT_STREAM,
                  "127.0.0.1", RS_PORT_BEACON,
                  8192, 1000, 2000, 4, 1.0, 0.25, 42, 0 };
    for (int i = 1; i < argc - 1 || (i < argc && !strcmp(argv[i], "--wait-start")); i++) {
        const char *a = argv[i], *v = (i + 1 < argc) ? argv[i + 1] : "";
        if (!strcmp(a, "--id")) cfg.id = v;
        else if (!strcmp(a, "--gateway")) {
            static char host[64]; int port;
            if (sscanf(v, "%63[^:]:%d", host, &port) == 2) {
                cfg.gw_host = host; cfg.gw_port = port; }
        } else if (!strcmp(a, "--beacon")) {
            static char host[64]; int port;
            if (sscanf(v, "%63[^:]:%d", host, &port) == 2) {
                cfg.bc_host = host; cfg.bc_port = port; }
        }
        else if (!strcmp(a, "--fs")) cfg.fs = (uint32_t)atoi(v);
        else if (!strcmp(a, "--acq-ms")) cfg.acq_ms = atoi(v);
        else if (!strcmp(a, "--period-ms")) cfg.period_ms = atoi(v);
        else if (!strcmp(a, "--count")) cfg.count = atoi(v);
        else if (!strcmp(a, "--speed")) cfg.speed = atof(v);
        else if (!strcmp(a, "--growth")) cfg.growth = atof(v);
        else if (!strcmp(a, "--seed")) cfg.seed = (unsigned)atoi(v);
        else if (!strcmp(a, "--wait-start")) cfg.wait_start = 1;
    }

    int us = udp_sock();
    int fd = -1;
    for (int tries = 0; tries < 50 && fd < 0; tries++) {
        fd = tcp_connect(cfg.gw_host, cfg.gw_port);
        if (fd < 0) { send_beacon(us, &cfg, cfg.fs, 0, 0, tries);
                      sleep_us(200000); }
    }
    if (fd < 0) { fprintf(stderr, "gateway unreachable\n"); return 2; }

    devstate_t st = { !cfg.wait_start, cfg.fs };
    const double scale = FULLSCALE_G / 32768.0;
    uint8_t frame[RS_MAX_FRAME];
    uint32_t seq = 0, uptime = 0;
    uint64_t t_started = now_us();

    /* HELLO */
    rs_data_hdr_t h = {0};
    h.type = RS_F_HELLO;
    snprintf(h.dev_id, sizeof h.dev_id, "%s", cfg.id);
    h.fs_hz = cfg.fs; h.scale_g = (float)scale; h.seq = seq++;
    h.t0_us = now_us();
    if (send_all(fd, frame, rs_frame_encode(&h, NULL, frame)) < 0) return 3;

    for (int acq = 0; acq < cfg.count; ) {
        handle_ctrl(fd, &st, &cfg);
        uptime = (uint32_t)((now_us() - t_started) / 1000000ull);
        send_beacon(us, &cfg, cfg.fs, st.streaming, 0, (int)uptime);
        if (!st.streaming) { sleep_us(100000); continue; }
        cfg.fs = st.fs_next;

        synth_t sy;
        synth_init(&sy, cfg.seed + (unsigned)acq * 977u,
                   cfg.growth * (0.4 + 0.6 * acq / (double)cfg.count));
        uint32_t total = (uint32_t)((uint64_t)cfg.fs * (uint64_t)cfg.acq_ms / 1000u);
        uint64_t t0 = now_us();
        int16_t chunk[CHUNK];
        for (uint32_t off = 0; off < total; off += CHUNK) {
            handle_ctrl(fd, &st, &cfg);
            uint16_t n = (uint16_t)((total - off) < CHUNK ? total - off : CHUNK);
            for (uint16_t i = 0; i < n; i++) {
                double g = synth_sample(&sy, (double)cfg.fs);
                double q = g / scale;
                if (q > 32767) q = 32767;
                if (q < -32768) q = -32768;
                chunk[i] = (int16_t)lrint(q);
            }
            h.type = RS_F_DATA; h.n = n; h.seq = seq++;
            h.t0_us = t0 + (uint64_t)(off * 1000000.0 / cfg.fs);
            h.fs_hz = cfg.fs; h.scale_g = (float)scale;
            if (send_all(fd, frame, rs_frame_encode(&h, chunk, frame)) < 0)
                return 3;
            if (cfg.speed > 0)
                sleep_us((uint64_t)(n * 1000000.0 / cfg.fs / cfg.speed));
        }
        /* ACQ_END: payload = total_samples u32, dropped u32 */
        int16_t aux[4];
        uint8_t auxb[8];
        rs_put32(auxb, total); rs_put32(auxb + 4, 0);
        memcpy(aux, auxb, 8);
        h.type = RS_F_ACQ_END; h.n = 4; h.seq = seq++;
        h.t0_us = t0; h.fs_hz = cfg.fs;
        if (send_all(fd, frame, rs_frame_encode(&h, aux, frame)) < 0) return 3;
        fprintf(stderr, "[%s] acq %d/%d sent (%u samples @ %u Hz)\n",
                cfg.id, acq + 1, cfg.count, total, cfg.fs);
        acq++;
        if (acq < cfg.count && cfg.speed > 0)
            sleep_us((uint64_t)(cfg.period_ms * 1000.0 / cfg.speed));
        else if (acq < cfg.count)
            sleep_us(1000);   /* keep cloud timestamps strictly ordered */
    }
    close(fd); close(us);
    return 0;
}
