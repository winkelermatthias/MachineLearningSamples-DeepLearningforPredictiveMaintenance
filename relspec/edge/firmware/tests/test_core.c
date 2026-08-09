/* Host-side unit tests for the portable firmware core. Plain asserts;
 * exit 0 = green. Run via `make test`. */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "../core/rs_proto.h"
#include "../core/rs_ring.h"

static void test_crc(void)
{
    /* canonical IEEE CRC-32 check vector */
    assert(rs_crc32((const uint8_t *)"123456789", 9) == 0xCBF43926u);
    assert(rs_crc32((const uint8_t *)"", 0) == 0x00000000u);
}

static void test_beacon(void)
{
    rs_beacon_t b = {0}, out;
    b.flags = RS_BF_STREAMING;
    b.fw_ver = 0x0102;
    strcpy(b.dev_id, "RS-A1B2C3");
    b.fs_hz = 25600; b.channels = 1; b.battery_mv = 3712;
    b.uptime_s = 86400; b.drop_total = 7;
    uint8_t buf[RS_BEACON_LEN];
    assert(rs_beacon_encode(&b, buf) == RS_BEACON_LEN);
    assert(rs_beacon_parse(buf, sizeof buf, &out) == 0);
    assert(out.ver == RS_PROTO_VER);
    assert(out.flags == RS_BF_STREAMING);
    assert(out.fw_ver == 0x0102);
    assert(strcmp(out.dev_id, "RS-A1B2C3") == 0);
    assert(out.fs_hz == 25600 && out.channels == 1);
    assert(out.battery_mv == 3712 && out.uptime_s == 86400);
    assert(out.drop_total == 7);
    buf[9] ^= 0x40;                       /* corrupt one id byte */
    assert(rs_beacon_parse(buf, sizeof buf, &out) == -1);
    assert(rs_beacon_parse(buf, 10, &out) == -1);
}

static void test_frame(void)
{
    int16_t pay[256];
    for (int i = 0; i < 256; i++) pay[i] = (int16_t)(i * 37 - 4000);
    rs_data_hdr_t h = {0}, out;
    h.type = RS_F_DATA; h.n = 256; h.seq = 99;
    strcpy(h.dev_id, "RS-000TEST");
    h.t0_us = 1765432100123456ull;
    h.fs_hz = 12800; h.scale_g = 0.000598f;
    uint8_t buf[RS_MAX_FRAME];
    size_t len = rs_frame_encode(&h, pay, buf);
    assert(len == rs_frame_len(256));
    const uint8_t *p;
    assert(rs_frame_parse(buf, len, &out, &p) == 0);
    assert(out.type == RS_F_DATA && out.n == 256 && out.seq == 99);
    assert(strcmp(out.dev_id, "RS-000TEST") == 0);
    assert(out.t0_us == 1765432100123456ull);
    assert(out.fs_hz == 12800);
    assert(out.scale_g > 0.000597f && out.scale_g < 0.000599f);
    for (int i = 0; i < 256; i++)
        assert((int16_t)rs_get16(p + 2*i) == pay[i]);
    /* truncated + corrupted must both fail */
    assert(rs_frame_parse(buf, len - 1, &out, &p) == -1);
    buf[50] ^= 1;
    assert(rs_frame_parse(buf, len, &out, &p) == -1);
    /* zero-payload frame (HELLO) */
    rs_data_hdr_t hello = {0};
    hello.type = RS_F_HELLO; strcpy(hello.dev_id, "RS-000TEST");
    len = rs_frame_encode(&hello, NULL, buf);
    assert(len == rs_frame_len(0));
    assert(rs_frame_parse(buf, len, &out, &p) == 0 && out.type == RS_F_HELLO);
}

static void test_ctrl(void)
{
    rs_ctrl_t c = { RS_C_SET_FS, 25600 }, out;
    uint8_t buf[RS_CTRL_LEN];
    assert(rs_ctrl_encode(&c, buf) == RS_CTRL_LEN);
    assert(rs_ctrl_parse(buf, sizeof buf, &out) == 0);
    assert(out.cmd == RS_C_SET_FS && out.arg == 25600);
    buf[8] ^= 0xFF;
    assert(rs_ctrl_parse(buf, sizeof buf, &out) == -1);
}

static void test_ring(void)
{
    int16_t store[8], tmp[8];
    rs_ring_t r;
    assert(!rs_ring_init(&r, store, 6));     /* not a power of two */
    assert(rs_ring_init(&r, store, 8));
    int16_t a[5] = {1,2,3,4,5};
    assert(rs_ring_push(&r, a, 5) == 5);
    assert(rs_ring_count(&r) == 5 && rs_ring_free(&r) == 3);
    /* all-or-none: a 4-sample block into 3 free slots is rejected */
    assert(rs_ring_push(&r, a, 4) == 0);
    assert(r.drops == 4);
    assert(rs_ring_pop(&r, tmp, 3) == 3);
    assert(tmp[0] == 1 && tmp[2] == 3);
    /* wrap-around correctness across the boundary */
    assert(rs_ring_push(&r, a, 5) == 5);
    assert(rs_ring_count(&r) == 7);
    assert(rs_ring_pop(&r, tmp, 8) == 7);
    assert(tmp[0] == 4 && tmp[1] == 5 && tmp[2] == 1 && tmp[6] == 5);
    /* free-running indices survive many wraps */
    for (int k = 0; k < 10000; k++) {
        assert(rs_ring_push(&r, a, 5) == 5);
        assert(rs_ring_pop(&r, tmp, 5) == 5);
        assert(tmp[4] == 5);
    }
    rs_ring_flush(&r);
    assert(rs_ring_count(&r) == 0);
}

int main(void)
{
    test_crc();
    test_beacon();
    test_frame();
    test_ctrl();
    test_ring();
    printf("firmware core: all tests passed\n");
    return 0;
}
