/* ESP32-S3 port of the relspec sensor firmware (reference target).
 *
 * Hardware reference design (see docs/EDGE_FIRMWARE.md for selection):
 *   - MCU     ESP32-S3-WROOM-1 (dual LX7 @240 MHz, 512 KB SRAM + 8 MB
 *             octal PSRAM, Wi-Fi b/g/n) — PSRAM holds the sample ring
 *   - sensor  TDK IIM-42352 (±16 g digital MEMS, 4 kHz flat band,
 *             2 KB FIFO, SPI @ 24 MHz) for the wireless node
 *   - alt.    ADXL1002 (±50 g analog, 11 kHz) + ADS127L01 24-bit ADC
 *             via I2S for the wired high-bandwidth variant
 *
 * Task layout (FreeRTOS):
 *   sample_task  (core 1, prio 20)  SPI DMA burst reads of the IIM-42352
 *                                   FIFO -> rs_ring (PSRAM, 256 K samples
 *                                   = 32 s @ 8 kHz headroom)
 *   stream_task  (core 0, prio 10)  pops CHUNK samples, rs_frame_encode,
 *                                   TCP send to the gateway; on WiFi loss
 *                                   the ring absorbs, drops counted
 *   beacon_task  (core 0, prio  5)  RSP/1 UDP broadcast every 5 s
 *   ctrl         (stream socket)    START/STOP/SET_FS/IDENT/REBOOT
 *
 * This file is a compile-ready skeleton: the RSP/1 core (rs_proto,
 * rs_ring) is identical to the host-tested code; only the HAL calls
 * (SPI, Wi-Fi, esp_timer) bind it to the S3. Build with ESP-IDF >= 5.2:
 *   idf.py set-target esp32s3 && idf.py build flash monitor
 */
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "lwip/sockets.h"
#include "rs_proto.h"
#include "rs_ring.h"

#define CHUNK 1024
#define RING_SAMPLES (256 * 1024)          /* PSRAM */
#define FS_DEFAULT 8000
#define SCALE_G (16.0f / 32768.0f)         /* IIM-42352 ±16 g */

static rs_ring_t s_ring;
static int16_t *s_ring_mem;                /* heap_caps_malloc SPIRAM */
static volatile uint32_t s_fs = FS_DEFAULT;
static volatile bool s_streaming = true;

/* ---- sample_task: IIM-42352 FIFO -> ring ------------------------------ */
static void sample_task(void *arg)
{
    (void)arg;
    /* iim_init(SPI3_HOST, s_fs);  -- configure ODR, ±16 g, FIFO stream */
    int16_t burst[512];
    for (;;) {
        /* int n = iim_read_fifo(burst, 512);  -- DMA burst, ~1 kHz walk */
        int n = 0; /* HAL stub */
        if (n > 0 && s_streaming)
            rs_ring_push(&s_ring, burst, (uint32_t)n);
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

/* ---- stream_task: ring -> RSP/1 TCP ----------------------------------- */
static void stream_task(void *arg)
{
    const char *gw_host = CONFIG_RELSPEC_GATEWAY_HOST;
    static uint8_t frame[RS_MAX_FRAME];
    static int16_t chunk[CHUNK];
    rs_data_hdr_t h = { .type = RS_F_DATA, .fs_hz = FS_DEFAULT,
                        .scale_g = SCALE_G };
    strlcpy(h.dev_id, CONFIG_RELSPEC_DEVICE_ID, sizeof h.dev_id);
    (void)arg;
    for (;;) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in a = { .sin_family = AF_INET,
                                 .sin_port = htons(RS_PORT_STREAM) };
        inet_pton(AF_INET, gw_host, &a.sin_addr);
        if (connect(fd, (struct sockaddr *)&a, sizeof a) != 0) {
            close(fd); vTaskDelay(pdMS_TO_TICKS(2000)); continue;
        }
        h.type = RS_F_HELLO; h.n = 0;
        h.t0_us = (uint64_t)esp_timer_get_time();
        send(fd, frame, rs_frame_encode(&h, NULL, frame), 0);
        h.type = RS_F_DATA;
        while (s_streaming) {
            uint32_t n = rs_ring_pop(&s_ring, chunk, CHUNK);
            if (!n) { vTaskDelay(pdMS_TO_TICKS(10)); continue; }
            h.n = (uint16_t)n; h.seq++; h.fs_hz = s_fs;
            h.t0_us = (uint64_t)esp_timer_get_time();
            if (send(fd, frame, rs_frame_encode(&h, chunk, frame), 0) < 0)
                break;                     /* reconnect; ring buffers */
            /* ctrl frames arrive on the same socket: poll + rs_ctrl_parse */
        }
        close(fd);
    }
}

/* ---- beacon_task ------------------------------------------------------ */
static void beacon_task(void *arg)
{
    (void)arg;
    int us = socket(AF_INET, SOCK_DGRAM, 0);
    int yes = 1;
    setsockopt(us, SOL_SOCKET, SO_BROADCAST, &yes, sizeof yes);
    struct sockaddr_in a = { .sin_family = AF_INET,
                             .sin_port = htons(RS_PORT_BEACON),
                             .sin_addr.s_addr = htonl(INADDR_BROADCAST) };
    rs_beacon_t b = { .fw_ver = 0x0100, .channels = 1 };
    strlcpy(b.dev_id, CONFIG_RELSPEC_DEVICE_ID, sizeof b.dev_id);
    uint8_t buf[RS_BEACON_LEN];
    for (;;) {
        b.flags = s_streaming ? RS_BF_STREAMING : 0;
        b.fs_hz = s_fs;
        b.battery_mv = 3650;               /* adc_oneshot on VBAT divider */
        b.uptime_s = (uint32_t)(esp_timer_get_time() / 1000000ULL);
        b.drop_total = s_ring.drops;
        rs_beacon_encode(&b, buf);
        sendto(us, buf, sizeof buf, 0, (struct sockaddr *)&a, sizeof a);
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}

void app_main(void)
{
    nvs_flash_init();
    /* wifi_connect(CONFIG_RELSPEC_WIFI_SSID, CONFIG_RELSPEC_WIFI_PASS); */
    s_ring_mem = heap_caps_malloc(RING_SAMPLES * sizeof(int16_t),
                                  MALLOC_CAP_SPIRAM);
    rs_ring_init(&s_ring, s_ring_mem, RING_SAMPLES);
    xTaskCreatePinnedToCore(sample_task, "sample", 4096, NULL, 20, NULL, 1);
    xTaskCreatePinnedToCore(stream_task, "stream", 8192, NULL, 10, NULL, 0);
    xTaskCreatePinnedToCore(beacon_task, "beacon", 4096, NULL, 5, NULL, 0);
}
