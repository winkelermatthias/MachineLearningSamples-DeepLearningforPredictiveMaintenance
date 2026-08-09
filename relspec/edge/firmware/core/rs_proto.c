/* RSP/1 serialization + CRC-32. Pure C99, no allocation, no OS calls. */
#include "rs_proto.h"

uint32_t rs_crc32(const uint8_t *data, size_t len)
{
    /* IEEE 802.3 reflected, bitwise (small, fast enough for 2 KB frames
     * even at 240 MHz MCU clocks; swap in a table if profiling says so) */
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int b = 0; b < 8; b++)
            crc = (crc >> 1) ^ (0xEDB88320u & (~(crc & 1u) + 1u));
    }
    return ~crc;
}

static void put_id(uint8_t *p, const char *id)
{
    memset(p, 0, RS_DEV_ID_LEN);
    for (int i = 0; i < RS_DEV_ID_LEN && id[i]; i++) p[i] = (uint8_t)id[i];
}

static void get_id(char *out, const uint8_t *p)
{
    memcpy(out, p, RS_DEV_ID_LEN);
    out[RS_DEV_ID_LEN] = '\0';
}

/* ---------------- beacon ---------------- */
size_t rs_beacon_encode(const rs_beacon_t *b, uint8_t out[RS_BEACON_LEN])
{
    rs_put32(out + 0, RS_MAGIC_BEACON);
    out[4] = RS_PROTO_VER;
    out[5] = b->flags;
    rs_put16(out + 6, b->fw_ver);
    put_id(out + 8, b->dev_id);
    rs_put32(out + 20, b->fs_hz);
    out[24] = b->channels;
    out[25] = 0;
    rs_put16(out + 26, b->battery_mv);
    rs_put32(out + 28, b->uptime_s);
    rs_put32(out + 32, b->drop_total);
    rs_put32(out + 36, rs_crc32(out, 36));
    return RS_BEACON_LEN;
}

int rs_beacon_parse(const uint8_t *buf, size_t len, rs_beacon_t *out)
{
    if (len != RS_BEACON_LEN) return -1;
    if (rs_get32(buf) != RS_MAGIC_BEACON) return -1;
    if (rs_get32(buf + 36) != rs_crc32(buf, 36)) return -1;
    out->ver = buf[4];
    out->flags = buf[5];
    out->fw_ver = rs_get16(buf + 6);
    get_id(out->dev_id, buf + 8);
    out->fs_hz = rs_get32(buf + 20);
    out->channels = buf[24];
    out->battery_mv = rs_get16(buf + 26);
    out->uptime_s = rs_get32(buf + 28);
    out->drop_total = rs_get32(buf + 32);
    return 0;
}

/* ---------------- data frame ---------------- */
size_t rs_frame_encode(const rs_data_hdr_t *hdr, const int16_t *payload,
                       uint8_t *out)
{
    rs_put32(out + 0, RS_MAGIC_DATA);
    out[4] = hdr->type;
    out[5] = 0;
    rs_put16(out + 6, hdr->n);
    rs_put32(out + 8, hdr->seq);
    put_id(out + 12, hdr->dev_id);
    rs_put64(out + 24, hdr->t0_us);
    rs_put32(out + 32, hdr->fs_hz);
    uint32_t sb; memcpy(&sb, &hdr->scale_g, 4);
    rs_put32(out + 36, sb);
    for (uint16_t i = 0; i < hdr->n; i++)
        rs_put16(out + RS_DATA_HDR_LEN + 2u*i, (uint16_t)payload[i]);
    size_t body = RS_DATA_HDR_LEN + 2u*hdr->n;
    rs_put32(out + body, rs_crc32(out, body));
    return body + 4;
}

int rs_frame_parse(const uint8_t *buf, size_t len,
                   rs_data_hdr_t *hdr, const uint8_t **payload)
{
    if (len < RS_DATA_HDR_LEN + 4) return -1;
    if (rs_get32(buf) != RS_MAGIC_DATA) return -1;
    uint16_t n = rs_get16(buf + 6);
    if (n > RS_MAX_SAMPLES) return -1;
    size_t total = rs_frame_len(n);
    if (len < total) return -1;
    size_t body = total - 4;
    if (rs_get32(buf + body) != rs_crc32(buf, body)) return -1;
    hdr->type = buf[4];
    hdr->n = n;
    hdr->seq = rs_get32(buf + 8);
    get_id(hdr->dev_id, buf + 12);
    hdr->t0_us = rs_get64(buf + 24);
    hdr->fs_hz = rs_get32(buf + 32);
    uint32_t sb = rs_get32(buf + 36); memcpy(&hdr->scale_g, &sb, 4);
    if (payload) *payload = buf + RS_DATA_HDR_LEN;
    return 0;
}

/* ---------------- control ---------------- */
size_t rs_ctrl_encode(const rs_ctrl_t *c, uint8_t out[RS_CTRL_LEN])
{
    rs_put32(out + 0, RS_MAGIC_CTRL);
    out[4] = c->cmd;
    out[5] = out[6] = out[7] = 0;
    rs_put32(out + 8, c->arg);
    rs_put32(out + 12, rs_crc32(out, 12));
    return RS_CTRL_LEN;
}

int rs_ctrl_parse(const uint8_t *buf, size_t len, rs_ctrl_t *out)
{
    if (len != RS_CTRL_LEN) return -1;
    if (rs_get32(buf) != RS_MAGIC_CTRL) return -1;
    if (rs_get32(buf + 12) != rs_crc32(buf, 12)) return -1;
    out->cmd = buf[4];
    out->arg = rs_get32(buf + 8);
    return 0;
}
