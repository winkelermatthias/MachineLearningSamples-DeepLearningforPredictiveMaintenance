/* relspec stream protocol (RSP/1) — wire format shared by firmware,
 * simulator and gateway.
 *
 * Everything is little-endian and byte-serialized by hand (no packed
 * structs on the wire), so the core is portable across MCUs regardless
 * of alignment rules. Three PDU families:
 *
 *   beacon  (UDP, port 47700)  device announce, 40 bytes
 *   data    (TCP, port 47701)  HELLO / DATA / ACQ_END / STATUS frames
 *   control (TCP, gateway->device) 16-byte commands
 *
 * CRC-32 (IEEE, reflected) over every PDU excluding its trailing crc.
 */
#ifndef RS_PROTO_H
#define RS_PROTO_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define RS_PORT_BEACON 47700
#define RS_PORT_STREAM 47701

#define RS_MAGIC_BEACON 0x31425352u /* "RSB1" */
#define RS_MAGIC_DATA   0x31445352u /* "RSD1" */
#define RS_MAGIC_CTRL   0x31435352u /* "RSC1" */

#define RS_PROTO_VER 1
#define RS_DEV_ID_LEN 12

/* data frame types. F_AUTH answers a CHALLENGE: payload is
 * HMAC-SHA256(device_key, dev_id[12] || nonce LE32) truncated to
 * 16 bytes (n = 8 words). See rs_sha256.h / rs_auth_tag(). */
enum { RS_F_HELLO = 1, RS_F_DATA = 2, RS_F_ACQ_END = 3, RS_F_STATUS = 4,
       RS_F_AUTH = 5 };

/* control commands. CHALLENGE: arg = random nonce the gateway remembers;
 * a keyed device must answer with F_AUTH before DATA is accepted. */
enum { RS_C_START = 1, RS_C_STOP = 2, RS_C_SET_FS = 3,
       RS_C_SET_ACQ_MS = 4, RS_C_IDENT = 5, RS_C_REBOOT = 6,
       RS_C_CHALLENGE = 7 };

/* beacon flags */
#define RS_BF_STREAMING 0x01
#define RS_BF_FAULT     0x02   /* device-side self-test failed */

#define RS_BEACON_LEN 40
#define RS_DATA_HDR_LEN 40
#define RS_CTRL_LEN 16
#define RS_MAX_SAMPLES 1024    /* per DATA frame */
#define RS_MAX_FRAME (RS_DATA_HDR_LEN + 2*RS_MAX_SAMPLES + 4)

typedef struct {
    uint8_t  ver, flags;
    uint16_t fw_ver;
    char     dev_id[RS_DEV_ID_LEN + 1]; /* NUL-terminated on parse */
    uint32_t fs_hz;
    uint8_t  channels;
    uint16_t battery_mv;
    uint32_t uptime_s;
    uint32_t drop_total;
} rs_beacon_t;

typedef struct {
    uint8_t  type;
    uint16_t n;                 /* payload length in int16 words */
    uint32_t seq;
    char     dev_id[RS_DEV_ID_LEN + 1];
    uint64_t t0_us;             /* first-sample timestamp, unix µs */
    uint32_t fs_hz;
    float    scale_g;           /* g per LSB for int16 payload */
} rs_data_hdr_t;

typedef struct {
    uint8_t  cmd;
    uint32_t arg;
} rs_ctrl_t;

/* ---- little-endian helpers ---- */
static inline void rs_put16(uint8_t *p, uint16_t v){
    p[0]=(uint8_t)(v&0xff); p[1]=(uint8_t)(v>>8); }
static inline void rs_put32(uint8_t *p, uint32_t v){
    p[0]=(uint8_t)(v&0xff); p[1]=(uint8_t)((v>>8)&0xff);
    p[2]=(uint8_t)((v>>16)&0xff); p[3]=(uint8_t)(v>>24); }
static inline void rs_put64(uint8_t *p, uint64_t v){
    rs_put32(p, (uint32_t)v); rs_put32(p+4, (uint32_t)(v>>32)); }
static inline uint16_t rs_get16(const uint8_t *p){
    return (uint16_t)(p[0] | (uint16_t)p[1]<<8); }
static inline uint32_t rs_get32(const uint8_t *p){
    return p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24; }
static inline uint64_t rs_get64(const uint8_t *p){
    return rs_get32(p) | (uint64_t)rs_get32(p+4)<<32; }

uint32_t rs_crc32(const uint8_t *data, size_t len);

/* beacon: returns bytes written (RS_BEACON_LEN) */
size_t rs_beacon_encode(const rs_beacon_t *b, uint8_t out[RS_BEACON_LEN]);
/* returns 0 on success, -1 on bad magic/len/crc */
int rs_beacon_parse(const uint8_t *buf, size_t len, rs_beacon_t *out);

/* data frame: payload may be NULL when hdr->n == 0.
 * out must hold RS_DATA_HDR_LEN + 2*hdr->n + 4 bytes; returns total. */
size_t rs_frame_encode(const rs_data_hdr_t *hdr, const int16_t *payload,
                       uint8_t *out);
/* parses header only and verifies total-frame CRC; payload pointer is
 * returned into the caller's buffer (aliased). 0 ok, -1 bad. */
int rs_frame_parse(const uint8_t *buf, size_t len,
                   rs_data_hdr_t *hdr, const uint8_t **payload);
/* how many bytes a full frame with this header occupies */
static inline size_t rs_frame_len(uint16_t n){
    return RS_DATA_HDR_LEN + 2u*n + 4u; }

size_t rs_ctrl_encode(const rs_ctrl_t *c, uint8_t out[RS_CTRL_LEN]);
int rs_ctrl_parse(const uint8_t *buf, size_t len, rs_ctrl_t *out);

#endif
