/* SHA-256 + HMAC-SHA256 for RSP/1 device authentication. Pure C99,
 * no allocation, no deps — small enough for any MCU port. */
#ifndef RS_SHA256_H
#define RS_SHA256_H
#include <stdint.h>
#include <stddef.h>

typedef struct {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t  buf[64];
    size_t   buflen;
} rs_sha256_t;

void rs_sha256_init(rs_sha256_t *c);
void rs_sha256_update(rs_sha256_t *c, const uint8_t *data, size_t len);
void rs_sha256_final(rs_sha256_t *c, uint8_t out[32]);
/* one-shot */
void rs_sha256(const uint8_t *data, size_t len, uint8_t out[32]);

void rs_hmac_sha256(const uint8_t *key, size_t keylen,
                    const uint8_t *msg, size_t msglen, uint8_t out[32]);

/* RSP/1 auth tag: HMAC-SHA256(key, dev_id[12, NUL-padded] || nonce LE32)
 * truncated to 16 bytes — the F_AUTH frame payload. */
#define RS_AUTH_TAG_LEN 16
void rs_auth_tag(const uint8_t *key, size_t keylen, const char *dev_id,
                 uint32_t nonce, uint8_t out[RS_AUTH_TAG_LEN]);

#endif
