/* SHA-256 (FIPS 180-4) + HMAC (RFC 2104), pinned by the RFC 4231 test
 * vectors in tests/test_core.c. Bytewise, alignment-safe, ~1.5 KB code. */
#include <string.h>
#include "rs_sha256.h"
#include "rs_proto.h"

static const uint32_t K[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,
    0x923f82a4u,0xab1c5ed5u,0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
    0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,0xe49b69c1u,0xefbe4786u,
    0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,
    0x06ca6351u,0x14292967u,0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
    0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,0xa2bfe8a1u,0xa81a664bu,
    0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,
    0x5b9cca4fu,0x682e6ff3u,0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
    0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u };

static uint32_t rotr(uint32_t x, unsigned n)
{
    return (x >> n) | (x << (32u - n));
}

static void transform(rs_sha256_t *c, const uint8_t p[64])
{
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = (uint32_t)p[4*i] << 24 | (uint32_t)p[4*i+1] << 16 |
               (uint32_t)p[4*i+2] << 8 | (uint32_t)p[4*i+3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a = c->state[0], b = c->state[1], cc = c->state[2],
             d = c->state[3], e = c->state[4], f = c->state[5],
             g = c->state[6], h = c->state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + K[i] + w[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t mj = (a & b) ^ (a & cc) ^ (b & cc);
        uint32_t t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1;
        d = cc; cc = b; b = a; a = t1 + t2;
    }
    c->state[0] += a; c->state[1] += b; c->state[2] += cc; c->state[3] += d;
    c->state[4] += e; c->state[5] += f; c->state[6] += g; c->state[7] += h;
}

void rs_sha256_init(rs_sha256_t *c)
{
    static const uint32_t iv[8] = {
        0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
        0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u };
    memcpy(c->state, iv, sizeof iv);
    c->bitlen = 0; c->buflen = 0;
}

void rs_sha256_update(rs_sha256_t *c, const uint8_t *data, size_t len)
{
    c->bitlen += (uint64_t)len * 8u;
    while (len) {
        size_t take = 64u - c->buflen;
        if (take > len) take = len;
        memcpy(c->buf + c->buflen, data, take);
        c->buflen += take; data += take; len -= take;
        if (c->buflen == 64) { transform(c, c->buf); c->buflen = 0; }
    }
}

void rs_sha256_final(rs_sha256_t *c, uint8_t out[32])
{
    c->buf[c->buflen++] = 0x80;
    if (c->buflen > 56) {
        memset(c->buf + c->buflen, 0, 64u - c->buflen);
        transform(c, c->buf);
        c->buflen = 0;
    }
    memset(c->buf + c->buflen, 0, 56u - c->buflen);
    for (int i = 0; i < 8; i++)
        c->buf[56 + i] = (uint8_t)(c->bitlen >> (56 - 8 * i));
    transform(c, c->buf);
    for (int i = 0; i < 8; i++) {
        out[4*i]   = (uint8_t)(c->state[i] >> 24);
        out[4*i+1] = (uint8_t)(c->state[i] >> 16);
        out[4*i+2] = (uint8_t)(c->state[i] >> 8);
        out[4*i+3] = (uint8_t)(c->state[i]);
    }
}

void rs_sha256(const uint8_t *data, size_t len, uint8_t out[32])
{
    rs_sha256_t c;
    rs_sha256_init(&c);
    rs_sha256_update(&c, data, len);
    rs_sha256_final(&c, out);
}

void rs_hmac_sha256(const uint8_t *key, size_t keylen,
                    const uint8_t *msg, size_t msglen, uint8_t out[32])
{
    uint8_t k[64] = {0}, pad[64], inner[32];
    if (keylen > 64) rs_sha256(key, keylen, k);
    else memcpy(k, key, keylen);
    rs_sha256_t c;
    for (int i = 0; i < 64; i++) pad[i] = (uint8_t)(k[i] ^ 0x36u);
    rs_sha256_init(&c);
    rs_sha256_update(&c, pad, 64);
    rs_sha256_update(&c, msg, msglen);
    rs_sha256_final(&c, inner);
    for (int i = 0; i < 64; i++) pad[i] = (uint8_t)(k[i] ^ 0x5cu);
    rs_sha256_init(&c);
    rs_sha256_update(&c, pad, 64);
    rs_sha256_update(&c, inner, 32);
    rs_sha256_final(&c, out);
}

void rs_auth_tag(const uint8_t *key, size_t keylen, const char *dev_id,
                 uint32_t nonce, uint8_t out[RS_AUTH_TAG_LEN])
{
    uint8_t msg[RS_DEV_ID_LEN + 4] = {0};
    for (int i = 0; i < RS_DEV_ID_LEN && dev_id[i]; i++)
        msg[i] = (uint8_t)dev_id[i];
    rs_put32(msg + RS_DEV_ID_LEN, nonce);
    uint8_t mac[32];
    rs_hmac_sha256(key, keylen, msg, sizeof msg, mac);
    memcpy(out, mac, RS_AUTH_TAG_LEN);
}
