/* Host-side unit tests for the portable firmware core. Plain asserts;
 * exit 0 = green. Run via `make test`. */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "../core/rs_proto.h"
#include "../core/rs_ring.h"
#include "../core/rs_sha256.h"

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

static void hex_eq(const uint8_t *got, const char *want_hex, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        unsigned b;
        assert(sscanf(want_hex + 2 * i, "%2x", &b) == 1);
        assert(got[i] == (uint8_t)b);
    }
}

static void test_sha256(void)
{
    uint8_t d[32];
    rs_sha256((const uint8_t *)"abc", 3, d);
    hex_eq(d, "ba7816bf8f01cfea414140de5dae2223"
              "b00361a396177a9cb410ff61f20015ad", 32);
    rs_sha256((const uint8_t *)"", 0, d);
    hex_eq(d, "e3b0c44298fc1c149afbf4c8996fb924"
              "27ae41e4649b934ca495991b7852b855", 32);
    /* multi-block + streamed update must agree with one-shot */
    uint8_t big[200];
    for (int i = 0; i < 200; i++) big[i] = (uint8_t)(i * 7);
    rs_sha256(big, sizeof big, d);
    uint8_t d2[32];
    rs_sha256_t c;
    rs_sha256_init(&c);
    rs_sha256_update(&c, big, 63);
    rs_sha256_update(&c, big + 63, 137);
    rs_sha256_final(&c, d2);
    assert(memcmp(d, d2, 32) == 0);
}

static void test_hmac(void)
{
    /* RFC 4231 test cases 1, 2 and 6 (key longer than block size) */
    uint8_t key[131], mac[32];
    memset(key, 0x0b, 20);
    rs_hmac_sha256(key, 20, (const uint8_t *)"Hi There", 8, mac);
    hex_eq(mac, "b0344c61d8db38535ca8afceaf0bf12b"
                "881dc200c9833da726e9376c2e32cff7", 32);
    rs_hmac_sha256((const uint8_t *)"Jefe", 4,
                   (const uint8_t *)"what do ya want for nothing?", 28, mac);
    hex_eq(mac, "5bdcc146bf60754e6a042426089575c7"
                "5a003f089d2739839dec58b964ec3843", 32);
    memset(key, 0xaa, 131);
    rs_hmac_sha256(key, 131, (const uint8_t *)
                   "Test Using Larger Than Block-Size Key - Hash Key First",
                   54, mac);
    hex_eq(mac, "60e431591ee0b67f0d8a26aacbf5b77f"
                "8e0bc6213728c5140546040f0ee37f54", 32);
}

static void test_auth_frame(void)
{
    /* tag pinned against Python: hmac(key=0x0b*20,
     * "RS-000TEST\0\0" + LE32(0xA1B2C3D4), sha256)[:16] */
    uint8_t key[20]; memset(key, 0x0b, sizeof key);
    uint8_t tag[RS_AUTH_TAG_LEN];
    rs_auth_tag(key, sizeof key, "RS-000TEST", 0xA1B2C3D4u, tag);
    hex_eq(tag, "0851b0eb205494213db7ef5d6d050849", RS_AUTH_TAG_LEN);
    /* CHALLENGE control roundtrip carries the nonce */
    rs_ctrl_t c = { RS_C_CHALLENGE, 0xA1B2C3D4u }, cout;
    uint8_t cbuf[RS_CTRL_LEN];
    rs_ctrl_encode(&c, cbuf);
    assert(rs_ctrl_parse(cbuf, sizeof cbuf, &cout) == 0);
    assert(cout.cmd == RS_C_CHALLENGE && cout.arg == 0xA1B2C3D4u);
    /* F_AUTH frame roundtrip: tag survives the int16 payload path */
    int16_t pay[RS_AUTH_TAG_LEN / 2];
    memcpy(pay, tag, RS_AUTH_TAG_LEN);
    rs_data_hdr_t h = {0}, out;
    h.type = RS_F_AUTH; h.n = RS_AUTH_TAG_LEN / 2;
    strcpy(h.dev_id, "RS-000TEST");
    uint8_t buf[RS_DATA_HDR_LEN + RS_AUTH_TAG_LEN + 4];
    size_t len = rs_frame_encode(&h, pay, buf);
    assert(len == rs_frame_len(RS_AUTH_TAG_LEN / 2));
    const uint8_t *p;
    assert(rs_frame_parse(buf, len, &out, &p) == 0);
    assert(out.type == RS_F_AUTH && out.n == RS_AUTH_TAG_LEN / 2);
    assert(memcmp(p, tag, RS_AUTH_TAG_LEN) == 0);
    /* a wrong key must not produce the same tag */
    key[0] ^= 1;
    uint8_t bad[RS_AUTH_TAG_LEN];
    rs_auth_tag(key, sizeof key, "RS-000TEST", 0xA1B2C3D4u, bad);
    assert(memcmp(bad, tag, RS_AUTH_TAG_LEN) != 0);
}

int main(void)
{
    test_crc();
    test_beacon();
    test_frame();
    test_ctrl();
    test_ring();
    test_sha256();
    test_hmac();
    test_auth_frame();
    printf("firmware core: all tests passed\n");
    return 0;
}
