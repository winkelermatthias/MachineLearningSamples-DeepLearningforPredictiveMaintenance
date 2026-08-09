#include "rs_ring.h"

bool rs_ring_init(rs_ring_t *r, int16_t *storage, uint32_t cap)
{
    if (cap < 2 || (cap & (cap - 1))) return false;
    r->buf = storage;
    r->mask = cap - 1;
    r->head = r->tail = r->drops = 0;
    return true;
}

uint32_t rs_ring_push(rs_ring_t *r, const int16_t *src, uint32_t n)
{
    if (n > rs_ring_free(r)) { r->drops += n; return 0; }
    uint32_t h = r->head;
    for (uint32_t i = 0; i < n; i++)
        r->buf[(h + i) & r->mask] = src[i];
    r->head = h + n;   /* single store publishes the block */
    return n;
}

uint32_t rs_ring_pop(rs_ring_t *r, int16_t *dst, uint32_t n)
{
    uint32_t avail = rs_ring_count(r);
    if (n > avail) n = avail;
    uint32_t t = r->tail;
    for (uint32_t i = 0; i < n; i++)
        dst[i] = r->buf[(t + i) & r->mask];
    r->tail = t + n;
    return n;
}

void rs_ring_flush(rs_ring_t *r) { r->tail = r->head; }
