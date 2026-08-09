/* SPSC sample ring buffer: ISR/DMA producer, task consumer.
 *
 * Capacity must be a power of two. On overflow the NEW samples are
 * rejected and counted — an acquisition window with a gap is worthless
 * for spectral work, so we keep what is contiguous and let the consumer
 * see `drops` grow (it aborts and restarts the window).
 *
 * head/tail are free-running uint32 indices; with power-of-two capacity
 * the difference is always the fill level even across wrap.
 */
#ifndef RS_RING_H
#define RS_RING_H
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

typedef struct {
    int16_t *buf;
    uint32_t mask;          /* capacity - 1 */
    volatile uint32_t head; /* producer writes */
    volatile uint32_t tail; /* consumer writes */
    volatile uint32_t drops;
} rs_ring_t;

/* cap must be a power of two >= 2; returns false otherwise */
bool rs_ring_init(rs_ring_t *r, int16_t *storage, uint32_t cap);
static inline uint32_t rs_ring_count(const rs_ring_t *r){
    return r->head - r->tail; }
static inline uint32_t rs_ring_free(const rs_ring_t *r){
    return (r->mask + 1u) - rs_ring_count(r); }

/* producer side: push n samples; returns samples accepted (all-or-none
 * per call so a DMA block never lands half-in). Rejected -> drops += n. */
uint32_t rs_ring_push(rs_ring_t *r, const int16_t *src, uint32_t n);

/* consumer side: pop up to n samples into dst, returns popped count */
uint32_t rs_ring_pop(rs_ring_t *r, int16_t *dst, uint32_t n);

/* consumer: discard everything currently buffered */
void rs_ring_flush(rs_ring_t *r);

#endif
