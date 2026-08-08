/* edge_policy.h - operating-state discovery, cloud config shadow, upload
 * queue and connection batching.
 *
 * Layering: edge_gate.{h,c} is DSP + codec. This file is policy. Everything
 * here is deterministic, allocation-free, and safe to run on an MCU.
 */
#ifndef EDGE_POLICY_H
#define EDGE_POLICY_H

#include "edge_gate.h"

/* =====================================================================
 * 1. OPERATING STATE DISCOVERY
 *
 * The device learns the idle/running split locally with no labels and no
 * commissioning step. Method: a decaying 64-bin histogram of log level, with
 * Otsu's threshold recomputed periodically.
 *
 * Otsu is the right tool here: O(bins), deterministic, no tuning constants,
 * and it returns a separability figure that doubles as a validity test.
 * A continuously-running machine is unimodal, separability stays low, and the
 * device correctly refuses to invent an idle state. That failure mode matters
 * more than the detection: wrongly declaring RUNNING costs a little battery,
 * wrongly declaring IDLE silently drops the data you exist to collect.
 * ===================================================================== */
#define EP_HIST_BINS   64
#define EP_HIST_LO_DB  (-80.0f)      /* 1e-4 g  */
#define EP_HIST_HI_DB  (20.0f)       /* 10 g    */
#define EP_HIST_CAP    4096          /* halve all bins above this total */
#define EP_AMBIG_DB    12.0f         /* band above thr where structure vetoes */

typedef enum {
    EP_STATE_UNKNOWN = 0,   /* discovery not yet valid                     */
    EP_STATE_IDLE,          /* below learned threshold: drop entirely      */
    EP_STATE_RUNNING,       /* above threshold: process normally           */
    EP_STATE_TRANSIENT      /* level says running, structure disagrees     */
} ep_state_t;

typedef struct {
    uint16_t hist[EP_HIST_BINS];
    uint32_t n_total;
    float    thr_db;          /* learned Otsu threshold, dB re 1 g         */
    float    sep;             /* Otsu separability, 0..1                   */
    float    mu_idle_db, mu_run_db;
    float    w_idle, w_run;
    bool     valid;           /* discovery has converged and passed checks  */
    ep_state_t state;
    ep_state_t state_prev;
    uint8_t  pend_ring;       /* persistence bits for state switching      */
    uint32_t n_idle, n_run, n_transient;
    uint32_t last_fit_n;
} ep_discovery_t;

/* =====================================================================
 * 2. CLOUD CONFIG SHADOW
 * Desired state pushed down; applied_version echoed up. Anything the fleet
 * operator may need to change without a firmware release lives here.
 * ===================================================================== */
typedef struct {
    uint32_t version;              /* monotonically increasing             */

    /* --- duty cycle --- */
    uint32_t wake_interval_s;      /* acquisition cadence                  */
    uint32_t heartbeat_s;          /* condition 2                          */

    /* --- transmission budget --- */
    uint16_t upload_max_per_window;/* hard cap on frames                   */
    uint32_t rate_window_s;        /* the window the cap applies over      */
    uint8_t  upload_reserve_change;/* of the cap, reserved for change only */
    uint8_t  conn_max_per_day;     /* radio sessions, not frames           */
    uint8_t  conn_priority_extra;  /* extra sessions a change event may use */

    /* --- batching: decouples frames from radio sessions --- */
    uint8_t  batch_max_frames;
    uint16_t batch_max_bytes;
    uint32_t batch_max_age_s;
    uint8_t  flush_on_change;      /* change events bypass batching        */
    /* Connection scheduling. Session setup costs about as much as 500 KB of
     * payload, so the count that decides battery life is connections, not
     * bytes. Default behaviour is therefore: queue everything, connect rarely
     * on a fixed schedule, and break that schedule only for a step change. */
    uint32_t conn_interval_s;      /* scheduled radio cadence              */
    uint8_t  conn_burst_max;       /* extra sessions a step change may use */
    uint32_t conn_burst_window_s;  /* over this window                     */
    uint8_t  drift_breaks_schedule;/* 0 = slow trends wait for the schedule*/

    /* --- sensitivity --- */
    float    s_hi, s_lo;           /* enter / exit change score            */
    uint8_t  persist_m, persist_n; /* N-of-M persistence                   */
    uint32_t refractory_s;
    uint8_t  change_burst_max;     /* uploads per burst before backing off */
    uint8_t  backoff_mult;         /* refractory multiplier after a burst  */
    float    cusum_k, cusum_h;     /* drift detector slack / alarm level   */
    float    step_min;             /* score jump that counts as a step     */
    uint32_t slow_freeze_n;        /* samples before the slow ref freezes  */
    float    min_vel_delta;        /* absolute significance floors         */
    float    min_acc_delta_db;

    /* --- operating-state discovery --- */
    uint8_t  idle_gating_enable;
    uint32_t discovery_min_n;      /* samples before the split is trusted  */
    float    discovery_min_sep;    /* Otsu separability required           */
    float    discovery_min_gap_db; /* physical gap between the two modes   */
    float    state_hyst_db;
    uint8_t  state_persist_n;

    /* --- safety --- */
    uint32_t fallback_after_s;     /* no cloud contact -> revert defaults  */
    uint8_t  debug_frames;         /* emit features every acq for N acqs   */

    /* --- one-shot commands, cleared once executed --- */
    uint8_t  cmd_reset_baseline;
    uint8_t  cmd_reset_discovery;
    uint8_t  cmd_force_upload;
    uint8_t  cmd_force_anchor;

    float    bearing[EG_NORDERS];  /* pushable: bearing changed on site    */
} ep_config_t;

/* =====================================================================
 * 3. UPLOAD QUEUE  (frames buffered, radio opened rarely)
 * A radio session costs roughly an order of magnitude more energy than the
 * bytes it carries, so the count that matters for battery is connections,
 * not uploads.
 * ===================================================================== */
#define EP_Q_SLOTS   12
#define EP_Q_SLOTSZ  1100

typedef struct {
    uint8_t  buf[EP_Q_SLOTS][EP_Q_SLOTSZ];
    uint16_t len[EP_Q_SLOTS];
    uint32_t ts[EP_Q_SLOTS];
    uint8_t  prio[EP_Q_SLOTS];
    uint8_t  head, count;
    uint32_t last_conn_s;
    uint32_t burst_t0;
    uint8_t  burst_used;
    uint32_t bytes;
    uint32_t conn_day;            /* day index of conn_used               */
    uint8_t  conn_used;
    uint32_t n_conn_total;
    uint32_t n_dropped;           /* queue overflow                        */
} ep_queue_t;

typedef struct {
    eg_ctx_t       eg;
    ep_discovery_t disc;
    ep_config_t    cfg;
    ep_queue_t     q;
    uint32_t       applied_version;
    uint32_t       last_contact_s;
    uint32_t       n_acq, n_idle_skipped;
    uint8_t        debug_left;
} ep_ctx_t;

/* result of one wake cycle */
typedef struct {
    ep_state_t    state;
    eg_decision_t decision;
    uint32_t      payload_len;
    uint8_t       queued;
    uint8_t       connected;      /* radio session opened this cycle       */
    uint16_t      conn_bytes;
    uint8_t       conn_frames;
    uint8_t       change_kind;    /* 1 step, 2 drift                       */
} ep_result_t;

void ep_defaults(ep_config_t *c);
void ep_init(ep_ctx_t *c, const ep_config_t *cfg);
void ep_apply_config(ep_ctx_t *c, const ep_config_t *cfg, uint32_t t_now_s);

/* one wake cycle: acquire -> classify -> gate -> encode -> queue -> maybe send */
ep_result_t ep_wake(ep_ctx_t *c, const float *x, uint32_t n, uint32_t t_now_s,
                    eg_features_t *feat, eg_change_t *chg);

const char *ep_state_name(ep_state_t s);
const uint8_t *eg_last_payload(void);
uint32_t eg_last_payload_len(void);

#endif
