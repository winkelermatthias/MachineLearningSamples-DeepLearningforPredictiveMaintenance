/* edge_gate.h - condition-monitoring edge gate + spectral GOP codec
 *
 * Decides whether an acquisition is worth transmitting, and if so encodes it.
 *
 * Upload happens when EITHER:
 *   (1) a meaningful, persistent change from the learned baseline is detected, or
 *   (2) the heartbeat interval has elapsed since the last upload.
 * Both are subject to a hard sliding-window rate cap.
 *
 * Target: Cortex-M4F class. Single precision float, no dynamic allocation,
 * no libc beyond math. All buffers are caller-owned or static.
 */
#ifndef EDGE_GATE_H
#define EDGE_GATE_H

#include <stdint.h>
#include <stdbool.h>

/* ------------------------------------------------------------------ config */
#define EG_FS_HZ          12000.0f   /* acquisition sample rate            */
#define EG_NFFT           4096       /* Welch segment, power of two        */
#define EG_MAXSAMP        16384      /* max samples per acquisition        */
#define EG_NSPEC          (EG_NFFT/2)/* usable spectral lines              */

#define EG_NBANDS         16         /* band-gain / mask / attribution     */
#define EG_NORDERS        8          /* tracked diagnostic orders          */
#define EG_NBINS_FINE     512        /* order axis: 0..20 orders           */
#define EG_NBINS_COARSE   256        /* order axis: 20..200 orders         */
#define EG_NBINS          (EG_NBINS_FINE + EG_NBINS_COARSE)
#define EG_ORD_FINE       20.0f
#define EG_ORD_MAX        200.0f

#define EG_ENV_LO_HZ      2000.0f    /* envelope demod band                */
#define EG_ENV_HI_HZ      5000.0f
#define EG_ENV_DECIM      4

#define EG_Q_DB           0.5f       /* uint8 quantiser step               */
#define EG_DB_OFF         (-128.0f)

#define EG_NFEAT          7          /* scalar features under baseline     */

/* ---- gate policy (tuned in s9 experiment; see RESULTS) ------------------ */
#define EG_HEARTBEAT_S    (3*24*3600)   /* condition 2: 3 days             */
#define EG_RATE_WINDOW_S  (7*24*3600)   /* sliding window for the cap      */
#define EG_RATE_MAX       14            /* hard cap: 14 uploads / 7 days   */
#define EG_WARMUP_N       20            /* acquisitions before change can fire */
#define EG_PERSIST_M      3             /* look back this many acquisitions*/
#define EG_PERSIST_N      2             /* need this many over threshold   */
#define EG_S_HI           4.0f          /* enter-change score              */
#define EG_S_LO           2.5f          /* exit-change score (hysteresis)  */
#define EG_REFRACTORY_S   (12*3600)     /* dead time after a change upload */
#define EG_ESCALATE_MULT  1.6f          /* ...unless score grows this much */
#define EG_MASK_BANDS_MIN 2             /* bands over mask to count        */
#define EG_MASK_MARGIN_DB 6.0f          /* mask = baseline + k*mad + this  */

/* absolute floors: statistics alone must not trigger on a physically
 * meaningless change. A very quiet machine has a tiny MAD. */
#define EG_MIN_VEL_DELTA  0.8f       /* mm/s RMS                           */
#define EG_MIN_ACC_DELTA_DB 2.5f     /* dB on acc rms                      */

/* ------------------------------------------------------------------ types */
typedef enum {
    EG_NOUP_WARMUP = 0,   /* baseline still learning, no heartbeat due     */
    EG_NOUP_QUIET,        /* nothing changed, heartbeat not due            */
    EG_NOUP_RATELIMIT,    /* wanted to send, rate cap refused              */
    EG_NOUP_REFRACTORY,   /* change seen but inside dead time              */
    EG_UP_HEARTBEAT,      /* condition 2                                   */
    EG_UP_CHANGE,         /* condition 1                                   */
    EG_UP_BASELINE_INIT   /* first frame, seeds the baseline + anchor      */
} eg_decision_t;

typedef struct {
    float acc_rms;        /* g, broadband                                  */
    float vel_rms;        /* mm/s, 10..1000 Hz, ISO 20816                  */
    float acc_kurt;       /* time-domain kurtosis                          */
    float acc_crest;      /* time-domain crest factor                      */
    float env_rms;        /* envelope RMS, g                               */
    float env_kurt;       /* envelope kurtosis                             */
    float env_crest;      /* envelope crest factor                         */
    float band_db[EG_NBANDS];    /* order-band levels, dB                  */
    float order_db[EG_NORDERS];  /* tracked orders, dB                     */
    float fr_hz;          /* estimated shaft speed                         */
    float fr_conf;        /* speed confidence                              */
    uint8_t align_tier;   /* 0=A 1=B 2=C 3=D                               */
} eg_features_t;

/* Streaming robust baseline: sign-update median + EWMA absolute deviation.
 * O(1) state per feature, no history buffer. Frozen while in change state so
 * the baseline never chases the fault it is supposed to detect. */
typedef struct {
    float med[EG_NFEAT];
    float mad[EG_NFEAT];
    float band_med[EG_NBANDS];
    float band_mad[EG_NBANDS];
    float order_med[EG_NORDERS];
    float order_mad[EG_NORDERS];
    /* Slow reference + CUSUM. The fast baseline above adapts, which is what
     * makes it robust to load and temperature, but it also means a slow
     * degradation is absorbed and never seen. The slow reference freezes once
     * it has enough support, and CUSUM accumulates small persistent shifts
     * against it. Steps are caught by the fast path, drift by this one. */
    float slow_med[EG_NFEAT];
    float slow_mad[EG_NFEAT];
    float cusum_p[EG_NFEAT];
    float cusum_n[EG_NFEAT];
    uint32_t n_slow;
    bool    slow_frozen;
    uint32_t n;
    bool    frozen;
} eg_baseline_t;

typedef struct {
    float    z[EG_NFEAT];         /* robust z per feature                   */
    float    score;               /* aggregate change score                 */
    float    band_delta_db[EG_NBANDS];   /* CHANGE ATTRIBUTION, signed      */
    float    order_delta_db[EG_NORDERS];
    uint8_t  mask_bands;          /* bands breaching the spectral mask      */
    float    mask_max_db;         /* worst breach                           */
    bool     abs_floor_ok;        /* physical-significance gate passed      */
    uint8_t  kind;                /* 0 none, 1 STEP (sudden), 2 DRIFT (slow) */
    float    jump;                /* score rise vs the previous acquisition  */
    float    drift;               /* max CUSUM statistic across features    */
    uint8_t  drift_feat;          /* which feature is drifting              */
} eg_change_t;

typedef struct {
    uint32_t last_upload_s;
    uint32_t last_change_up_s;
    float    last_change_score;
    uint32_t hist[EG_RATE_MAX];   /* ring of upload timestamps              */
    uint8_t  hist_head;
    uint8_t  hist_count;
    uint8_t  persist_ring;        /* bitfield of recent over-threshold      */
    uint8_t  burst_count;         /* change uploads since last escalation    */
    float    prev_score;
    uint8_t  step_latch;          /* a step was seen since in_change armed   */
    bool     in_change;           /* hysteresis latch                       */
    uint32_t n_acq;
} eg_gate_t;

/* Codec reference state (the anchor held on device) */
typedef struct {
    uint8_t  u8[EG_NBINS];
    uint16_t peak_idx[64];
    uint8_t  peak_amp[64];
    uint8_t  n_peaks;
    uint8_t  mad[EG_NBINS];
    bool     valid;
} eg_anchor_t;

/* Runtime-tunable policy. Mirrors the cloud shadow; the #defines above are
 * only the shipped defaults used to seed this. */
typedef struct {
    float    s_hi, s_lo, escalate;
    uint32_t heartbeat_s, refractory_s, rate_window_s;
    uint8_t  rate_max, reserve_change, persist_m, persist_n, warmup_n;
    uint8_t  change_burst_max, backoff_mult;
    float    cusum_k, cusum_h, step_min;
    uint32_t slow_freeze_n;
    float    min_vel_delta, min_acc_delta_db;
} eg_tune_t;

typedef struct {
    eg_baseline_t base;
    eg_gate_t     gate;
    eg_anchor_t   anchor;
    eg_tune_t     tune;
    float         bearing[EG_NORDERS];  /* order multipliers               */
} eg_ctx_t;

/* ------------------------------------------------------------------- api */
void  eg_init(eg_ctx_t *c, const float *bearing_orders);

/* Full per-acquisition pipeline. `x` is raw acceleration in g.
 * Returns the decision; fills `feat`, `chg`, and on upload writes the encoded
 * payload into `out` (capacity `out_cap`) and sets *out_len. */
eg_decision_t eg_process(eg_ctx_t *c, const float *x, uint32_t n,
                         uint32_t t_now_s,
                         eg_features_t *feat, eg_change_t *chg,
                         uint8_t *out, uint32_t out_cap, uint32_t *out_len);

/* exposed for test harnesses */
void  eg_features(const float *x, uint32_t n, eg_features_t *f,
                  const float *bearing, float *ord_amp_out);
float eg_estimate_speed(const float *x, uint32_t n, float *conf);
void  eg_score(eg_ctx_t *c, const eg_features_t *f, eg_change_t *g);
const char *eg_decision_name(eg_decision_t d);
uint32_t eg_enc_anchor(eg_ctx_t *c, const uint8_t *u8, uint8_t *o, uint32_t cap);
uint32_t eg_enc_residual(eg_ctx_t *c, const uint8_t *u8, uint8_t tier, uint8_t *o, uint32_t cap);
const uint8_t *eg_last_u8(void);
const float   *eg_last_ord(void);
void  eg_quantize(const float *ord, uint8_t *u8);
void  eg_baseline_update_public(eg_baseline_t *b, const eg_features_t *f);

#endif /* EDGE_GATE_H */
