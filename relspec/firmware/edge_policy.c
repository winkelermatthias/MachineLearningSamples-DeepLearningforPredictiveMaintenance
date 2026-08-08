/* edge_policy.c */
#include "edge_policy.h"
#include <math.h>
#include <string.h>

/* Push the cloud shadow into the gate's runtime tunables. */
static void ep_sync_tune(ep_ctx_t *c)
{
    eg_tune_t *t = &c->eg.tune; const ep_config_t *g = &c->cfg;
    t->s_hi = g->s_hi; t->s_lo = g->s_lo;
    t->heartbeat_s = g->heartbeat_s; t->refractory_s = g->refractory_s;
    t->rate_window_s = g->rate_window_s;
    t->rate_max = (uint8_t)(g->upload_max_per_window > EG_RATE_MAX
                            ? EG_RATE_MAX : g->upload_max_per_window);
    t->reserve_change = g->upload_reserve_change;
    t->persist_m = g->persist_m; t->persist_n = g->persist_n;
    if (g->change_burst_max) t->change_burst_max = g->change_burst_max;
    if (g->backoff_mult)     t->backoff_mult = g->backoff_mult;
    if (g->cusum_h > 0)      t->cusum_h = g->cusum_h;
    if (g->cusum_k > 0)      t->cusum_k = g->cusum_k;
    if (g->slow_freeze_n)    t->slow_freeze_n = g->slow_freeze_n;
    if (g->step_min > 0)     t->step_min = g->step_min;
    t->min_vel_delta = g->min_vel_delta; t->min_acc_delta_db = g->min_acc_delta_db;
}

/* ===================== 1. operating-state discovery ===================== */

static int hist_bin(float db)
{
    float t = (db - EP_HIST_LO_DB) / (EP_HIST_HI_DB - EP_HIST_LO_DB) * EP_HIST_BINS;
    int i = (int)t;
    if (i < 0) i = 0;
    if (i >= EP_HIST_BINS) i = EP_HIST_BINS - 1;
    return i;
}
static float bin_db(int i)
{
    return EP_HIST_LO_DB + (i + 0.5f) *
           (EP_HIST_HI_DB - EP_HIST_LO_DB) / EP_HIST_BINS;
}

static void hist_push(ep_discovery_t *d, float db)
{
    d->hist[hist_bin(db)]++;
    d->n_total++;
    if (d->n_total > EP_HIST_CAP) {           /* decay: adapt to duty change */
        uint32_t s = 0;
        for (int i = 0; i < EP_HIST_BINS; i++) { d->hist[i] >>= 1; s += d->hist[i]; }
        d->n_total = s;
    }
}

/* Otsu: pick the level that maximises between-class variance. Returns
 * separability eta = sigma_b^2 / sigma_total^2 in [0,1]. */
static void otsu(ep_discovery_t *d, const ep_config_t *cfg)
{
    double tot = 0, sum = 0;
    for (int i = 0; i < EP_HIST_BINS; i++) { tot += d->hist[i]; sum += (double)d->hist[i]*bin_db(i); }
    if (tot < 4) { d->valid = false; return; }
    double mu = sum / tot, var_t = 0;
    for (int i = 0; i < EP_HIST_BINS; i++) {
        double dv = bin_db(i) - mu; var_t += d->hist[i] * dv * dv;
    }
    var_t /= tot;

    /* Tie handling matters more than the search. When the two modes are well
     * separated, every split point between them yields identical between-class
     * variance. A strict '>' then parks the threshold on the upper edge of the
     * idle cluster, leaving no margin, and idle frames whose level wobbles by a
     * dB get called running. Take the midpoint of the tied plateau instead. */
    double w0 = 0, s0 = 0, best = -1; int bi_lo = -1, bi_hi = -1;
    for (int i = 0; i < EP_HIST_BINS - 1; i++) {
        w0 += d->hist[i]; s0 += (double)d->hist[i]*bin_db(i);
        double w1 = tot - w0;
        if (w0 < 1 || w1 < 1) continue;
        double m0 = s0 / w0, m1 = (sum - s0) / w1;
        double sb = (w0/tot) * (w1/tot) * (m0 - m1) * (m0 - m1);
        if (sb > best * 1.000001) { best = sb; bi_lo = bi_hi = i; }
        else if (best > 0 && sb >= best * 0.999999) { bi_hi = i; }
    }
    if (bi_lo < 0) { d->valid = false; return; }
    int bi = (bi_lo + bi_hi) / 2;

    w0 = 0; s0 = 0;
    for (int i = 0; i <= bi; i++) { w0 += d->hist[i]; s0 += (double)d->hist[i]*bin_db(i); }
    double w1 = tot - w0;
    d->thr_db    = bin_db(bi);
    d->mu_idle_db= (float)(w0 > 0 ? s0/w0 : 0);
    d->mu_run_db = (float)(w1 > 0 ? (sum-s0)/w1 : 0);
    d->w_idle    = (float)(w0/tot);
    d->w_run     = (float)(w1/tot);
    d->sep       = (float)(var_t > 1e-9 ? best/var_t : 0.0);

    /* Validity. All four must hold, and the gap test is the important one:
     * Otsu will always return *a* threshold, including on a unimodal
     * distribution. A continuously-running machine must stay UNKNOWN. */
    d->valid = (d->n_total >= cfg->discovery_min_n)
            && (d->sep    >= cfg->discovery_min_sep)
            && (d->w_idle >= 0.05f && d->w_run >= 0.05f)
            && ((d->mu_run_db - d->mu_idle_db) >= cfg->discovery_min_gap_db);
    d->last_fit_n = d->n_total;
}

/* Classify one acquisition. Level is primary; spectral structure is the veto.
 * A machine coasting down is loud but incoherent, and must not be treated as
 * a running sample: it would poison the baseline with a speed that no longer
 * exists. That is what TRANSIENT is for. */
static ep_state_t classify(ep_ctx_t *c, const eg_features_t *f)
{
    ep_discovery_t *d = &c->disc;
    float db = 20.0f * log10f(f->acc_rms > 1e-9f ? f->acc_rms : 1e-9f);

    hist_push(d, db);
    if (d->n_total - d->last_fit_n >= 16 || !d->valid) otsu(d, &c->cfg);

    if (!c->cfg.idle_gating_enable) return EP_STATE_RUNNING;
    if (!d->valid)                  return EP_STATE_UNKNOWN;

    /* The ambiguous band scales with the separation the device actually
     * discovered. A fixed width is wrong: on a machine with a 23 dB idle/run
     * gap, 12 dB swallows both clusters, and everything needs persistence. */
    float ambig = 0.25f * (d->mu_run_db - d->mu_idle_db);
    if (ambig < 3.0f) ambig = 3.0f;
    if (ambig > EP_AMBIG_DB) ambig = EP_AMBIG_DB;
    float hy = c->cfg.state_hyst_db;
    ep_state_t want;
    if (d->state == EP_STATE_RUNNING) want = (db < d->thr_db - hy) ? EP_STATE_IDLE : EP_STATE_RUNNING;
    else                             want = (db > d->thr_db + hy) ? EP_STATE_RUNNING : EP_STATE_IDLE;

    /* Structural veto, but only inside the ambiguous level band just above the
     * threshold. A machine clearly at full level is running whatever its
     * envelope coherence looks like; applying the veto there just discards
     * good data from quiet, well-balanced machines. Coast-down, which is what
     * TRANSIENT exists for, sits in the ambiguous band by construction. */
    if (want == EP_STATE_RUNNING && f->fr_conf < 1.2f &&
        db < d->thr_db + ambig) want = EP_STATE_TRANSIENT;

    /* Persistence exists to stop flapping at the boundary, so require it only
     * near the boundary. Far from the threshold there is nothing to flap
     * about, and demanding N consecutive votes there simply mislabels the
     * first acquisition of every run block. With short duty cycles that is a
     * large fraction of the data. */
    c->disc.pend_ring = (uint8_t)((c->disc.pend_ring << 1) |
                                  (want != d->state ? 1 : 0));
    int votes = 0;
    for (int i = 0; i < c->cfg.state_persist_n; i++)
        if (c->disc.pend_ring & (1u << i)) votes++;
    bool unambiguous = fabsf(db - d->thr_db) > ambig;
    if (want != d->state && (unambiguous || votes >= c->cfg.state_persist_n)) {
        d->state_prev = d->state;
        d->state = want;
        c->disc.pend_ring = 0;
    }
    return d->state;
}

/* ============================ 2. config ================================= */
void ep_defaults(ep_config_t *c)
{
    memset(c, 0, sizeof(*c));
    c->version = 1;
    c->wake_interval_s = 6*3600;          /* 4 acquisitions/day            */
    c->heartbeat_s     = 3*24*3600;
    c->upload_max_per_window = 14;
    c->rate_window_s   = 7*24*3600;
    c->upload_reserve_change = 6;         /* heartbeats cannot eat these   */
    c->conn_max_per_day = 2;
    c->conn_priority_extra = 1;
    c->change_burst_max = 3; c->backoff_mult = 6;
    c->cusum_k = 1.0f; c->cusum_h = 10.0f; c->step_min = 3.0f; c->slow_freeze_n = 120;
    c->batch_max_frames = 12;
    c->batch_max_bytes  = 11000;
    c->batch_max_age_s  = 7*24*3600;
    c->flush_on_change  = 0;          /* drift no longer breaks the schedule */
    c->conn_interval_s  = 3*24*3600;
    c->conn_burst_max   = 3;
    c->conn_burst_window_s = 2*24*3600;
    c->drift_breaks_schedule = 0;
    c->s_hi = 4.0f; c->s_lo = 2.5f;
    c->persist_m = 3; c->persist_n = 2;
    c->refractory_s = 12*3600;
    c->min_vel_delta = 0.8f;
    c->min_acc_delta_db = 2.5f;
    c->idle_gating_enable = 1;
    c->discovery_min_n = 48;
    c->discovery_min_sep = 0.55f;
    c->discovery_min_gap_db = 6.0f;
    c->state_hyst_db = 2.0f;
    c->state_persist_n = 2;
    c->fallback_after_s = 30*24*3600;
    static const float br[EG_NORDERS] =
        {1.0f, 2.0f, 3.0f, 3.5848f, 5.4152f, 4.7135f, 0.3983f, 10.8304f};
    memcpy(c->bearing, br, sizeof(br));
}

void ep_apply_config(ep_ctx_t *c, const ep_config_t *n, uint32_t t)
{
    if (n->version <= c->applied_version) return;
    ep_config_t old = c->cfg;
    c->cfg = *n;
    c->last_contact_s = t;

    if (n->cmd_reset_baseline) {
        memset(&c->eg.base, 0, sizeof(c->eg.base));
        c->eg.gate.in_change = false;
        c->eg.gate.persist_ring = 0;
    }
    if (n->cmd_reset_discovery) {
        memset(&c->disc, 0, sizeof(c->disc));
    }
    if (n->cmd_force_anchor) c->eg.anchor.valid = false;
    if (n->debug_frames)     c->debug_left = n->debug_frames;

    memcpy(c->eg.bearing, n->bearing, sizeof(c->eg.bearing));
    ep_sync_tune(c);
    c->applied_version = n->version;
    c->cfg.cmd_reset_baseline = c->cfg.cmd_reset_discovery = 0;
    c->cfg.cmd_force_anchor = 0;
    (void)old;
}

void ep_init(ep_ctx_t *c, const ep_config_t *cfg)
{
    memset(c, 0, sizeof(*c));
    c->cfg = *cfg;
    eg_init(&c->eg, cfg->bearing);
    ep_sync_tune(c);
    c->applied_version = cfg->version;
    c->disc.state = EP_STATE_UNKNOWN;
}

const char *ep_state_name(ep_state_t s)
{
    switch (s) { case EP_STATE_IDLE: return "idle";
                 case EP_STATE_RUNNING: return "running";
                 case EP_STATE_TRANSIENT: return "transient";
                 default: return "unknown"; }
}

/* ============================ 3. queue ================================== */
static void q_push(ep_queue_t *q, const uint8_t *p, uint32_t len, uint32_t t, uint8_t prio)
{
    if (len == 0 || len > EP_Q_SLOTSZ) { q->n_dropped++; return; }
    if (q->count >= EP_Q_SLOTS) {          /* drop oldest non-priority     */
        uint8_t i = q->head;
        q->bytes -= q->len[i];
        q->head = (uint8_t)((q->head+1) % EP_Q_SLOTS);
        q->count--; q->n_dropped++;
    }
    uint8_t s = (uint8_t)((q->head + q->count) % EP_Q_SLOTS);
    memcpy(q->buf[s], p, len);
    q->len[s] = (uint16_t)len; q->ts[s] = t; q->prio[s] = prio;
    q->count++; q->bytes += len;
}

/* prio_kind: 0 none, 1 step change, 2 drift */
static bool q_should_flush(ep_ctx_t *c, uint32_t t, uint8_t prio_kind)
{
    ep_queue_t *q = &c->q;
    if (q->count == 0) return false;
    uint32_t day = t / 86400u;
    if (q->conn_day != day) { q->conn_day = day; q->conn_used = 0; }

    /* Hard reasons: the queue cannot hold any more without dropping data. */
    if (q->count >= c->cfg.batch_max_frames) return true;
    if (q->bytes >= c->cfg.batch_max_bytes)  return true;
    if (q->count >= EP_Q_SLOTS - 1)          return true;

    /* A step change may break the schedule, within a bounded burst budget. */
    bool step = (prio_kind == 1) ||
                (prio_kind == 2 && c->cfg.drift_breaks_schedule);
    if (step) {
        if (t - q->burst_t0 > c->cfg.conn_burst_window_s) {
            q->burst_t0 = t; q->burst_used = 0;
        }
        if (q->burst_used < c->cfg.conn_burst_max &&
            q->conn_used < (uint8_t)(c->cfg.conn_max_per_day +
                                     c->cfg.conn_priority_extra)) {
            q->burst_used++;
            return true;
        }
    }

    /* Otherwise wait for the scheduled call. */
    if (q->last_conn_s == 0) return (t - q->ts[q->head]) >= c->cfg.conn_interval_s;
    if ((t - q->last_conn_s) >= c->cfg.conn_interval_s)
        return q->conn_used < c->cfg.conn_max_per_day;
    if ((t - q->ts[q->head]) >= c->cfg.batch_max_age_s) return true;
    return false;
}

static void q_flush(ep_ctx_t *c, uint32_t t, ep_result_t *r)
{
    ep_queue_t *q = &c->q;
    uint32_t day = t / 86400u;
    if (q->conn_day != day) { q->conn_day = day; q->conn_used = 0; }
    r->connected = 1; r->conn_bytes = (uint16_t)q->bytes; r->conn_frames = q->count;
    q->conn_used++; q->n_conn_total++; q->last_conn_s = t;
    q->count = 0; q->head = 0; q->bytes = 0;
    c->last_contact_s = t;
}

static uint8_t last_pl[EP_Q_SLOTSZ]; static uint32_t last_pl_len;
const uint8_t *eg_last_payload(void){ return last_pl; }
uint32_t eg_last_payload_len(void){ return last_pl_len; }

/* ============================ 4. wake cycle ============================= */
ep_result_t ep_wake(ep_ctx_t *c, const float *x, uint32_t n, uint32_t t,
                    eg_features_t *f, eg_change_t *g)
{
    ep_result_t r; memset(&r, 0, sizeof(r));
    c->n_acq++;

    /* fail-safe: long silence from the cloud reverts to shipped defaults */
    if (c->cfg.fallback_after_s && c->last_contact_s &&
        (t - c->last_contact_s) > c->cfg.fallback_after_s) {
        ep_config_t d; ep_defaults(&d); d.version = c->cfg.version + 1;
        memcpy(d.bearing, c->cfg.bearing, sizeof(d.bearing));
        c->cfg = d; c->last_contact_s = t;
    }

    eg_features(x, n, f, c->eg.bearing, NULL);
    r.state = classify(c, f);

    /* IDLE and TRANSIENT: no transmission, and critically no baseline update.
     * Letting idle samples into the baseline drags the learned median down,
     * which then makes every running sample look like a huge anomaly. */
    if (r.state == EP_STATE_IDLE || r.state == EP_STATE_TRANSIENT) {
        c->n_idle_skipped++;
        r.decision = EG_NOUP_QUIET;
        if (q_should_flush(c, t, 0)) q_flush(c, t, &r);
        return r;
    }

    uint8_t buf[EP_Q_SLOTSZ]; uint32_t len = 0;
    r.decision = eg_process(&c->eg, x, n, t, f, g, buf, sizeof(buf), &len);

    if (c->cfg.cmd_force_upload && len == 0) {
        const uint8_t *u8 = eg_last_u8();
        len = eg_enc_anchor(&c->eg, u8, buf, sizeof(buf));
        r.decision = EG_UP_CHANGE;
        c->cfg.cmd_force_upload = 0;
    }

    r.payload_len = len;
    if (len) { memcpy(last_pl, buf, len); last_pl_len = len; }
    /* Urgency belongs to the ONSET of an episode, not to every upload while a
     * known fault persists. The first change upload of an episode may break the
     * connection schedule; subsequent ones ride the next scheduled call. */
    uint8_t kind = (r.decision == EG_UP_CHANGE)
                 ? ((g->kind == 1 && c->eg.gate.burst_count <= 1) ? 1 : 2) : 0;
    r.change_kind = kind;
    if (len > 0) { q_push(&c->q, buf, len, t, kind); r.queued = 1; }
    if (q_should_flush(c, t, len > 0 ? kind : 0)) q_flush(c, t, &r);
    if (c->debug_left) c->debug_left--;
    return r;
}
