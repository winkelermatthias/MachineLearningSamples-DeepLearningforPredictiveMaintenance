/* edge_gate.c - see edge_gate.h */
#include "edge_gate.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================ static work buffers ========================= */
/* Sized once, reported in the RAM budget. No malloc anywhere. */
static float  wk_re[EG_NFFT], wk_im[EG_NFFT];
static float  wk_spec[EG_NSPEC];      /* averaged acceleration magnitude    */
static float  wk_espec[EG_NSPEC];     /* averaged envelope magnitude        */
static float  wk_env[EG_MAXSAMP/EG_ENV_DECIM + 8];
static float  wk_ord[EG_NBINS];
static uint8_t wk_u8[EG_NBINS];
static float  wk_win[EG_NFFT];
static bool   wk_win_ready = false;

/* ================================ radix-2 FFT ============================ */
static void fft_r2(float *re, float *im, int n)
{
    int i, j = 0, k, m, step;
    for (i = 1; i < n; i++) {                 /* bit reversal */
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { float t;
            t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (step = 2; step <= n; step <<= 1) {
        float ang = -2.0f * (float)M_PI / (float)step;
        float wr = cosf(ang), wi = sinf(ang);
        for (m = 0; m < n; m += step) {
            float cr = 1.0f, ci = 0.0f;
            for (k = 0; k < step / 2; k++) {
                int a = m + k, b = m + k + step / 2;
                float xr = re[b] * cr - im[b] * ci;
                float xi = re[b] * ci + im[b] * cr;
                re[b] = re[a] - xr; im[b] = im[a] - xi;
                re[a] += xr;        im[a] += xi;
                float nr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr; cr = nr;
            }
        }
    }
}

static void make_window(void)
{
    if (wk_win_ready) return;
    for (int i = 0; i < EG_NFFT; i++)
        wk_win[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (EG_NFFT - 1));
    wk_win_ready = true;
}

/* Welch magnitude spectrum, 50% overlap, Hann, amplitude-scaled. */
static int welch(const float *x, uint32_t n, float *out, int nfft)
{
    make_window();
    float wsum = 0.0f;
    for (int i = 0; i < nfft; i++) wsum += wk_win[i];
    int half = nfft / 2, nseg = 0;
    memset(out, 0, sizeof(float) * (nfft / 2));
    for (uint32_t s = 0; s + nfft <= n; s += half) {
        float mean = 0.0f;
        for (int i = 0; i < nfft; i++) mean += x[s + i];
        mean /= nfft;
        for (int i = 0; i < nfft; i++) {
            wk_re[i] = (x[s + i] - mean) * wk_win[i];
            wk_im[i] = 0.0f;
        }
        fft_r2(wk_re, wk_im, nfft);
        for (int i = 0; i < nfft / 2; i++) {
            float m = sqrtf(wk_re[i] * wk_re[i] + wk_im[i] * wk_im[i]);
            out[i] += m * m;
        }
        nseg++;
    }
    if (!nseg) return 0;
    float sc = 2.0f / (wsum * wsum);
    for (int i = 0; i < nfft / 2; i++) out[i] = sqrtf(out[i] / nseg * sc) * 2.0f;
    return nseg;
}

/* ============================== biquad ================================== */
typedef struct { float b0,b1,b2,a1,a2,z1,z2; } biq_t;
static float biq(biq_t *s, float x)
{
    float y = s->b0 * x + s->z1;
    s->z1 = s->b1 * x - s->a1 * y + s->z2;
    s->z2 = s->b2 * x - s->a2 * y;
    return y;
}
static void biq_bp(biq_t *s, float f0, float q, float fs)
{
    float w = 2.0f*(float)M_PI*f0/fs, a = sinf(w)/(2.0f*q), c = cosf(w);
    float a0 = 1.0f + a;
    s->b0 = a/a0; s->b1 = 0.0f; s->b2 = -a/a0;
    s->a1 = -2.0f*c/a0; s->a2 = (1.0f-a)/a0; s->z1 = s->z2 = 0.0f;
}
static void biq_lp(biq_t *s, float f0, float q, float fs)
{
    float w = 2.0f*(float)M_PI*f0/fs, a = sinf(w)/(2.0f*q), c = cosf(w);
    float a0 = 1.0f + a;
    s->b0 = (1.0f-c)/2.0f/a0; s->b1 = (1.0f-c)/a0; s->b2 = s->b0;
    s->a1 = -2.0f*c/a0; s->a2 = (1.0f-a)/a0; s->z1 = s->z2 = 0.0f;
}

/* Envelope by band-pass, rectify, low-pass, decimate. Cheaper than Hilbert
 * and it is what real MCU firmware does. */
static uint32_t envelope(const float *x, uint32_t n, float *env)
{
    biq_t bp1, bp2, lp1, lp2;
    float fc = sqrtf(EG_ENV_LO_HZ * EG_ENV_HI_HZ);
    float q  = fc / (EG_ENV_HI_HZ - EG_ENV_LO_HZ);
    biq_bp(&bp1, fc, q, EG_FS_HZ); biq_bp(&bp2, fc, q, EG_FS_HZ);
    biq_lp(&lp1, 1000.0f, 0.707f, EG_FS_HZ); biq_lp(&lp2, 1000.0f, 0.707f, EG_FS_HZ);
    uint32_t m = 0;
    for (uint32_t i = 0; i < n; i++) {
        float v = biq(&bp2, biq(&bp1, x[i]));
        v = fabsf(v);
        v = biq(&lp2, biq(&lp1, v));
        if ((i % EG_ENV_DECIM) == 0 && m < EG_MAXSAMP/EG_ENV_DECIM) env[m++] = v;
    }
    float mean = 0.0f;
    for (uint32_t i = 0; i < m; i++) mean += env[i];
    mean /= (m ? m : 1);
    for (uint32_t i = 0; i < m; i++) env[i] -= mean;
    return m;
}

/* ============================ moment stats ============================== */
static void moments(const float *x, uint32_t n, float *rms, float *kurt, float *crest)
{
    double s1=0, s2=0;
    for (uint32_t i=0;i<n;i++) s1 += x[i];
    double mu = s1/n;
    for (uint32_t i=0;i<n;i++){ double d=x[i]-mu; s2 += d*d; }
    double var = s2/n, sd = sqrt(var>0?var:1e-30);
    double s4=0; float pk=0;
    for (uint32_t i=0;i<n;i++){ double d=(x[i]-mu)/sd; s4 += d*d*d*d;
        float a=fabsf(x[i]-(float)mu); if(a>pk) pk=a; }
    *rms   = (float)sqrt(var);
    *kurt  = (float)(s4/n);
    *crest = (*rms > 1e-12f) ? pk / *rms : 0.0f;
}

/* ==================== tacholess speed, envelope domain ==================
 * The acceleration spectrum's low-frequency region is dominated by mount
 * resonances; HPS there locks onto a fixed structural line. The envelope
 * exposes shaft 1x cleanly. Parabolic sub-bin refinement is mandatory:
 * without it the error floor is the bin width. */
static float hps_env(const float *espec, int ns, float df,
                     float lo, float hi, float *conf)
{
    const int NH = 3;
    float best = -1.0f, bestf = lo, rival = 0.0f;
    float step = df * 0.05f;
    for (float f = lo; f < hi; f += step) {
        float p = 1.0f;
        for (int h = 1; h <= NH; h++) {
            float b = f * h / df;
            int i = (int)b; float fr = b - i;
            if (i + 1 >= ns) { p = 0.0f; break; }
            float v = espec[i] * (1.0f - fr) + espec[i + 1] * fr;
            p *= (v > 1e-12f ? v : 1e-12f);
        }
        p = powf(p, 1.0f / NH);
        if (p > best) { best = p; bestf = f; }
    }
    for (float f = lo; f < hi; f += step) {
        if (fabsf(f - bestf) <= 0.5f) continue;
        float p = 1.0f;
        for (int h = 1; h <= NH; h++) {
            float b = f * h / df; int i = (int)b; float fr = b - i;
            if (i + 1 >= ns) { p = 0.0f; break; }
            float v = espec[i]*(1.0f-fr) + espec[i+1]*fr;
            p *= (v > 1e-12f ? v : 1e-12f);
        }
        p = powf(p, 1.0f/NH);
        if (p > rival) rival = p;
    }
    *conf = (rival > 1e-20f) ? best / rival : 1.0f;
    /* parabolic refine on the fundamental */
    int j = (int)(bestf / df + 0.5f);
    if (j > 0 && j + 1 < ns) {
        float y0 = espec[j-1], y1 = espec[j], y2 = espec[j+1];
        float den = 2.0f*(y0 - 2.0f*y1 + y2);
        if (fabsf(den) > 1e-20f) {
            float d = (y0 - y2) / den;
            if (d > -1.0f && d < 1.0f) bestf = (j + d) * df;
        }
    }
    return bestf;
}

float eg_estimate_speed(const float *x, uint32_t n, float *conf)
{
    uint32_t m = envelope(x, n, wk_env);
    float efs = EG_FS_HZ / EG_ENV_DECIM;
    int nf = EG_NFFT / 4;
    if (m < (uint32_t)nf) nf = 512;
    make_window();
    /* single long FFT: resolution beats variance reduction for speed */
    int N = 1; while (N < (int)m && N < EG_NFFT) N <<= 1;
    for (int i = 0; i < N; i++) {
        float w = 0.5f - 0.5f*cosf(2.0f*(float)M_PI*i/(N-1));
        wk_re[i] = (i < (int)m ? wk_env[i] : 0.0f) * w; wk_im[i] = 0.0f;
    }
    fft_r2(wk_re, wk_im, N);
    int ns = N/2;
    for (int i = 0; i < ns; i++)
        wk_espec[i] = 2.0f*sqrtf(wk_re[i]*wk_re[i]+wk_im[i]*wk_im[i])/m;
    return hps_env(wk_espec, ns, efs/N, 20.0f, 40.0f, conf);
}

/* ======================== order axis + binning ========================== */
static float ord_edge(int i)
{
    if (i <= EG_NBINS_FINE) return EG_ORD_FINE * (float)i / EG_NBINS_FINE;
    return EG_ORD_FINE + (EG_ORD_MAX - EG_ORD_FINE) *
           (float)(i - EG_NBINS_FINE) / EG_NBINS_COARSE;
}

/* Peak-preserving: MAX within each bin, never the mean. Averaging destroys
 * sideband spacing, which is the diagnosis. */
static void bin_orders(const float *spec, int ns, float df, float fr, float *out)
{
    for (int i = 0; i < EG_NBINS; i++) out[i] = 0.0f;
    for (int i = 1; i < ns; i++) {
        float o = (i * df) / fr;
        if (o < 0.0f || o >= EG_ORD_MAX) continue;
        int b;
        if (o < EG_ORD_FINE) b = (int)(o / EG_ORD_FINE * EG_NBINS_FINE);
        else b = EG_NBINS_FINE + (int)((o - EG_ORD_FINE) /
                 (EG_ORD_MAX - EG_ORD_FINE) * EG_NBINS_COARSE);
        if (b < 0 || b >= EG_NBINS) continue;
        if (spec[i] > out[b]) out[b] = spec[i];
    }
    for (int i = 1; i < EG_NBINS; i++) if (out[i] == 0.0f) out[i] = out[i-1];
}

static inline float amp_db(float a) { return 20.0f * log10f(a > 1e-12f ? a : 1e-12f); }
static inline uint8_t q_u8(float a)
{
    float v = (amp_db(a) - EG_DB_OFF) / EG_Q_DB;
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    return (uint8_t)(v + 0.5f);
}

/* ============================== features ================================ */
void eg_features(const float *x, uint32_t n, eg_features_t *f,
                 const float *bearing, float *ord_amp_out)
{
    memset(f, 0, sizeof(*f));
    moments(x, n, &f->acc_rms, &f->acc_kurt, &f->acc_crest);

    int nseg = welch(x, n, wk_spec, EG_NFFT);
    float df = EG_FS_HZ / EG_NFFT;
    (void)nseg;

    /* velocity RMS 10..1000 Hz via spectral integration, Parseval.
     * a[g] -> m/s^2 (x9.80665) -> v = a/(2 pi f) -> mm/s (x1000) */
    double acc2 = 0.0;
    for (int i = 1; i < EG_NSPEC; i++) {
        float fr_hz = i * df;
        if (fr_hz < 10.0f || fr_hz > 1000.0f) continue;
        float a_ms2 = wk_spec[i] * 9.80665f;
        float v = a_ms2 / (2.0f * (float)M_PI * fr_hz) * 1000.0f;
        acc2 += 0.5 * (double)v * v;             /* amplitude -> rms        */
    }
    f->vel_rms = (float)sqrt(acc2);

    uint32_t m = envelope(x, n, wk_env);
    moments(wk_env, m, &f->env_rms, &f->env_kurt, &f->env_crest);

    f->fr_hz = eg_estimate_speed(x, n, &f->fr_conf);
    if (f->fr_conf >= 3.0f)      f->align_tier = 1;   /* B: trust           */
    else if (f->fr_conf >= 1.8f) f->align_tier = 2;   /* C: refine          */
    else { f->align_tier = 3; }                       /* D: do not warp     */
    if (f->fr_hz < 5.0f || f->fr_hz > 200.0f) { f->fr_hz = 30.0f; f->align_tier = 3; }

    bin_orders(wk_spec, EG_NSPEC, df, f->fr_hz, wk_ord);
    if (ord_amp_out) memcpy(ord_amp_out, wk_ord, sizeof(wk_ord));

    /* band levels: RMS within each of NBANDS order bands */
    for (int b = 0; b < EG_NBANDS; b++) {
        int lo = b * EG_NBINS / EG_NBANDS, hi = (b + 1) * EG_NBINS / EG_NBANDS;
        double s = 0.0;
        for (int i = lo; i < hi; i++) s += (double)wk_ord[i] * wk_ord[i];
        f->band_db[b] = amp_db((float)sqrt(s / (hi - lo)));
    }
    /* tracked diagnostic orders: max within +/-0.15 order */
    for (int k = 0; k < EG_NORDERS; k++) {
        float o = bearing[k]; float best = 0.0f;
        for (int i = 0; i < EG_NBINS; i++) {
            float c = 0.5f * (ord_edge(i) + ord_edge(i + 1));
            if (fabsf(c - o) < 0.15f && wk_ord[i] > best) best = wk_ord[i];
        }
        f->order_db[k] = amp_db(best);
    }
}

/* ======================== robust streaming baseline =====================
 * Sign-update median plus EWMA absolute deviation. O(1) state, no history.
 * Update is skipped while the gate is latched in change, so the baseline
 * never chases the fault it exists to detect. */
#define EG_MED_STEP 0.02f
#define EG_MAD_A    0.02f

static void base_update_one(float *med, float *mad, float x, uint32_t n)
{
    if (n == 0) { *med = x; *mad = fabsf(x) * 0.05f + 1e-6f; return; }
    float scale = (*mad > 1e-9f) ? *mad : (fabsf(x) * 0.05f + 1e-6f);
    *med += EG_MED_STEP * scale * ((x > *med) ? 1.0f : -1.0f);
    *mad = (1.0f - EG_MAD_A) * (*mad) + EG_MAD_A * fabsf(x - *med);
    if (*mad < 1e-9f) *mad = 1e-9f;
}

/* Positive, log-normal-ish features are tracked in log domain. */
static void feat_vector(const eg_features_t *f, float *v)
{
    v[0] = logf(f->acc_rms  > 1e-9f ? f->acc_rms  : 1e-9f);
    v[1] = logf(f->vel_rms  > 1e-9f ? f->vel_rms  : 1e-9f);
    v[2] = f->acc_kurt;
    v[3] = f->acc_crest;
    v[4] = logf(f->env_rms  > 1e-9f ? f->env_rms  : 1e-9f);
    v[5] = f->env_kurt;
    v[6] = f->env_crest;
}

static void base_update(eg_baseline_t *b, const eg_features_t *f)
{
    float v[EG_NFEAT]; feat_vector(f, v);
    for (int i = 0; i < EG_NFEAT; i++) base_update_one(&b->med[i], &b->mad[i], v[i], b->n);
    if (!b->slow_frozen) {
        for (int i = 0; i < EG_NFEAT; i++) {
            if (b->n_slow == 0) { b->slow_med[i] = v[i]; b->slow_mad[i] = b->mad[i]; }
            else {                                   /* 10x slower than fast */
                float sc = b->slow_mad[i] > 1e-9f ? b->slow_mad[i] : 1e-6f;
                b->slow_med[i] += 0.002f * sc * ((v[i] > b->slow_med[i]) ? 1.f : -1.f);
                b->slow_mad[i] = 0.998f*b->slow_mad[i] + 0.002f*fabsf(v[i]-b->slow_med[i]);
                if (b->slow_mad[i] < 1e-9f) b->slow_mad[i] = 1e-9f;
            }
        }
        b->n_slow++;
    }
    for (int i = 0; i < EG_NBANDS; i++)
        base_update_one(&b->band_med[i], &b->band_mad[i], f->band_db[i], b->n);
    for (int i = 0; i < EG_NORDERS; i++)
        base_update_one(&b->order_med[i], &b->order_mad[i], f->order_db[i], b->n);
    b->n++;
}

/* ============================== scoring ================================= */
void eg_score(eg_ctx_t *c, const eg_features_t *f, eg_change_t *g)
{
    eg_baseline_t *b = &c->base;
    memset(g, 0, sizeof(*g));
    float v[EG_NFEAT]; feat_vector(f, v);

    for (int i = 0; i < EG_NFEAT; i++) {
        float s = 1.4826f * b->mad[i];
        if (s < 1e-6f) s = 1e-6f;
        float z = (v[i] - b->med[i]) / s;
        g->z[i] = z;
    }

    /* Aggregate with a p-mean over one-sided, clipped z. p=3 emphasises the
     * largest deviation without collapsing to a pure max, which is jumpy. */
    double acc = 0.0; const float P = 3.0f;
    for (int i = 0; i < EG_NFEAT; i++) {
        float z = g->z[i]; if (z < 0) z = -z;    /* two-sided: drops matter  */
        if (z > 12.0f) z = 12.0f;
        acc += powf(z, P);
    }
    float pm = powf((float)(acc / EG_NFEAT), 1.0f / P);

    /* spectral mask: baseline + k*mad + margin, per band */
    for (int i = 0; i < EG_NBANDS; i++) {
        float s = 1.4826f * b->band_mad[i]; if (s < 0.5f) s = 0.5f;
        float mask = b->band_med[i] + 3.0f * s + EG_MASK_MARGIN_DB;
        g->band_delta_db[i] = f->band_db[i] - b->band_med[i];
        float over = f->band_db[i] - mask;
        if (over > 0.0f) { g->mask_bands++; if (over > g->mask_max_db) g->mask_max_db = over; }
    }
    for (int i = 0; i < EG_NORDERS; i++)
        g->order_delta_db[i] = f->order_db[i] - b->order_med[i];

    /* CUSUM drift against the slow reference. k is the slack (shifts smaller
     * than k*sigma are ignored), h the alarm level. Two-sided. */
    g->drift = 0.0f; g->drift_feat = 0;
    if (b->slow_frozen) {
        for (int i = 0; i < EG_NFEAT; i++) {
            float sc = 1.4826f * b->slow_mad[i]; if (sc < 1e-6f) sc = 1e-6f;
            float zs = (v[i] - b->slow_med[i]) / sc;
            b->cusum_p[i] = fmaxf(0.0f, b->cusum_p[i] + zs - c->tune.cusum_k);
            b->cusum_n[i] = fmaxf(0.0f, b->cusum_n[i] - zs - c->tune.cusum_k);
            float m = fmaxf(b->cusum_p[i], b->cusum_n[i]);
            if (m > 400.0f) { b->cusum_p[i] = fminf(b->cusum_p[i],400.f);
                              b->cusum_n[i] = fminf(b->cusum_n[i],400.f); m = 400.0f; }
            if (m > g->drift) { g->drift = m; g->drift_feat = (uint8_t)i; }
        }
    }
    float drift_term = (g->drift > c->tune.cusum_h)
                     ? 1.0f + (g->drift - c->tune.cusum_h) / c->tune.cusum_h : 0.0f;

    float mask_term = 0.0f;
    if (g->mask_bands >= EG_MASK_BANDS_MIN)
        mask_term = 1.0f + g->mask_max_db / 6.0f + 0.5f * (g->mask_bands - EG_MASK_BANDS_MIN);

    g->score = pm + mask_term + drift_term;

    /* Physical-significance floor. Statistics alone must not trigger: a very
     * quiet machine has a tiny MAD and will otherwise trip on nothing. */
    float vel_base = expf(b->med[1]);
    float acc_base_db = 20.0f * b->med[0] / 2.302585f;   /* ln -> dB        */
    float acc_now_db  = 20.0f * v[0] / 2.302585f;
    g->abs_floor_ok = (fabsf(f->vel_rms - vel_base) >= c->tune.min_vel_delta) ||
                      (fabsf(acc_now_db - acc_base_db) >= c->tune.min_acc_delta_db) ||
                      (g->mask_bands >= EG_MASK_BANDS_MIN);
    /* Drift must still clear the physical floor. Letting CUSUM bypass it was a
     * bug: CUSUM will eventually accumulate to any threshold on a stationary
     * signal with a slightly mis-centred reference, and without the floor that
     * turns into a steady trickle of meaningless alarms. */
}

/* ============================ rate limiting ============================= */
/* Count uploads inside the sliding window and compare against the effective
 * limit. Heartbeats get a smaller limit than change events so that routine
 * traffic can never consume the budget a real event will need. */
static uint8_t rate_used(const eg_gate_t *g, uint32_t now, uint32_t win)
{
    uint8_t k = 0;
    for (uint8_t i = 0; i < g->hist_count; i++)
        if ((now - g->hist[i]) < win) k++;
    return k;
}
static bool rate_ok(const eg_gate_t *g, const eg_tune_t *tn,
                    uint32_t now, bool is_change)
{
    uint8_t used = rate_used(g, now, tn->rate_window_s);
    uint8_t lim  = is_change ? tn->rate_max
                             : (uint8_t)(tn->rate_max > tn->reserve_change
                                         ? tn->rate_max - tn->reserve_change : 1);
    return used < lim;
}
static void rate_note(eg_gate_t *g, uint32_t now)
{
    g->hist[g->hist_head] = now;
    g->hist_head = (uint8_t)((g->hist_head + 1) % EG_RATE_MAX);
    if (g->hist_count < EG_RATE_MAX) g->hist_count++;
    g->last_upload_s = now;
}

/* ================================ codec ================================= */
static int enc_varint(uint8_t *o, uint32_t v)
{ int n=0; do { uint8_t b=v&0x7F; v>>=7; o[n++]=b|(v?0x80:0);} while(v); return n; }
static inline uint32_t zz(int32_t v){ return (uint32_t)((v<<1)^(v>>31)); }

uint32_t eg_enc_anchor(eg_ctx_t *c, const uint8_t *u8, uint8_t *o, uint32_t cap)
{
    if (cap < EG_NBINS + 200u) return 0;
    /* peak table: local maxima ranked by value + prominence */
    uint8_t np = 0;
    for (int i = 1; i < EG_NBINS - 1 && np < 40; i++) {
        if (u8[i] >= u8[i-1] && u8[i] >= u8[i+1]) {
            int pr = u8[i] - (u8[i-1] < u8[i+1] ? u8[i-1] : u8[i+1]);
            if (pr >= 4) { c->anchor.peak_idx[np] = (uint16_t)i;
                           c->anchor.peak_amp[np] = u8[i]; np++; }
        }
    }
    c->anchor.n_peaks = np;
    memcpy(c->anchor.u8, u8, EG_NBINS);
    c->anchor.valid = true;
    uint32_t k = 0;
    o[k++] = 0x01; o[k++] = np;
    memcpy(o + k, u8, EG_NBINS); k += EG_NBINS;
    for (int i = 0; i < np; i++) {
        o[k++] = (uint8_t)(c->anchor.peak_idx[i] & 0xFF);
        o[k++] = (uint8_t)(c->anchor.peak_idx[i] >> 8);
        o[k++] = c->anchor.peak_amp[i];
    }
    return k;
}

uint32_t eg_enc_residual(eg_ctx_t *c, const uint8_t *u8, uint8_t tier,
                             uint8_t *o, uint32_t cap)
{
    const uint8_t *a = c->anchor.u8;
    int32_t d[EG_NBINS];
    for (int i = 0; i < EG_NBINS; i++) d[i] = (int32_t)u8[i] - a[i];

    /* layer 0: global gain (median over the live part of the spectrum) */
    int32_t hist[256]; memset(hist, 0, sizeof(hist));
    int live = 0;
    for (int i = 0; i < EG_NBINS; i++) { int v = d[i] + 128;
        if (v < 0) v = 0; if (v > 255) v = 255; hist[v]++; live++; }
    int cum = 0, g0 = 0;
    for (int v = 0; v < 256; v++) { cum += hist[v]; if (cum >= live/2) { g0 = v-128; break; } }

    /* layer 1: band gains */
    int8_t gb[EG_NBANDS];
    for (int b = 0; b < EG_NBANDS; b++) {
        int lo = b*EG_NBINS/EG_NBANDS, hi = (b+1)*EG_NBINS/EG_NBANDS;
        int h2[256]; memset(h2,0,sizeof(h2)); int nn=0;
        for (int i=lo;i<hi;i++){ int v=d[i]-g0+128; if(v<0)v=0; if(v>255)v=255; h2[v]++; nn++; }
        int cc=0, m=0;
        for (int v=0;v<256;v++){ cc+=h2[v]; if(cc>=nn/2){ m=v-128; break; } }
        gb[b] = (int8_t)(m > 127 ? 127 : (m < -127 ? -127 : m));
    }
    int32_t pred[EG_NBINS];
    for (int i = 0; i < EG_NBINS; i++) pred[i] = g0 + gb[i*EG_NBANDS/EG_NBINS];

    uint32_t k = 7 + EG_NBANDS;      /* header + band gains, filled at end  */
    if (k > cap) return 0;

    /* layer 2: tracked peaks */
    uint8_t npk = 0; uint32_t kp = k;
    int w = (tier == 3) ? 3 : 2;
    for (int p = 0; p < c->anchor.n_peaks; p++) {
        int idx = c->anchor.peak_idx[p];
        int lo = idx - w < 0 ? 0 : idx - w, hi = idx + w >= EG_NBINS ? EG_NBINS-1 : idx + w;
        int j = lo; for (int i = lo; i <= hi; i++) if (u8[i] > u8[j]) j = i;
        int damp = (int)u8[j] - ((int)a[idx] + pred[idx]);
        int dfrq = (tier == 3) ? 0 : (j - idx);
        if (damp > 127) damp = 127; if (damp < -127) damp = -127;
        if (damp > -3 && damp < 3 && dfrq == 0) continue;
        if (kp + 3 > cap) break;
        o[kp++] = (uint8_t)p; o[kp++] = (uint8_t)(int8_t)damp;
        if (tier != 3) o[kp++] = (uint8_t)(int8_t)dfrq;
        pred[j] += damp; npk++;
    }

    /* layer 3: residual bitmap, thresholded against the stored per-bin MAD */
    uint32_t kr = kp; int prev = -1; uint16_t nact = 0;
    for (int i = 0; i < EG_NBINS; i++) {
        int32_t r = d[i] - pred[i];
        int thr = (int)c->anchor.mad[i] * 5 / 2;   /* k_MAD = 2.5           */
        if (thr < 6) thr = 6;                      /* 3 dB floor            */
        if (r > -thr && r < thr) continue;
        if (kr + 8 > cap) break;
        kr += enc_varint(o + kr, (uint32_t)(i - prev - 1));
        kr += enc_varint(o + kr, zz(r));
        prev = i; nact++;
    }

    o[0] = 0x02; o[1] = tier; o[2] = (uint8_t)(int8_t)g0;
    o[3] = npk;  o[4] = (uint8_t)(nact & 0xFF); o[5] = (uint8_t)(nact >> 8);
    o[6] = 0;
    memcpy(o + 7, gb, EG_NBANDS);
    return kr;
}

/* ============================== main entry ============================== */
void eg_init(eg_ctx_t *c, const float *bearing_orders)
{
    memset(c, 0, sizeof(*c));
    for (int i = 0; i < EG_NORDERS; i++) c->bearing[i] = bearing_orders[i];
    for (int i = 0; i < EG_NBINS; i++) c->anchor.mad[i] = 4;   /* 2 dB seed  */
    c->tune.s_hi = EG_S_HI; c->tune.s_lo = EG_S_LO;
    c->tune.escalate = EG_ESCALATE_MULT;
    c->tune.heartbeat_s = EG_HEARTBEAT_S;
    c->tune.refractory_s = EG_REFRACTORY_S;
    c->tune.rate_window_s = EG_RATE_WINDOW_S;
    c->tune.rate_max = EG_RATE_MAX; c->tune.reserve_change = 6;
    c->tune.persist_m = EG_PERSIST_M; c->tune.persist_n = EG_PERSIST_N;
    c->tune.warmup_n = EG_WARMUP_N;
    c->tune.min_vel_delta = EG_MIN_VEL_DELTA;
    c->tune.min_acc_delta_db = EG_MIN_ACC_DELTA_DB;
    c->tune.change_burst_max = 3; c->tune.backoff_mult = 6;
    c->tune.cusum_k = 1.0f; c->tune.cusum_h = 10.0f; c->tune.step_min = 3.0f; c->tune.slow_freeze_n = 120;
}

const char *eg_decision_name(eg_decision_t d)
{
    switch (d) {
    case EG_NOUP_WARMUP:     return "no_warmup";
    case EG_NOUP_QUIET:      return "no_quiet";
    case EG_NOUP_RATELIMIT:  return "no_ratelimit";
    case EG_NOUP_REFRACTORY: return "no_refractory";
    case EG_UP_HEARTBEAT:    return "up_heartbeat";
    case EG_UP_CHANGE:       return "up_change";
    case EG_UP_BASELINE_INIT:return "up_init";
    }
    return "?";
}

eg_decision_t eg_process(eg_ctx_t *c, const float *x, uint32_t n, uint32_t t,
                         eg_features_t *f, eg_change_t *g,
                         uint8_t *out, uint32_t cap, uint32_t *out_len)
{
    *out_len = 0;
    eg_features(x, n, f, c->bearing, wk_ord);
    for (int i = 0; i < EG_NBINS; i++) wk_u8[i] = q_u8(wk_ord[i]);
    eg_score(c, f, g);
    c->gate.n_acq++;

    /* ---- first acquisition seeds baseline and anchor ---- */
    if (!c->anchor.valid) {
        base_update(&c->base, f);
        *out_len = eg_enc_anchor(c, wk_u8, out, cap);
        rate_note(&c->gate, t);
        return EG_UP_BASELINE_INIT;
    }

    /* Step vs drift. A step is a sudden jump the fast baseline has not yet
     * absorbed; drift is what the CUSUM found. They deserve different
     * transmission urgency: a step is worth breaking the connection schedule
     * for, a slow trend is not and can ride along on the next scheduled call. */
    g->jump = g->score - c->gate.prev_score;
    if (g->score >= c->tune.s_hi && g->jump >= c->tune.step_min) g->kind = 1;
    else if (g->drift > c->tune.cusum_h)                         g->kind = 2;
    else g->kind = 0;
    c->gate.prev_score = g->score;
    /* Latch the step. Persistence delays the upload by one acquisition, by
     * which time the jump has already been absorbed into prev_score and the
     * event would be reclassified as drift. The urgency belongs to the episode,
     * not to the single acquisition that happens to carry the upload. */
    if (g->kind == 1) c->gate.step_latch = 1;

    /* ---- persistence ring: bit set when score over the entry threshold ---- */
    bool over = (g->score >= (c->gate.in_change ? c->tune.s_lo : c->tune.s_hi)) && g->abs_floor_ok;
    c->gate.persist_ring = (uint8_t)(((c->gate.persist_ring << 1) | (over ? 1 : 0))
                                     & ((1u << c->tune.persist_m) - 1u));
    int votes = 0;
    for (int i = 0; i < c->tune.persist_m; i++) if (c->gate.persist_ring & (1u << i)) votes++;

    if (!c->base.slow_frozen && c->base.n_slow >= c->tune.slow_freeze_n)
        c->base.slow_frozen = true;
    bool warm = (c->base.n >= c->tune.warmup_n);
    bool change_now = warm && (votes >= c->tune.persist_n);

    /* hysteresis latch */
    if (change_now) c->gate.in_change = true;
    else if (g->score < c->tune.s_lo) {
        c->gate.in_change = false;
        c->gate.burst_count = 0;           /* return to normal re-arms it    */
        c->gate.step_latch = 0;
        c->gate.last_change_score = 0.0f;
    }

    /* baseline freezes while latched, so it cannot learn the fault */
    c->base.frozen = c->gate.in_change;
    if (!c->base.frozen) base_update(&c->base, f);

    bool heartbeat_due = (t - c->gate.last_upload_s) >= c->tune.heartbeat_s;

    eg_decision_t dec;
    if (change_now) {
        bool escalated  = g->score > c->gate.last_change_score * c->tune.escalate;
        /* Change fatigue. A fault that is present and stable does not need to
         * be re-reported twice a day forever: after a burst, back off hard
         * unless the condition is actually getting worse. This turns "fault
         * present" into dense sampling at onset and sparse sampling after,
         * which is what an analyst wants and what the battery can afford. */
        uint32_t refr = c->tune.refractory_s;
        if (c->gate.burst_count >= c->tune.change_burst_max && !escalated)
            refr *= c->tune.backoff_mult;
        bool refractory = (t - c->gate.last_change_up_s) < refr;
        if (refractory && !escalated) {
            if (!heartbeat_due) return EG_NOUP_REFRACTORY;
            dec = EG_UP_HEARTBEAT;
        } else dec = EG_UP_CHANGE;
    } else if (heartbeat_due) {
        dec = EG_UP_HEARTBEAT;
    } else if (!warm) {
        return EG_NOUP_WARMUP;
    } else {
        return EG_NOUP_QUIET;
    }

    /* hard cap applies to everything. Heartbeat at 3 d is ~2.33/week so it can
     * never be the thing that saturates a 14/week budget. */
    if (!rate_ok(&c->gate, &c->tune, t, dec == EG_UP_CHANGE)) return EG_NOUP_RATELIMIT;

    /* change uploads send a fresh anchor (full fidelity when it matters);
     * heartbeats send a residual against the standing anchor. */
    if (dec == EG_UP_CHANGE) {
        if (c->gate.step_latch) g->kind = 1;   /* report the episode's kind  */
        *out_len = eg_enc_anchor(c, wk_u8, out, cap);
        if (g->score > c->gate.last_change_score * c->tune.escalate)
             c->gate.burst_count = 1;      /* escalation re-arms the burst   */
        else c->gate.burst_count++;
        c->gate.last_change_up_s = t;
        c->gate.last_change_score = g->score;
        for (int i = 0; i < EG_NFEAT; i++)      /* reported: reset the CUSUM */
            { c->base.cusum_p[i] = 0.0f; c->base.cusum_n[i] = 0.0f; }
    } else {
        *out_len = eg_enc_residual(c, wk_u8, f->align_tier, out, cap);
        if (*out_len == 0 || *out_len > (EG_NBINS * 6 / 10))
            *out_len = eg_enc_anchor(c, wk_u8, out, cap);   /* early anchor    */
    }
    rate_note(&c->gate, t);
    return dec;
}

/* ---- exposed for the policy layer ---- */
const uint8_t *eg_last_u8(void) { return wk_u8; }
const float   *eg_last_ord(void){ return wk_ord; }
void eg_quantize(const float *ord, uint8_t *u8)
{ for (int i=0;i<EG_NBINS;i++) u8[i]=q_u8(ord[i]); }
void eg_baseline_update_public(eg_baseline_t *b, const eg_features_t *f){ base_update(b,f); }
