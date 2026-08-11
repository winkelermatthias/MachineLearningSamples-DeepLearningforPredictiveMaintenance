"""Extraction v2: v1's pipeline generalised across rigs.

Differences from pipeline.extract, each forced by real data:

  - The demodulation band comes from a kurtogram, not a constant. The v1
    band (2-5 kHz) does not exist below a 5.12 kHz sample rate and misses
    MFPT's sub-2 kHz outer-race ringing. Chosen once per machine state and
    cached: band identity is an asset property, not an acquisition property.
  - Speed search runs in a window around the nominal speed when one is known
    (every real rig publishes one); the unconstrained 20-70 Hz search remains
    the fallback. This is what makes 25 Hz MFPT and 20 Hz SEU tractable with
    the same code that handles 29.95 Hz CWRU.
  - Welch nperseg scales with fs so spectral resolution is constant in Hz
    across rigs, keeping order-bin occupancy comparable.

The output is the same Extract dataclass: everything downstream (codecs,
patterns, gate) is rig-agnostic by construction.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import welch
from .pipeline import (Extract, EDGES, ENV_EDGES, NBANDS, NBINS, bin_orders,
                       moments, to_db, to_u8, coherence_limit)
from .dsp2 import kurtogram_band, envelope_banded, hps_grid, acc_comb_surface
from .datasets import nperseg_for

_band_cache: dict = {}

# Bayesian speed posterior (estimate_speed3) in extract2. Module flag, not a
# per-call kwarg: encoder and analyst must agree on which estimator produced
# the stored fr, so the choice is a deployment property. Default on; the
# dual-evidence gate below remains both the fallback (guards in extract2)
# and the fr_nominal=None path.
BAYES_SPEED = True

def band_for(key, x, fs):
    """Kurtogram once per (machine state) key; every later acquisition of that
    state reuses the answer. Determinism matters: encoder and analyst must
    agree on what the envelope rail means."""
    if key not in _band_cache:
        _band_cache[key] = kurtogram_band(x, fs)[0]
    return _band_cache[key]

def estimate_speed2(x, fs, band, fr_nominal=None, span=0.35, nh=3):
    """Envelope-domain HPS with parabolic refinement, searched over
    [nominal*(1-span), nominal*(1+span)] when a nominal is known."""
    e = envelope_banded(x, fs, band)
    N = 1 << int(np.ceil(np.log2(len(e))))
    A = np.abs(np.fft.rfft(e*np.hanning(len(e)), N))*2/len(e)
    f = np.fft.rfftfreq(N, 1/fs)
    lo, hi = ((1-span)*fr_nominal, (1+span)*fr_nominal) if fr_nominal \
        else (20.0, 70.0)
    grid = np.arange(max(lo, 2.0), hi, 0.005)
    hps = np.ones_like(grid)
    for h in range(1, nh+1):
        hps *= np.maximum(np.interp(grid*h, f, A), 1e-12)
    hps = hps**(1/nh)
    j = int(np.argmax(hps)); fr = float(grid[j])
    rival = hps[np.abs(grid-grid[j]) > 0.5]
    conf = float(hps[j]/max(rival.max() if rival.size else 1e-12, 1e-12))
    k = int(round(fr/(f[1]-f[0])))
    if 0 < k < len(A)-1:
        y0, y1, y2 = A[k-1], A[k], A[k+1]
        den = 2*(y0-2*y1+y2)
        if abs(den) > 1e-20:
            d = (y0-y2)/den
            if -1 < d < 1: fr = float((k+d)*(f[1]-f[0]))
    return fr, conf

def acc_comb_speed(f, a, fr_nominal, span=0.35, nh=6):
    """Speed from the ACCELERATION harmonic comb. The envelope estimator is
    blind on a healthy machine - no impacts, no impact modulation, no shaft
    comb in the envelope - which is most of a fleet on most days. But every
    rotating machine drives 1x..Nx into the casing whether or not anything
    is wrong. Searched only near the nominal, with the product over six
    harmonics, so a single mount line cannot win: it would need five
    accomplices at exact multiples."""
    grid = np.arange(max(2.0, (1-span)*fr_nominal), (1+span)*fr_nominal, 0.005)
    if not len(grid): return fr_nominal, 0.0
    sc = np.ones_like(grid)
    for h in range(1, nh+1):
        sc *= np.maximum(np.interp(grid*h, f, a), 1e-12)
    sc = sc**(1/nh)
    j = int(np.argmax(sc)); fr = float(grid[j])
    if 0 < j < len(sc)-1:
        y0, y1, y2 = sc[j-1], sc[j], sc[j+1]
        den = 2*(y0-2*y1+y2)
        if abs(den) > 1e-20:
            d = (y0-y2)/den
            if -1 < d < 1: fr = float(grid[j]+d*0.005)
    # The rival exclusion zone must clear the WELCH MAIN LOBE, not a fixed
    # 0.5 Hz. At 2.93 Hz bins the peak's own shoulders extend +/-3 Hz on the
    # interpolated grid; measuring the "rival" inside the same lobe pins
    # confidence at ~1 and the estimate can never be believed.
    excl = max(2.5*(f[1]-f[0]), 0.04*fr_nominal)
    rival = sc[np.abs(grid-grid[j]) > excl]
    conf = float(sc[j]/max(rival.max() if rival.size else 1e-12, 1e-12))
    return fr, conf

# ------------------------------------------------ speed3: Bayesian posterior
# estimate_speed2's failure mode is structural: on an outer-race fault the
# envelope comb is BPFO-spaced, BPFO/3 (~1.19x for a 6205) lands inside the
# search span, and the envelope HPS is CONFIDENTLY wrong - a confidence gate
# cannot catch it. The dual-evidence if/else in extract2 patches this; the
# posterior replaces the patch with the thing it approximates: a joint
# posterior over shaft speed in which the envelope rail's harmonic index is
# an explicit latent variable. Promoted from experiments/bayes/speed_bayes.py
# (84-acq gate: median 0.145%, p95 0.349%, 0 gross incl. 12 adversarial
# BPFO/3 cases where estimate_speed2 errs 19-32%).

KS = (1/3., 1/2., 1.0, 2.0, 3.0)          # envelope harmonic hypotheses
# k=2,3 exist to catch envelope harmonic locks, but they are also where the
# bearing-comb alias hides (BPFO 3.58x read as "3x shaft" puts f at 1.19x
# true - the same trap in new clothes), so they start with a real handicap.
K_PRIOR = (0.045, 0.045, 0.87, 0.02, 0.02)
# +/-2 sigma empirical coverage calibration, frozen from the experiment's dev
# slice (seeds 5xxx, disjoint from its benchmark corpus): the extrapolation
# variance term over-counts (same frames set f and f_dot), dev max err/sigma
# 0.73 pre-cal, so 0.6 keeps a 1.6x margin while staying informative.
SIGMA_CAL = 0.6


class SpeedPosterior:
    """Rao-Blackwellized speed filter: 2-state Kalman (f, f_dot) per
    harmonic hypothesis + discrete posterior over hypotheses.

    be/ba are the rail evidence exponents (likelihood sharpness of the
    floor-normalised envelope / acc surfaces). ba > be on purpose: the acc
    comb is what DEFINES shaft speed (extract2's own reasoning), while the
    envelope comb is sharp but can sit at a bearing rate; a severe outer
    race fault puts a 35x-floor peak in the envelope surface and the k=3
    hypothesis will happily read BPFO as 3x shaft unless the acc rail
    carries enough weight to veto it. Tuned once on a dev slice with seeds
    disjoint from the benchmark corpus; see experiments/bayes/bench_speed.py."""

    def __init__(self, fr_nominal, span=0.35, be=1.2, ba=2.6, step=0.01,
                 nh_env=3, nh_acc=6):
        self.fn = float(fr_nominal)
        self.span, self.be, self.ba, self.step = span, be, ba, step
        self.nh_env, self.nh_acc = nh_env, nh_acc
        self.flo = max(2.0, (1-span)*self.fn)
        self.fhi = (1+span)*self.fn
        self.fgrid = np.arange(self.flo, self.fhi, step)
        s0 = 0.20*self.fn                      # ~covers the search span
        self.mu = [np.array([self.fn, 0.0]) for _ in KS]
        self.P = [np.diag([s0**2, (0.02*self.fn)**2]) for _ in KS]
        self.logw = np.log(np.array(K_PRIOR))

    def step_frame(self, x, fs, band, dt=1.0):
        """One sub-frame: build both rails once, update every hypothesis."""
        # Envelope HPS over the union of all hypothesis ranges (one call,
        # sliced per k by interpolation).
        eg, es = hps_grid(x, fs, lo=max(2.0, self.flo*min(KS)),
                          hi=self.fhi*max(KS)+1.0, nh=self.nh_env,
                          step=self.step, band=band)
        es = es/max(float(np.median(es)), 1e-30)      # floor -> ~1
        ac = acc_comb_surface(x, fs, self.fgrid, nh=self.nh_acc)
        ac = ac/max(float(np.median(ac)), 1e-30)
        df = fs/len(x)                                # raw FFT bin width
        La = ac**self.ba                              # shared by all k
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = np.diag([(0.004*self.fn*dt)**2, (0.010*self.fn*dt)**2])
        for i, k in enumerate(KS):
            mu = F@self.mu[i]
            P = F@self.P[i]@F.T+Q
            env = np.interp(self.fgrid*k, eg, es, left=1.0, right=1.0)
            L = (env**self.be)*La                     # joint, common scale
            sig = float(np.sqrt(P[0, 0]))
            pri = np.exp(-0.5*((self.fgrid-mu[0])/max(sig, 1e-9))**2)
            # hypothesis marginal: common normaliser cancels across k
            m = float(np.sum(pri*L))*self.step/(sig*np.sqrt(2*np.pi))
            self.logw[i] += np.log(max(m, 1e-300))
            # Kalman measurement: local moments of the joint surface around
            # the prior-gated argmax (mean -> z, spread -> R). A flat or
            # contested surface yields a large R and a weak update - the
            # uncertainty is carried instead of hidden.
            j = int(np.argmax(L*pri))
            half = max(0.015*self.fgrid[j], 1.2*df, 5*self.step)
            loc = np.abs(self.fgrid-self.fgrid[j]) <= half
            Ll = np.where(loc, L, 0.0)
            s = float(Ll.sum())
            if s > 0:
                z = float(np.sum(self.fgrid*Ll)/s)
                R = float(np.sum((self.fgrid-z)**2*Ll)/s)+self.step**2/12
                R = max(R, (5e-4*z)**2)
                S = P[0, 0]+R
                K = P[:, 0]/S
                mu = mu+K*(z-mu[0])
                P = P-np.outer(K, P[0, :])
            self.mu[i], self.P[i] = mu, P
        self.logw -= self.logw.max()

    def estimate(self, dt_extrap=0.0):
        """Point estimate = MAP-hypothesis mean, extrapolated by dt_extrap
        seconds along f_dot (the wrapper passes last-frame-centre -> record
        centre, so a ramp is reported at its record-MEAN speed, matching
        what a whole-record estimator measures). sigma = mixture spread
        about that point, so hypothesis disagreement inflates it."""
        w = np.exp(self.logw)
        w /= w.sum()
        i = int(np.argmax(w))
        fr = float(self.mu[i][0]+dt_extrap*self.mu[i][1])
        var = 0.0
        for j in range(len(KS)):
            P = self.P[j]
            fj = float(self.mu[j][0]+dt_extrap*self.mu[j][1])
            vj = float(P[0, 0]+2*dt_extrap*P[0, 1]+dt_extrap**2*P[1, 1])
            var += w[j]*(vj+(fj-fr)**2)
        return fr, float(SIGMA_CAL*np.sqrt(var)), KS[i], w


def estimate_speed3(x, fs, band, fr_nominal, span=0.35, nframes=3, frac=0.6,
                    **kw):
    """Per-acquisition posterior speed: split the record into nframes
    overlapping sub-frames, accumulate both rails, collapse. Returns
    (fr, sigma, k_map, weights). Needs a nominal (defines the search span);
    extract2 keeps estimate_speed2 for the fr_nominal=None path."""
    n = len(x)
    win = n if nframes == 1 else int(frac*n)
    hop = (n-win)//max(nframes-1, 1)
    sp = SpeedPosterior(fr_nominal, span=span, **kw)
    dt = max(hop, 1)/fs
    for t in range(nframes):
        sp.step_frame(x[t*hop:t*hop+win], fs, band, dt=dt)
    last_centre = ((nframes-1)*hop+win/2)/fs
    return sp.estimate(dt_extrap=n/(2*fs)-last_centre)


def _speed_dual_evidence(x, fs, band, fr_nominal):
    """The pre-speed3 estimator, kept verbatim: envelope HPS gated by the
    acceleration comb. Serves as extract2's fallback when the posterior is
    contested or wide, and as the whole answer when no nominal is known
    (estimate_speed3 needs one to define its search span)."""
    fr_e, conf_e = estimate_speed2(x, fs, band, fr_nominal)
    if fr_nominal:
        # DUAL EVIDENCE. The envelope comb is sharp but treacherous: on an
        # outer-race fault the envelope contains a BPFO-spaced comb and no
        # shaft comb at all, and BPFO/3 (1.19x for a 6205) lands inside the
        # search span and wins with GOOD confidence. A confidence gate cannot
        # catch a confidently wrong estimator; only independent evidence can.
        # The acceleration comb defines what "shaft speed" means, so the
        # envelope answer is accepted only when the acc comb corroborates it.
        # The comb search runs on a FULL-LENGTH FFT, not the Welch amplitude:
        # Welch's 2.93 Hz bins cannot resolve a 6.4 Hz rotor's comb (the
        # fleet's slow mixer sat at 35% error until this), while the full
        # record gives 0.5 Hz bins from the same samples for one extra FFT.
        af = np.abs(np.fft.rfft(x*np.hanning(len(x))))
        ff = np.fft.rfftfreq(len(x), 1/fs)
        fr_a, conf_a = acc_comb_speed(ff, af, fr_nominal)
        agree = conf_a >= 1.3 and abs(fr_e-fr_a) < 0.03*fr_a
        if conf_e >= 1.8 and (agree or conf_a < 1.3):
            fr, conf = fr_e, conf_e
            tier = 1 if (conf_e >= 3.0 and agree) else 2
        elif conf_a >= 1.3:
            fr, conf = fr_a, conf_a
            tier = 2 if conf_a >= 3.0 else 3
        else:
            fr, conf, tier = fr_nominal, min(conf_e, conf_a), 3
    else:
        fr, conf = fr_e, conf_e
        tier = 1 if conf >= 3.0 else (2 if conf >= 1.8 else 3)
    if not (2 < fr < 300): fr, tier = (fr_nominal or 30.0), 3
    return fr, conf, tier


def extract2(x, fs, band_key='default', fr_nominal=None,
             fr_override=None) -> Extract:
    """Rig-agnostic extraction. fr_override lets a tracker (Viterbi, order
    tracking) supply the speed; the estimator's answer is still computed so
    confidence tiers stay meaningful."""
    nps = nperseg_for(fs)
    f, p = welch(x, fs=fs, nperseg=min(nps, len(x)),
                 noverlap=min(nps, len(x))//2, window='hann',
                 scaling='spectrum', detrend='constant')
    a = np.sqrt(np.maximum(p, 0))*np.sqrt(2)

    m = (f >= 10) & (f <= 1000)
    v = a[m]*9.80665/(2*np.pi*np.maximum(f[m], 1e-9))*1000.0
    vel_rms = float(np.sqrt(0.5*np.sum(v**2)))

    band = band_for(band_key, x, fs)
    fr, srel = None, None
    if BAYES_SPEED and fr_nominal:
        # Speed from the Bayesian posterior (see speed3 section above).
        # Guards first (contested hypotheses or a wide posterior fall through
        # to the dual-evidence gate, which also owns the fr_nominal=None
        # path), then SIGMA drives conf and tier. The tier thresholds are in
        # sigma units, anchored to what the conf-based tiers DELIVERED, via
        # the dev-frozen ratio err/sigma p95 ~ 0.7 (post-SIGMA_CAL): srel =
        # sigma/fr <= 0.001 delivers ~0.07% p95 (old tier1 assumed 0.05%);
        # srel <= 0.01 delivers ~0.7% p95 (old tier2 measured 0.69% on the
        # bench corpus). conf = 1.8*(0.01/srel) pins the tier2/3 boundary to
        # the historical conf=1.8 threshold and stays monotone in precision;
        # it is a reporting field only (Gate scores spectra, not conf).
        fr_b, sig, _k, w = estimate_speed3(x, fs, band, fr_nominal)
        sr = sig/max(fr_b, 1e-9)
        if float(np.max(w)) >= 0.5 and sr <= 0.02 and 2 < fr_b < 300:
            fr, srel = fr_b, sr
            conf = 1.8*0.01/max(sr, 1e-5)
            tier = 1 if sr <= 0.001 else (2 if sr <= 0.01 else 3)
    if fr is None:
        fr, conf, tier = _speed_dual_evidence(x, fs, band, fr_nominal)
    if fr_override is not None: fr = float(fr_override)

    e = envelope_banded(x, fs, band)
    fe, pe = welch(e, fs=fs, nperseg=min(nps, len(e)),
                   noverlap=min(nps, len(e))//2, window='hann',
                   scaling='spectrum', detrend='constant')
    ae = np.sqrt(np.maximum(pe, 0))*np.sqrt(2)

    acc = bin_orders(f, a, fr, EDGES)
    env = bin_orders(fe, ae, fr, ENV_EDGES)
    ar, ak, ac = moments(x); er, ek, ec = moments(e)
    bandmap = np.minimum((np.arange(NBINS)*NBANDS)//NBINS, NBANDS-1)
    band_db = np.array([to_db(np.sqrt(np.mean(acc[bandmap == b]**2)))
                        for b in range(NBANDS)])
    # On the posterior path the calibrated sigma IS the speed error - feed it
    # to coherence_limit directly (clipped to the ladder's range) instead of
    # the tier-quantized assumption the fallback path must still use.
    speed_err = min(max(srel, 0.0005), 0.02) if srel is not None else \
        (0.0005 if tier == 1 else (0.003 if tier == 2 else 0.02))
    return Extract(acc, env, to_u8(acc), to_u8(env), fr, conf, tier,
                   coherence_limit(3.0, speed_err, fr),
                   ar, vel_rms, ak, ac, er, ek, ec, band_db)
