"""EXPERIMENT B: Bayesian speed posterior with a harmonic-index mixture.

estimate_speed2's failure mode is structural, not statistical: on an outer
race fault the envelope comb is BPFO-spaced, BPFO/3 (~1.19x for a 6205)
lands inside the search span, and the envelope HPS is CONFIDENTLY wrong -
a confidence gate cannot catch it. extract2 patches this with a hard
two-rail if/else. This experiment replaces the gate with the thing the gate
approximates: a joint posterior over shaft speed in which the envelope
rail's harmonic index is an explicit latent variable.

  state       (f_r, f_r_dot), one small Kalman per harmonic hypothesis
              k in {1/3, 1/2, 1, 2, 3}  (Rao-Blackwellized mixture)
  evidence    the SAME two surfaces the shipped code builds:
                envelope HPS  - dsp2.hps_grid, reused verbatim
                acc comb      - pipeline2.acc_comb_speed's score, computed
                                as a surface instead of an argmax
  update      per sub-frame of one acquisition, so evidence from BOTH
              rails accumulates before the mixture collapses

Under hypothesis k the envelope comb sits at k*f_r, so its surface is read
at k*f; the acc comb DEFINES shaft speed and is read at f for every k. The
joint pseudo-likelihood is the product of floor-normalised surfaces raised
to small powers; a rail with no peak near f contributes ~1 and abstains.
The BPFO/3 lock then loses automatically: its envelope peak has no acc
support, while the true speed has a 6-harmonic acc comb behind it - no
thresholds, no if/else.

Read-only reuse of relspec modules; nothing shipped is touched.
"""
from __future__ import annotations
import numpy as np
from relspec.dsp2 import hps_grid

KS = (1/3., 1/2., 1.0, 2.0, 3.0)          # envelope harmonic hypotheses
# k=2,3 exist to catch envelope harmonic locks, but they are also where the
# bearing-comb alias hides (BPFO 3.58x read as "3x shaft" puts f at 1.19x
# true - the same trap in new clothes), so they start with a real handicap.
K_PRIOR = (0.045, 0.045, 0.87, 0.02, 0.02)

# +/-2 sigma empirical coverage calibration, frozen from the dev slice
# (seeds 5xxx, disjoint from the benchmark corpus). After the record-centre
# extrapolation the model sigma is conservative (dev err/sigma p95 = 0.42,
# max 0.73): the extrapolation variance term over-counts because the same
# frames that set f also set f_dot. 0.6 keeps a 1.6x margin over the worst
# dev case while making the interval informative.
SIGMA_CAL = 0.6


def acc_comb_surface(x, fs, fgrid, nh=6):
    """pipeline2.acc_comb_speed's score over an explicit grid: geometric
    mean of the full-record FFT magnitude over nh shaft harmonics. Kept as
    a surface because the posterior consumes the whole row, not the max."""
    a = np.abs(np.fft.rfft(x*np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1/fs)
    sc = np.ones_like(fgrid)
    for h in range(1, nh+1):
        sc *= np.maximum(np.interp(fgrid*h, f, a), 1e-12)
    return sc**(1/nh)


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
    disjoint from the benchmark corpus; see bench_speed.py."""

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
        # sliced per k by interpolation). dsp2.hps_grid reused verbatim.
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


def speed_bayes(x, fs, band, fr_nominal, span=0.35, nframes=3, frac=0.6,
                **kw):
    """Drop-in per-acquisition estimator: split the record into nframes
    overlapping sub-frames, accumulate, collapse. Returns
    (fr, sigma, k_map, weights)."""
    n = len(x)
    win = n if nframes == 1 else int(frac*n)
    hop = (n-win)//max(nframes-1, 1)
    sp = SpeedPosterior(fr_nominal, span=span, **kw)
    dt = max(hop, 1)/fs
    for t in range(nframes):
        sp.step_frame(x[t*hop:t*hop+win], fs, band, dt=dt)
    last_centre = ((nframes-1)*hop+win/2)/fs
    return sp.estimate(dt_extrap=n/(2*fs)-last_centre)
