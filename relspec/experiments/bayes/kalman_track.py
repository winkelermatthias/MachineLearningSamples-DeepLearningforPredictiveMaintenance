"""Experiment A: per-track Kalman (level+slope) vs the shipped static z.

Two detectors over the same per-track E_dB series (E_dB = 10*log10 of the
pattern's owned spectral energy, one number per track per acquisition):

  StaticZ      faithful reimplementation of the math in
               relspec_service/health.py update_z — Welford mean/std taught
               only on gate-quiet, non-anomalous frames, z clamped to +/-10,
               std floored at 0.25 dB, reported after 10 taught frames.
               Deliberately NOT imported from the server: this file must run
               without a database, and the point is to compare the *math*.

  KalmanTrack  2-state linear-Gaussian model, hand-rolled:
                   level_{k+1} = level_k + slope_k + w_l
                   slope_{k+1} = slope_k + w_s,     y_k = level_k + v_k
               Change is scored two ways, both ~N(0,1) under H0 and both
               one-sided (fault energy goes UP):
                 innov_sig = innovation / sqrt(S)   (signed sqrt of the NIS)
                 slope_sig = slope / sqrt(P_ss)     (ramp evidence)
                 raw_disp  = the shipped z itself, embedded verbatim (an
                             internal StaticZ). Two failed attempts taught us
                             why: a level+slope filter TRACKS a ramp, so
                             innovations vanish exactly when the fault grows
                             steadily; and every homegrown displacement scale
                             (R-based, EWMA spread) lost to the Welford-over-
                             taught-frames scale at matched FA — the static
                             baseline is genuinely well built. So the honest
                             experiment is: keep z as one channel and measure
                             what the Kalman states ADD on top of it.
                 disp_sig  = filtered-level displacement from the warmup-
                             frozen level posterior (reported, off by
                             default: it lags raw_disp by the filter's time
                             constant and never beat it).
               stat = max over `channels`. The slope state with variance also
               gives a predicted frames-to-threshold with a credible interval
               by sampling the (level, slope) posterior — the thing the static
               baseline cannot produce at all.

Neither detector's raw threshold is trusted: the benchmark sweeps both to the
same empirical false-alarm count on quiet tracks and compares latency there.
"""
from __future__ import annotations
import numpy as np

BASELINE_MIN_N = 10          # mirrors health.BASELINE_MIN_N
MAD_FLOOR_DB = 0.25          # mirrors health.MAD_FLOOR_DB


class StaticZ:
    """One track's baseline, exactly as update_z keeps it in pattern_baseline:
    row = (n, mean, sd); insert with n=1/0 and sd=1.0; Welford on taught
    frames only; teach iff the gate was quiet AND (no z yet or |z| < 3)."""

    def __init__(self):
        self.row = None                       # (n, mean, sd)

    def step(self, e_db: float, quiet: bool) -> float | None:
        z = None
        if self.row is not None and self.row[0] >= BASELINE_MIN_N:
            n, mean, sd = self.row
            z = float(np.clip((e_db - mean) / max(sd, MAD_FLOOR_DB), -10, 10))
        teach = quiet and (z is None or abs(z) < 3.0)
        if self.row is None:
            self.row = (1 if teach else 0, e_db, 1.0)
        elif teach:
            n, mean, sd = self.row
            n += 1
            var = sd * sd
            delta = e_db - mean
            mean += delta / n
            var = ((n - 1) * var + delta * (e_db - mean)) / n
            self.row = (n, mean, float(np.sqrt(max(var, 0.0))))
        return z


class KalmanTrack:
    """2-state (level, slope) Kalman filter over one track's E_dB series.

    Process/measurement noises scale with R, which is re-estimated once from
    the first `warmup` observations (robust: MAD of first differences), so the
    same dimensionless tuning serves quiet and noisy tracks. Missing frames
    are pure time updates — the clock keeps ticking, uncertainty grows."""

    def __init__(self, q_level=0.005, q_slope=0.001, warmup=10,
                 r_init=1.0, r_lo=MAD_FLOOR_DB ** 2, r_hi=25.0,
                 r_beta=0.05,
                 channels=('raw_disp', 'slope_sig', 'innov_sig')):
        self.q_l, self.q_s = q_level, q_slope
        self.warmup, self.r_lo, self.r_hi = warmup, r_lo, r_hi
        self.r_beta = r_beta
        self.channels = channels
        self.R = r_init
        self.zb = StaticZ()                  # raw_disp channel = shipped z
        self.m = None                        # state mean [level, slope]
        self.P = None                        # state covariance 2x2
        self.n = 0                           # observations absorbed
        self._ys = []                        # first observations, for R
        self.base = None                     # (level, P_ll) frozen at warmup

    def _Q(self):
        return self.R * np.array([[self.q_l, 0.0], [0.0, self.q_s]])

    def step(self, y: float | None, quiet: bool = True) -> dict | None:
        """One frame. y=None is a missed frame (time update only). `quiet`
        is the gate's verdict for this frame — the same signal update_z gets
        at ingest; baselines and noise scales teach only on quiet frames."""
        if self.m is not None:
            l, s = self.m
            self.m = np.array([l + s, s])
            F = np.array([[1.0, 1.0], [0.0, 1.0]])
            self.P = F @ self.P @ F.T + self._Q()
        if y is None:
            return None
        if self.m is None:
            self.m = np.array([y, 0.0])
            self.P = np.array([[4.0 * self.R, 0.0], [0.0, 0.25 * self.R]])
            self.n = 1
            self._ys = [y]
            self.zb.step(y, quiet)
            return dict(level=y, slope=0.0, slope_var=self.P[1, 1],
                        nis=0.0, innov_sig=0.0, slope_sig=0.0,
                        disp_sig=0.0, raw_disp=0.0, stat=0.0)
        # measurement update, H = [1, 0]. The STATE update is Huberized:
        # an innovation beyond 3 sigma moves the state as if it were 3 sigma,
        # so one ownership flip cannot yank level/slope (and so the slope the
        # crossing prediction runs on stays a trend, not flip chasing). The
        # detection statistic below uses the RAW innovation on purpose.
        S = self.P[0, 0] + self.R
        innov = y - self.m[0]
        innov_u = float(np.clip(innov, -3.0 * np.sqrt(S), 3.0 * np.sqrt(S)))
        K = self.P[:, 0] / S
        self.m = self.m + K * innov_u
        IKH = np.eye(2) - np.outer(K, [1.0, 0.0])
        self.P = IKH @ self.P @ IKH.T + np.outer(K, K) * self.R  # Joseph form
        self.n += 1
        if self.n <= self.warmup:
            self._ys.append(y)
            if self.n == self.warmup:
                d = np.diff(self._ys)
                if d.size:
                    # max(MAD, variance) estimate: MAD alone undersells R on
                    # BIMODAL tracks (pattern-energy ownership flips between
                    # frames are the rule, not the exception, and a robust
                    # estimator writes them off as outliers — then every flip
                    # is a 30-sigma innovation and the track false-alarms
                    # forever). Variance absorbs the flips like the static
                    # baseline's Welford std does.
                    sd = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
                    v2 = float(np.var(d)) / 2.0
                    r_new = float(np.clip(max(sd * sd, v2),
                                          self.r_lo, self.r_hi))
                    # P grew out of r_init; rescale so the posterior matches
                    # the track's actual noise, not the placeholder's
                    self.P *= r_new / self.R
                    self.R = r_new
                self.base = (float(self.m[0]), float(self.P[0, 0]))
        nis = innov * innov / S
        innov_sig = innov / np.sqrt(S)
        z = self.zb.step(y, quiet)
        raw_disp = 0.0 if z is None else z
        disp_sig = 0.0
        if self.base is not None:
            l0, v0 = self.base
            disp_sig = (self.m[0] - l0) / np.sqrt(max(self.P[0, 0] + v0,
                                                      1e-12))
        # keep R honest on tracks whose noise the first frames under-sold
        # (bimodal energy assignment is common). Taught only on quiet frames
        # that do not look anomalous — update_z's own teaching rule.
        if self.n > self.warmup and quiet and abs(innov_sig) < 3.0:
            self.R = float(np.clip((1 - self.r_beta) * self.R
                                   + self.r_beta * innov * innov,
                                   self.r_lo, self.r_hi))
        slope_sig = self.m[1] / np.sqrt(max(self.P[1, 1], 1e-12))
        out = dict(level=float(self.m[0]), slope=float(self.m[1]),
                   slope_var=float(self.P[1, 1]), nis=float(nis),
                   innov_sig=float(innov_sig), slope_sig=float(slope_sig),
                   disp_sig=float(disp_sig), raw_disp=float(raw_disp))
        out['stat'] = float(max(out[ch] for ch in self.channels))
        return out

    def predict_crossing(self, threshold: float, nsamp=4000, seed=0):
        """Frames until level crosses `threshold`, from the current posterior:
        sample (level, slope), t = (T-l)/s for s>0 else never. Returns
        (median, p10, p90) in frames — p90 may be inf when a big share of the
        posterior says 'never' — or None before the filter has state."""
        if self.m is None:
            return None
        rng = np.random.default_rng(seed)
        ls = rng.multivariate_normal(self.m, self.P, size=nsamp)
        l, s = ls[:, 0], ls[:, 1]
        t = np.where(l >= threshold, 0.0,
                     np.where(s > 1e-9, (threshold - l) / np.maximum(s, 1e-9),
                              np.inf))
        med = float(np.median(t))
        lo, hi = np.percentile(t, [10, 90])
        return med, float(lo), float(hi)
