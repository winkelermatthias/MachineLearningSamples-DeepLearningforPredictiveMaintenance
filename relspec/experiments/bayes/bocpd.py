"""Bayesian online change-point detection (Adams & MacKay 2007).

Student-t predictive from a Normal-Inverse-Gamma conjugate prior over a
scalar observation stream, constant hazard, exact run-length recursion with
tail pruning. The stream here is the v1 Gate's own scalar score: the
experiment asks whether a run-length posterior beats the shipped
threshold+persistence+CUSUM decision layer ON THE SAME EVIDENCE.

Decision statistic: p_recent = P(run length < w) after absorbing x_t, i.e.
the posterior probability that a changepoint happened within the last w
frames. Alarm = p_recent >= threshold (the benchmark sweeps the threshold).

State is four parameter arrays plus the run-length posterior, pruned to the
support that carries mass: small enough for firmware if it earns promotion.
"""
from __future__ import annotations
import numpy as np
from scipy.special import gammaln


def fit_prior(scores, kappa0=1.0, alpha0=1.0, std_floor=0.05):
    """NIG hyperparameters matched to a warmup slice: predictive location =
    robust center, predictive scale ~ MAD std, floored so a freakishly quiet
    warmup cannot make the detector hair-triggered. alpha0=1 gives a nu=2
    Student-t predictive - heavy tails, so one bad frame is absorbed rather
    than declared a changepoint."""
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    mu0 = float(np.median(s))
    std = max(1.4826*float(np.median(np.abs(s-mu0))), std_floor)
    # predictive scale^2 = beta*(kappa+1)/(alpha*kappa)  ->  solve for beta
    beta0 = std*std*alpha0*kappa0/(kappa0+1.0)
    return mu0, beta0


class BOCPD:
    def __init__(self, mu0, beta0, kappa0=1.0, alpha0=1.0,
                 hazard=1.0/300.0, w=8, rmax=512, prune=1e-12):
        self.mu0, self.kappa0 = float(mu0), float(kappa0)
        self.alpha0, self.beta0 = float(alpha0), float(beta0)
        self.h, self.w = float(hazard), int(w)
        self.rmax, self.prune = int(rmax), float(prune)
        self.R = np.array([1.0])                    # P(run length = r | x_1:t)
        self.mu = np.array([self.mu0])
        self.kappa = np.array([self.kappa0])
        self.alpha = np.array([self.alpha0])
        self.beta = np.array([self.beta0])
        self.t = 0

    def _log_pred(self, x):
        """Student-t predictive per run length: nu = 2*alpha, location mu,
        scale^2 = beta*(kappa+1)/(alpha*kappa)."""
        nu = 2.0*self.alpha
        var = self.beta*(self.kappa+1.0)/(self.alpha*self.kappa)
        return (gammaln((nu+1.0)/2.0) - gammaln(nu/2.0)
                - 0.5*np.log(nu*np.pi*var)
                - (nu+1.0)/2.0*np.log1p((x-self.mu)**2/(nu*var)))

    def step(self, x):
        """Absorb one observation; return the changepoint evidence."""
        x = float(x)
        pred = np.exp(np.maximum(self._log_pred(x), -700.0))
        growth = self.R*pred*(1.0-self.h)           # run continues
        cp0 = float(np.sum(self.R*pred*self.h))     # run resets
        R = np.concatenate(([cp0], growth))
        tot = R.sum()
        if tot <= 0 or not np.isfinite(tot):        # degenerate frame: reset
            R = np.zeros(len(R)); R[0] = 1.0
        else:
            R = R/tot
        # NIG recursion using the PRE-update parameters; index 0 = fresh prior
        d = x-self.mu
        self.beta = np.concatenate(([self.beta0],
                                    self.beta+self.kappa*d*d/(2.0*(self.kappa+1.0))))
        self.mu = np.concatenate(([self.mu0], (self.kappa*self.mu+x)/(self.kappa+1.0)))
        self.kappa = np.concatenate(([self.kappa0], self.kappa+1.0))
        self.alpha = np.concatenate(([self.alpha0], self.alpha+0.5))
        # prune: drop zero-mass tail, cap support at rmax (overflow mass folds
        # into the last kept cell, which then represents "very long run")
        keep = int(np.max(np.nonzero(R > self.prune)[0]))+1 if np.any(R > self.prune) else 1
        keep = min(max(keep, 1), self.rmax)
        if len(R) > keep:
            R[keep-1] += R[keep:].sum()
        R = R[:keep]
        s = R.sum()
        self.R = R/s if s > 0 else R
        self.mu, self.kappa = self.mu[:keep], self.kappa[:keep]
        self.alpha, self.beta = self.alpha[:keep], self.beta[:keep]
        self.t += 1
        p_recent = float(self.R[:self.w].sum())
        return dict(p_recent=p_recent, cp_now=float(self.R[0]),
                    rl_mode=int(np.argmax(self.R)))
