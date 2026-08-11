"""Fast Spectral Correlation and the Improved Envelope Spectrum (IES).

Antoni's STFT-based spectral-correlation estimator (Fast-SC, MSSP 2017) in a
plain vectorised form, normalised to cyclic spectral coherence. The physics:
second-order cyclostationarity at cyclic frequency alpha means spectral
components alpha apart are CORRELATED. With an absolute-time-phase STFT
X_i(f), the cross product X_i(f) X_i*(f - p*dF) of bins p apart rotates
across frames at exp(+j2pi(alpha - p*dF) t_i), so a DFT over the frame index
resolves alpha to fs/(K*hop) while the bin shift p extends the alpha range
far beyond the frame-rate Nyquist. The p=0 term alone is the classical
cyclic modulation spectrum, blind above one STFT bin of alpha; the bin
shifts are what make a 100+ Hz bearing tone reachable from a 46.9 Hz grid.

Two deliberate simplifications versus the paper, both flagged honestly:
  - only the NEAREST shift p = round(alpha/dF) is evaluated per alpha (the
    "scanning" light form). An off-grid carrier smears over adjacent bin
    pairs, so some cross-shift energy is left on the table - an amplitude
    loss the coherence normalisation mostly forgives.
  - the frame-constant (DC across frames) part of every cross product is
    removed before the frame DFT. A stationary signal has genuinely
    correlated NEIGHBOURING bins (the hann window couples bins +/-2), which
    would otherwise print false coherence ridges at alpha = p*dF; removal
    nulls only a ~1 Hz sliver of alpha around each multiple of dF.

The Improved Envelope Spectrum is |coherence| averaged over the spectral
axis. The bet under test: a weak fault whispering coherently across a wide
carrier band integrates to a visible alpha line even when no single band
clears the envelope spectrum's noise floor.
"""
from __future__ import annotations
import numpy as np

def fast_sc(x, fs, nw=256, hop=None, alpha_max=300.0):
    """Cyclic spectral coherence. Returns (alpha_hz, f_hz, gamma), gamma
    shaped (n_alpha, n_f), real, ~[0,1]; entries at f below the p-th bin
    shift are zero (no valid cross product exists there)."""
    x = np.asarray(x, np.float64)
    x = x - x.mean()
    if hop is None:
        hop = nw // 8
    n = x.size
    K = (n - nw) // hop + 1
    if K < 32:
        raise ValueError('record too short for Fast-SC')
    w = np.hanning(nw)
    frames = np.lib.stride_tricks.sliding_window_view(x, nw)[::hop][:K]
    X = np.fft.rfft(frames * w, axis=1)
    F = nw // 2 + 1
    f = np.arange(F) * (fs / nw)
    tk = np.arange(K) * (hop / fs)
    X = X * np.exp(-2j * np.pi * tk[:, None] * f[None, :])   # absolute-time phase
    fw = np.hanning(K)                  # frame window: alpha-leakage control
    fwsum = fw.sum()
    S0 = (fw[:, None] * np.abs(X)**2).sum(0) / fwsum         # mean PSD, unnormalised
    dF = fs / nw
    da = fs / (K * hop)
    alphas = np.arange(1, int(np.floor(alpha_max / da)) + 1) * da
    p_of = np.rint(alphas / dF).astype(int)
    w2 = w * w
    nn = np.arange(nw)
    gamma = np.zeros((alphas.size, F))
    for p in np.unique(p_of):
        sel = np.flatnonzero(p_of == p)
        eps = alphas[sel] - p * dF
        C = X[:, p:] * np.conj(X[:, :F - p])
        C = C - (fw[:, None] * C).sum(0) / fwsum             # stationary ridge out
        S = (np.exp(-2j * np.pi * eps[:, None] * tk[None, :])
             @ (fw[:, None] * C)) / fwsum
        # intra-frame window loss at the residual eps, corrected so the
        # alpha axis carries no |alpha - p*dF| scalloping
        Rw = np.abs(np.exp(-2j * np.pi * eps[:, None] * nn[None, :] / fs)
                    @ w2) / w2.sum()
        denom = np.sqrt(S0[p:] * S0[:F - p]) + 1e-20
        gamma[np.ix_(sel, np.arange(p, F))] = np.minimum(
            np.abs(S) / denom / np.maximum(Rw, 0.25)[:, None], 1.5)
    return alphas, f, gamma

def ies_full(f, gamma, fmin, fmax):
    """Improved Envelope Spectrum: |coherence| averaged over [fmin, fmax]."""
    m = (f >= fmin) & (f <= fmax)
    return gamma[:, m].mean(axis=1)

def ies_top_band(alpha, f, gamma, fmin, fmax, nbands=8, alpha_min=10.0):
    """Crude, TARGET-AGNOSTIC IESFOgram: split [fmin, fmax] into nbands equal
    slices, score each by its mean squared coherence over alpha >= alpha_min
    (peaky slices win; no fault frequency is consulted, so healthy records
    get no free lift), integrate |coherence| over the best slice only.
    Returns (ies, (band_lo, band_hi))."""
    edges = np.linspace(fmin, fmax, nbands + 1)
    am = np.flatnonzero(alpha >= alpha_min)
    best, best_score = (fmin, fmax), -1.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        cols = np.flatnonzero((f >= lo) & (f < hi))
        if cols.size == 0:
            continue
        sc = float((gamma[np.ix_(am, cols)]**2).mean())
        if sc > best_score:
            best_score, best = sc, (float(lo), float(hi))
    m = (f >= best[0]) & (f < best[1])
    return gamma[:, m].mean(axis=1), best

def selfcheck(fs=12000.0, seconds=4.0, a0=87.3, seed=0):
    """Known-alpha cyclostationary signal - bandpassed white noise, AM at a0,
    buried in stationary noise - plus a stationary control. The go/no-go for
    trusting the estimator at all: the IES must peak at a0 within one alpha
    bin and the control must stay flat."""
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    t = np.arange(n) / fs
    c = rng.normal(0, 1, n)
    C = np.fft.rfft(c)
    fr_ = np.fft.rfftfreq(n, 1 / fs)
    C[(fr_ < 2500) | (fr_ > 3500)] = 0
    c = np.fft.irfft(C, n)
    c /= c.std() + 1e-12
    x = (1 + 0.9 * np.cos(2 * np.pi * a0 * t)) * c + 2.0 * rng.normal(0, 1, n)
    alpha, f, g = fast_sc(x, fs, alpha_max=300.0)
    fhi = 0.94 * fs / 2
    e = ies_full(f, g, 400.0, fhi)
    m = np.flatnonzero(alpha >= 10.0)
    j = m[int(np.argmax(e[m]))]
    y = c + 2.0 * rng.normal(0, 1, n)                        # stationary control
    a2, f2, g2 = fast_sc(y, fs, alpha_max=300.0)
    e2 = ies_full(f2, g2, 400.0, fhi)
    m2 = a2 >= 10.0
    return dict(alpha_true=float(a0), alpha_hat=float(alpha[j]),
                da=float(alpha[1] - alpha[0]), peak=float(e[j]),
                floor_med=float(np.median(e[m])),
                stationary_max=float(e2[m2].max()),
                stationary_med=float(np.median(e2[m2])))
