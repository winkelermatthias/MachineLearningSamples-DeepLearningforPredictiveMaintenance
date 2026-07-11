"""Grover-inspired amplification primitives (H4).

Grover = oracle marks target + iterative rotation amplifies it relative to
the mean. Classical transfers here:

1. slip_scan: rotate the demodulation frame over candidate slip values
   until the bearing tone amplitude is maximized. The scan curve peak IS
   the mean slip (a load/health indicator); the peak sharpness is a
   detection feature. This is "rotating until the marked state aligns".

2. predictive_aligned_average (PPA): bearing tones are not phase-locked
   (phase random-walks with slip jitter) so plain coherent averaging
   destroys them. PPA aligns each block by the phase measured in the
   PREVIOUS block: a real narrowband tone has slowly varying phase and
   survives alignment (~N gain), white noise does not (~sqrt(N)).
   Statistical control: permutation null over block order (exchangeable
   for noise, continuity-breaking for a drifting tone). Report a z-score,
   never a raw amplitude.
"""
import numpy as np


def zoom_order_dft(xa, spr, orders, chunk=32):
    """Amplitude of the angular-resampled signal at arbitrary fractional
    orders (Goertzel-style zoom). xa: angular samples, spr per rev."""
    xa = np.asarray(xa, float)
    xa = xa - xa.mean()
    n = np.arange(len(xa))
    out = np.empty(len(orders))
    for i in range(0, len(orders), chunk):
        o = np.asarray(orders[i:i + chunk])[:, None]
        e = np.exp(-2j * np.pi * o * n[None, :] / spr)
        out[i:i + chunk] = 2 * np.abs(e @ xa) / len(xa)
    return out


def slip_scan(xa, spr, o_kin, span=0.03, n=601):
    """Scan slip s in [-span, +span] around kinematic order o_kin.
    Returns dict: slip estimate (parabolic-interpolated), amplitude,
    sharpness (peak / median of scan curve), and the curve itself."""
    s = np.linspace(-span, span, n)
    A = zoom_order_dft(xa, spr, o_kin * (1 + s))
    i = int(np.argmax(A))
    if 0 < i < n - 1:
        d = 0.5 * (A[i - 1] - A[i + 1]) / (A[i - 1] - 2 * A[i] + A[i + 1] + 1e-15)
        d = float(np.clip(d, -0.5, 0.5))
    else:
        d = 0.0
    step = s[1] - s[0]
    return {"slip": float(s[i] + d * step),
            "amp": float(A[i]),
            "sharpness": float(A[i] / (np.median(A) + 1e-15)),
            "curve_s": s, "curve_a": A}


def ppa_zscore(zb, n_perm=200, seed=20260709):
    """Predictive phase-aligned average with permutation null.
    zb: complex per-block values at one order bin (from block_spectra).
    Returns (zscore, aligned_amp, null_mean, null_std)."""
    zb = np.asarray(zb, complex)
    rng = np.random.default_rng(seed)

    def aligned(z):
        rot = np.exp(-1j * np.angle(z[:-1]))
        return float(np.abs((z[1:] * rot).mean()))

    obs = aligned(zb)
    null = np.array([aligned(rng.permutation(zb)) for _ in range(n_perm)])
    mu, sd = float(null.mean()), float(null.std() + 1e-15)
    return (obs - mu) / sd, obs, mu, sd


def ppa_spectrum_zscores(Z, order_idx, n_perm=200):
    """PPA z-score at each requested bin index of Z (nblocks x nbins)."""
    return {int(i): ppa_zscore(Z[:, int(i)], n_perm)[0] for i in order_idx}


def template_pursuit(A, orders, templates, n_iter=5):
    """Grover-as-matching-pursuit over fault templates (harmonic combs at
    rational base orders). Each iteration: score all templates against the
    residual amplitude spectrum, take the best (oracle marks it), subtract
    its projection (inversion about the mean, crudely), repeat.

    templates: list of (name, [orders...]). Returns list of
    (name, score, iteration). Scores are floor-normalized sums so one huge
    line cannot fake a family (same lesson as the comb capture guard)."""
    A = np.asarray(A, float).copy()
    floor = np.median(A) + 1e-15
    out = []
    for it in range(n_iter):
        best, best_s = None, 0.0
        for name, ords in templates:
            idx = [int(np.argmin(np.abs(orders - o))) for o in ords
                   if o <= orders[-1]]
            s = float(np.sum(np.log1p(A[idx] / floor)))
            if s > best_s:
                best, best_s, best_idx = name, s, idx
        if best is None or best_s < 2.0:
            break
        out.append((best, best_s, it))
        A[best_idx] = floor  # subtract the marked component
    return out
