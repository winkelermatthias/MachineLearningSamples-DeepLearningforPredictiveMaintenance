"""External baseline panel (PHASE3 A3) + shared fs-invariant features.

B1  classical envelope analyzer: fast-kurtogram band selection ->
    Hilbert envelope -> envelope spectrum -> significance-tested peak
    at nominal defect orders. The "1990s bar" every SHOR-branded
    component must beat (REVIEW_EXTERNAL_AUDIT.md M3).

Also: `shape_features` — unit- and fs-invariant waveform descriptors
used by the sim-vs-real discriminator gate (Stage 4).
"""
import numpy as np
from scipy.signal import butter, sosfilt, hilbert, resample_poly
from fractions import Fraction

COMMON_FS = 5120.0          # SEU-limited common analysis rate
WIN_S = 4.0


# ---------------------------------------------------------------- B1 --
def fast_kurtogram_band(x, fs, nlevels=4):
    """Simplified fast kurtogram: binary/ternary tree of band-passes,
    return (f_lo, f_hi) of the max-spectral-kurtosis band."""
    best = (-np.inf, (fs * 0.25, fs * 0.5))
    nyq = fs / 2
    for lev in range(1, nlevels + 1):
        nb = 2 ** lev
        width = nyq / nb
        for b in range(nb):
            lo, hi = b * width, (b + 1) * width
            if lo < 200:                       # skip DC-adjacent band
                lo = 200
                if hi <= lo + 50:
                    continue
            sos = butter(4, [max(lo, 1) / nyq, min(hi / nyq, 0.999)],
                         btype="band", output="sos")
            xb = sosfilt(sos, x)
            env = np.abs(hilbert(xb))
            e = env - env.mean()
            m2 = np.mean(e ** 2)
            if m2 <= 0:
                continue
            k = np.mean(e ** 4) / m2 ** 2 - 3
            if k > best[0]:
                best = (k, (lo, hi))
    return best[1], best[0]


def envelope_spectrum(x, fs, band):
    sos = butter(4, [band[0] / (fs / 2), min(band[1] / (fs / 2), 0.999)],
                 btype="band", output="sos")
    env = np.abs(hilbert(sosfilt(sos, x)))
    env = env - env.mean()
    p = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    fr = np.fft.rfftfreq(len(env), 1 / fs)
    return fr, p


def peak_sig(fr, p, f_target, tol=0.04, floor_span=3.0):
    """Peak significance at f_target: max power within +-tol (relative)
    over the local median floor. Returns (snr, f_peak)."""
    if f_target <= 0 or f_target >= fr[-1]:
        return 0.0, np.nan
    band = np.abs(fr / f_target - 1) < tol
    if not band.any():
        return 0.0, np.nan
    lo = f_target / floor_span
    hi = min(f_target * floor_span, fr[-1])
    floor_band = (fr > lo) & (fr < hi) & ~band
    floor = np.median(p[floor_band]) if floor_band.any() else np.nan
    if not np.isfinite(floor) or floor <= 0:
        return 0.0, np.nan
    j = np.argmax(p[band])
    return float(p[band][j] / floor), float(fr[band][j])


def b1_envelope_verdict(x, fs, f_shaft_nominal, orders,
                        snr_thresh=10.0, tol=0.04):
    """B1 verdict for one record. orders: {name: order_per_rev}.
    f_shaft_nominal: NAMEPLATE speed (Hz) — deployment-legal knowledge.
    Returns dict with per-order envelope SNR, best call, and the
    selected kurtogram band. Detection rule: any defect order's
    envelope SNR >= snr_thresh (and its harmonic at 2x >= sqrt rule)."""
    band, kurt = fast_kurtogram_band(x, fs)
    fr, p = envelope_spectrum(x, fs, band)
    out = {"band_lo": band[0], "band_hi": band[1], "band_kurt": kurt}
    calls = {}
    for name, o in orders.items():
        f_t = o * f_shaft_nominal
        snr1, fp1 = peak_sig(fr, p, f_t, tol)
        snr2, _ = peak_sig(fr, p, 2 * f_t, tol)
        out[f"snr_{name}"] = round(snr1, 2)
        out[f"snr2_{name}"] = round(snr2, 2)
        calls[name] = snr1 >= snr_thresh and (snr2 >= 3.0 or
                                              snr1 >= 3 * snr_thresh)
    hit = [k for k, v in calls.items() if v]
    out["detected"] = bool(hit)
    out["called"] = max(hit, key=lambda k: out[f"snr_{k}"]) if hit else ""
    return out


# ------------------------------------------- discriminator features --
def _common(x, fs):
    """Resample to COMMON_FS, cut WIN_S windows, unit-RMS normalize."""
    frac = Fraction(int(COMMON_FS), int(round(fs))).limit_denominator(2000)
    y = resample_poly(np.asarray(x, float), frac.numerator,
                      frac.denominator)
    n = int(COMMON_FS * WIN_S)
    wins = []
    for k in range(len(y) // n):
        w = y[k * n:(k + 1) * n]
        w = w - w.mean()
        r = np.sqrt(np.mean(w ** 2))
        if r > 0:
            wins.append(w / r)
    return wins


def shape_features(x, fs):
    """Unit-free, fs-free waveform-shape descriptors, one dict per
    4-s window at the common rate."""
    from scipy.stats import kurtosis, skew
    rows = []
    for w in _common(x, fs):
        n = len(w)
        f = np.fft.rfftfreq(n, 1 / COMMON_FS)
        p = np.abs(np.fft.rfft(w * np.hanning(n))) ** 2
        p = p / p.sum()
        d = {}
        # 24 log-spaced band energies 5..2500 Hz
        edges = np.geomspace(5, 2500, 25)
        for b in range(24):
            m = (f >= edges[b]) & (f < edges[b + 1])
            d[f"band_{b:02d}"] = float(np.log10(p[m].sum() + 1e-12))
        ps = p[f > 5]
        d["spec_entropy"] = float(-(ps * np.log(ps + 1e-15)).sum()
                                  / np.log(len(ps)))
        d["spec_flatness"] = float(np.exp(np.mean(np.log(ps + 1e-15)))
                                   / (ps.mean() + 1e-15))
        med = np.median(ps)
        d["peak_density"] = float(np.mean(ps > 10 * med))
        d["psd_crest"] = float(np.log10(ps.max() / (med + 1e-15) + 1e-9))
        d["kurtosis"] = float(kurtosis(w))
        d["skew"] = float(skew(w))
        d["crest"] = float(np.max(np.abs(w)))
        d["kurt_diff"] = float(kurtosis(np.diff(w)))
        e = np.sort(w ** 2)[::-1]
        d["top1pct_energy"] = float(e[:max(n // 100, 1)].sum() / e.sum())
        # HF envelope behaviour (800-2400 Hz at common rate)
        sos = butter(4, [800 / (COMMON_FS / 2), 2400 / (COMMON_FS / 2)],
                     btype="band", output="sos")
        env = np.abs(hilbert(sosfilt(sos, w)))
        env = env - env.mean()
        pe = np.abs(np.fft.rfft(env * np.hanning(n))) ** 2
        pes = pe[1:] / (pe[1:].sum() + 1e-15)
        d["env_flatness"] = float(np.exp(np.mean(np.log(pes + 1e-15)))
                                  / (pes.mean() + 1e-15))
        d["env_peak_snr"] = float(np.log10(
            pes.max() / (np.median(pes) + 1e-15) + 1e-9))
        d["env_kurt"] = float(kurtosis(env))
        rows.append(d)
    return rows
