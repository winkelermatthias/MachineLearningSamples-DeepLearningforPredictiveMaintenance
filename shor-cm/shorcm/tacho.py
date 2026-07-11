"""Phase reference extraction. Four variants:

V-TACHO   : real tachometer channel (ground truth phase)
V-1X      : bandpass around nominal speed + Hilbert (signal-derived phase)
V-COMB    : harmonic-comb f0 tracking on spectrogram + integrate (signal-derived)
V-NOMINAL : constant speed from filename, phase = 2*pi*f_nom*t (no tracking)

All return (phase[rad, len n], meta dict). Phase may contain NaN at edges
(tacho spline outside first/last pulse); angular_resample handles that.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt, hilbert, stft
from scipy.ndimage import median_filter

VARIANTS = ("tacho", "onex", "comb", "nominal")


def tacho_pulse_times(x, fs):
    """Rising-edge times (float sample index) of tacho pulses, one per rev."""
    x = np.asarray(x, float)
    x = x - np.median(x)
    thr = 0.5 * np.percentile(np.abs(x), 99.5)
    above = x > thr
    idx = np.flatnonzero(~above[:-1] & above[1:]) + 1
    if len(idx) < 4:
        raise ValueError("too few tacho pulses (%d)" % len(idx))
    frac = (thr - x[idx - 1]) / (x[idx] - x[idx - 1])
    t = idx - 1 + frac
    # debounce: drop pulses closer than 20% of median period
    med = np.median(np.diff(t))
    keep = [0]
    for i in range(1, len(t)):
        if t[i] - t[keep[-1]] > 0.2 * med:
            keep.append(i)
    return t[keep]


def phase_from_tacho(x_tacho, fs, n=None):
    n = n if n is not None else len(x_tacho)
    t = tacho_pulse_times(x_tacho, fs)
    phi = 2 * np.pi * np.arange(len(t), dtype=float)
    cs = CubicSpline(t, phi, extrapolate=False)
    phase = cs(np.arange(n, dtype=float))
    per = np.diff(t)
    meta = {
        "variant": "tacho",
        "rate_hz": float(fs / np.mean(per)),
        "quality": float(np.std(per) / np.mean(per)),  # rev-to-rev jitter
        "n_pulses": int(len(t)),
    }
    return phase, meta


def phase_from_1x(x_acc, fs, f_nom, rel_bw=0.25):
    """Signal-derived phase from the 1x component itself.

    NOTE self-referential bias: order-1.0 coherence ratio is ~1 by
    construction under this variant. Exclude order 1.0 ratio from features.
    """
    lo = max(f_nom * (1 - rel_bw), 0.5)
    hi = min(f_nom * (1 + rel_bw), 0.45 * fs)
    sos = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
    xb = sosfiltfilt(sos, np.asarray(x_acc, float))
    phase = np.unwrap(np.angle(hilbert(xb)))
    rate = (phase[-1] - phase[0]) / (2 * np.pi * (len(xb) - 1) / fs)
    # strength of the 1x reference: analytic amplitude vs residual
    amp = np.abs(hilbert(xb))
    meta = {"variant": "onex", "rate_hz": float(rate),
            "ref_snr": float(np.mean(amp) / (np.std(x_acc) + 1e-12))}
    return phase, meta


def phase_from_comb(x_acc, fs, f_nom, n_harm=5, rel_span=0.06,
                    frame_s=0.4, hop_s=0.1, n_cand=121, prior_rel_sigma=0.01):
    """Track f0(t) by scoring a harmonic comb near f_nom on the STFT,
    median-smooth, refine phase on the best harmonic line.

    CAPTURE GUARD (found via smoke test): a dominant NON-synchronous tone
    (e.g. BPFO with slip) can hijack the comb if scored by raw magnitude
    sum, making a bearing tone look perfectly coherent. Two defenses:
    (1) log-compressed scoring, so a harmonic FAMILY of moderate lines
        beats one huge line;
    (2) Gaussian prior around the provided speed reading f_nom
        (default sigma 1%: bearing slip 1-2% is penalized, true speed
        error of a decent estimate is not).
    """
    nper = int(frame_s * fs)
    f, tt, Z = stft(np.asarray(x_acc, float), fs, nperseg=nper,
                    noverlap=nper - int(hop_s * fs), padded=False)
    S = np.abs(Z)
    med = np.median(S, axis=0, keepdims=True) + 1e-15
    cand = f_nom * np.linspace(1 - rel_span, 1 + rel_span, n_cand)
    df = f[1] - f[0]
    score = np.zeros((n_cand, S.shape[1]))
    for h in range(1, n_harm + 1):
        bins = np.clip(np.round(cand * h / df).astype(int), 0, len(f) - 1)
        score += np.log1p(S[bins, :] / med)
    prior = np.exp(-0.5 * ((cand - f_nom) / (prior_rel_sigma * f_nom)) ** 2)
    score *= prior[:, None]
    f0 = cand[np.argmax(score, axis=0)]
    f0 = median_filter(f0, size=5, mode="nearest")
    f0_mean = float(np.mean(f0))
    spread = float(np.std(f0) / f0_mean)
    # refinement: integrating coarse f0 accumulates phase drift, so instead
    # lock onto the STRONGEST harmonic line (works when 1x is weak) via
    # Hilbert phase and divide by harmonic number. Constant 2*pi/h phase
    # offset is irrelevant for block coherence.
    # refinement harmonic: LOWEST h whose line is prominent (>5x local
    # median). Picking the globally strongest line risks re-capture by a
    # nearby non-synchronous tone.
    h = 1
    for hc in range(1, n_harm + 1):
        b = int(np.clip(round(f0_mean * hc / df), 0, len(f) - 1))
        line = S[max(b - 1, 0):b + 2, :].max(axis=0)
        if float(np.median(line / med[0])) > 5.0:
            h = hc
            break
    bw = 0.06 + 3 * spread
    lo = max(h * f0_mean * (1 - bw), 0.5)
    hi = min(h * f0_mean * (1 + bw), 0.45 * fs)
    sos = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
    xb = sosfiltfilt(sos, np.asarray(x_acc, float))
    phase = np.unwrap(np.angle(hilbert(xb))) / h
    meta = {"variant": "comb", "rate_hz": float(f0_mean),
            "f0_spread": spread, "ref_harmonic": h}
    return phase, meta


def phase_nominal(n, fs, f_nom):
    """Constant-speed assumption, e.g. from MAFAULDA filename (Hz)."""
    phase = 2 * np.pi * f_nom * np.arange(n) / fs
    return phase, {"variant": "nominal", "rate_hz": float(f_nom)}


def get_phase(variant, run, fs, f_nom, ref_channel=None):
    """Dispatch. run: (n_samples, 8) array. ref_channel: accel column for
    signal-derived variants (pick strongest 1x upstream)."""
    n = run.shape[0]
    if variant == "tacho":
        return phase_from_tacho(run[:, 0], fs, n)
    if variant == "onex":
        return phase_from_1x(run[:, ref_channel], fs, f_nom)
    if variant == "comb":
        return phase_from_comb(run[:, ref_channel], fs, f_nom)
    if variant == "nominal":
        return phase_nominal(n, fs, f_nom)
    raise ValueError(variant)
