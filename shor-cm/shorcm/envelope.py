"""HF-resonance envelope channel.

An impulsive bearing fault excites a high-frequency structural
resonance; the defect rate lives in the ENVELOPE of that band, not as
a narrowband tone at the defect order (the MFPT transfer gate measured
exactly this: raw-spectrum attribution 0.0 at BPFO/BPFI while the
envelope shows the structure immediately).

Band selection: small fixed candidate set, pick the band whose
envelope has maximum kurtosis (poor-man's spectral kurtosis — the
impulse train is what makes an envelope leptokurtic).
"""
import numpy as np
from scipy.signal import hilbert, butter, filtfilt, decimate


def _bands(fs):
    cands = [(1500.0, 4000.0), (2500.0, 6500.0), (5000.0, 11000.0),
             (0.25 * fs, 0.45 * fs)]
    out = []
    for lo, hi in cands:
        hi = min(hi, 0.47 * fs)
        if hi - lo > 800.0 and lo > 600.0:
            out.append((lo, hi))
    return out or [(0.25 * fs, 0.45 * fs)]


def envelope_signal(x, fs, fsd_target=6000.0):
    """(env, fs_env, q, band): mean-removed Hilbert envelope of the
    max-kurtosis HF band, decimated by integer q so downstream order
    analysis is tractable. q == 1 when fs is already low."""
    x = np.asarray(x, float)
    best = None
    for lo, hi in _bands(fs):
        b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
        xb = filtfilt(b, a, x)
        env = np.abs(hilbert(xb))
        e = env - env.mean()
        k = float(np.mean(e ** 4) / (np.mean(e ** 2) ** 2 + 1e-30))
        if best is None or k > best[0]:
            best = (k, e, (lo, hi))
    _, env, band = best
    q = max(int(fs // fsd_target), 1)
    if q > 1:
        env = decimate(env, q, ftype="fir")
    return env, fs / q, q, band
