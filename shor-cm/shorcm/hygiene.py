"""Input hygiene + unit contract for field data (audit F1/F2).

Field records arrive with unknown units (g, m/s^2, mm/s, arbitrary
DAQ gains), DC offsets, NaN gaps, dropouts, and clipping. Two rules:

 1. VALIDATE and FLAG before anything downstream sees the signal —
    dropouts are the most leptokurtic thing in a record, so an
    unflagged dropout would capture the envelope band selector and
    fabricate bearing evidence.
 2. NORMALIZE to unit RMS at the API entry: every downstream feature
    becomes scale-invariant (a gain change or unit change cannot move
    a single model input). The absolute level is preserved separately
    in the output (signal_rms) for level-based policies that know
    their units.
"""
import numpy as np


class SignalHygieneError(ValueError):
    pass


def validate_signal(x, fs, max_gap_frac=0.005):
    """(x_clean, flags). Raises SignalHygieneError when the record is
    unusable (too many NaNs, dead, or too short). Repairs short NaN
    gaps by linear interpolation; removes DC; flags dropouts and
    clipping for the caller to surface."""
    x = np.asarray(x, float).ravel()
    flags = {}
    if len(x) < 4096:
        raise SignalHygieneError(f"record too short: {len(x)} samples")
    bad = ~np.isfinite(x)
    if bad.any():
        frac = float(bad.mean())
        flags["nan_frac"] = round(frac, 5)
        if frac > max_gap_frac:
            raise SignalHygieneError(
                f"{frac:.2%} non-finite samples (limit "
                f"{max_gap_frac:.2%})")
        idx = np.arange(len(x))
        x = x.copy()
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms < 1e-12:
        raise SignalHygieneError("dead signal (zero RMS)")
    # dropout: runs of (near-)constant samples
    dx = np.diff(x)
    const = np.abs(dx) < 1e-10 * max(np.abs(x).max(), 1e-12)
    if const.any():
        runs = np.diff(np.flatnonzero(np.diff(np.concatenate(
            ([0], const.astype(int), [0])))))[::2]
        longest = int(runs.max()) if len(runs) else 0
        if longest > 0.002 * len(x):
            flags["dropout_run"] = longest
    # clipping: mass at the extremes
    peak = np.abs(x).max()
    clip_frac = float((np.abs(x) > 0.999 * peak).mean())
    if clip_frac > 0.005:
        flags["clip_frac"] = round(clip_frac, 4)
    return x, flags
