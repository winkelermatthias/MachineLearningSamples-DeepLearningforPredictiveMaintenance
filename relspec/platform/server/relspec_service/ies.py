"""Fast-SC second opinion: the coherence-based Improved Envelope Spectrum,
computed on demand from a STORED raw waveform. Strictly read-only and
advisory — it touches no codec payloads, rails, z/health state, or Gate
decisions. The 8 dB advisory line comes from the promotion benchmark: the
worst healthy record (CWRU normal 97 at BPFI) reached 5.74 dB, and 8 dB
still kept 13 extra low-severity detects with zero healthy false lines.
"""
from __future__ import annotations
import numpy as np
import zstandard
from . import config  # noqa: F401  (sets sys.path for relspec import)
from relspec.fastsc import fast_sc, ies_full

MIN_FS = 4000.0        # below this the bin-shift alpha range collapses
MIN_SECONDS = 2.0      # fewer frames than this and the alpha grid is mush
MAX_SECONDS = 8.0      # cap the frame count; the STFT matrices grow as n
ALPHA_MIN = 10.0       # near-DC alpha sliver is never a bearing line
F_LO = 400.0           # spectral-axis integration band (benchmark values)
F_HI_FRAC = 0.94
GUARD_HZ, NOISE_HZ = 12.0, 90.0   # Hz-fixed SNR windows, as benchmarked
ADVISORY_DB = 8.0
MAX_POINTS = 256
N_PEAKS = 8

class IesError(Exception):
    def __init__(self, status: int, detail: str):
        self.status, self.detail = status, detail

def decode_stored(encoding: str, scale: float, data) -> np.ndarray:
    if encoding != 'int16-zstd':
        raise IesError(422, f'unsupported stored encoding {encoding!r}')
    raw = zstandard.ZstdDecompressor().decompress(bytes(data))
    return np.frombuffer(raw, dtype='<i2').astype(np.float64) * float(scale)

def _peak_snrs(alpha: np.ndarray, e: np.ndarray) -> list[dict]:
    """Local maxima of the IES above ALPHA_MIN, each scored against its own
    median ring (same guard/noise geometry as the promotion benchmark)."""
    da = float(alpha[1] - alpha[0])
    guard = max(2, int(round(GUARD_HZ / da)))
    ring_w = max(8, int(round(NOISE_HZ / da)))
    is_max = np.r_[False, (e[1:-1] > e[:-2]) & (e[1:-1] >= e[2:]), False]
    cand = np.flatnonzero(is_max & (alpha >= ALPHA_MIN))
    peaks = []
    for j in sorted(cand, key=lambda j: -e[j]):
        if any(abs(alpha[j] - p['alpha_hz']) < GUARD_HZ for p in peaks):
            continue                       # one peak per neighbourhood
        lo, hi = max(0, j - ring_w), min(len(e), j + ring_w + 1)
        ring = np.r_[e[lo:max(lo, j - guard)], e[min(hi, j + guard + 1):hi]]
        if ring.size == 0:
            continue
        snr = 20 * np.log10(max(e[j], 1e-12) / max(np.median(ring), 1e-12))
        peaks.append(dict(alpha_hz=round(float(alpha[j]), 3),
                          ies=round(float(e[j]), 5),
                          snr_db=round(float(snr), 2)))
        if len(peaks) >= N_PEAKS:
            break
    return peaks

def compute(x: np.ndarray, fs: float, fr: float | None) -> dict:
    """IES for one stored waveform. Returns a JSON-ready dict: a downsampled
    spectrum (<= MAX_POINTS, block-max so lines survive decimation) plus the
    top peaks with SNR against the local median floor."""
    if fs < MIN_FS:
        raise IesError(422, f'fs {fs:g} Hz below the {MIN_FS:g} Hz Fast-SC '
                            'floor (bin-shift alpha range collapses)')
    if len(x) < MIN_SECONDS * fs:
        raise IesError(422, f'record {len(x)/fs:.2f} s too short for Fast-SC '
                            f'(needs >= {MIN_SECONDS:g} s)')
    n_used = min(len(x), int(MAX_SECONDS * fs))
    alpha_max = min(300.0, 0.45 * fs / 2)
    try:
        alpha, f, g = fast_sc(x[:n_used], fs, alpha_max=alpha_max)
    except ValueError as e:
        raise IesError(422, str(e))
    e = ies_full(f, g, F_LO, F_HI_FRAC * fs / 2)
    peaks = _peak_snrs(alpha, e)
    if fr:
        for p in peaks:
            p['order'] = round(p['alpha_hz'] / fr, 4)
    # decimate for transport: block max keeps narrow lines visible
    stride = max(1, int(np.ceil(alpha.size / MAX_POINTS)))
    nb = alpha.size // stride
    e_ds = e[:nb * stride].reshape(nb, stride).max(axis=1)
    a_ds = alpha[:nb * stride].reshape(nb, stride).mean(axis=1)
    return dict(fs=float(fs), n=int(len(x)), analyzed_s=round(n_used / fs, 3),
                fr=(float(fr) if fr else None),
                da_hz=round(float(alpha[1] - alpha[0]), 4),
                alpha_max_hz=float(alpha_max),
                band_hz=[F_LO, round(F_HI_FRAC * fs / 2, 1)],
                advisory_db=ADVISORY_DB,
                advisory=any(p['snr_db'] > ADVISORY_DB for p in peaks),
                peaks=peaks,
                alpha_hz=[round(float(v), 3) for v in a_ds],
                ies=[round(float(v), 5) for v in e_ds])
