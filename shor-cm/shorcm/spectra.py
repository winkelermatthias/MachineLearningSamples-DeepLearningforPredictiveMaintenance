"""Angular resampling, order spectra, coherent/incoherent split,
continued-fraction snapping, synthetic tone injection."""
import numpy as np
from fractions import Fraction
from scipy.interpolate import CubicSpline


def angular_resample(x, phase, spr=1024):
    """Resample x (time domain) to fixed samples-per-rev using phase[rad].
    Returns xa with len = whole number of (1/spr) rev steps."""
    x = np.asarray(x, float)
    valid = np.isfinite(phase)
    i0 = int(np.argmax(valid))
    i1 = int(len(valid) - np.argmax(valid[::-1]))
    ph = phase[i0:i1].copy()
    xs = x[i0:i1]
    # enforce strictly increasing phase (guards Hilbert glitches)
    ph = np.maximum.accumulate(ph)
    step = 2 * np.pi / spr
    k0 = int(np.ceil(ph[0] / step))
    k1 = int(np.floor(ph[-1] / step))
    tgt = np.arange(k0, k1 + 1) * step
    idx = np.interp(tgt, ph, np.arange(len(xs), dtype=float))
    return CubicSpline(np.arange(len(xs)), xs)(idx)


def block_spectra(xa, spr=1024, revs=5, window="rect"):
    """Complex order spectra of consecutive revs-long blocks.
    Order grid = k/revs, k=0..spr*revs/2. Rectangular window is exact for
    components on the k/revs grid (integer cycles per block)."""
    L = spr * revs
    nb = len(xa) // L
    if nb < 2:
        raise ValueError("need >=2 blocks, got %d" % nb)
    B = xa[: nb * L].reshape(nb, L).astype(float)
    B = B - B.mean(axis=1, keepdims=True)
    if window == "hann":
        w = np.hanning(L)
        B = B * w
        norm = w.sum() / 2
    else:
        norm = L / 2
    Z = np.fft.rfft(B, axis=1) / norm
    orders = np.arange(Z.shape[1]) / revs
    return Z, orders


def coherent_split(Z, eps=1e-12):
    """coh: |complex mean| (phase-locked survives, ~N gain vs sqrt(N) noise)
    inc: mean |.| (power-average, everything survives)
    ratio in [0,1]: ~1 locked to phase reference, ~0 unlocked."""
    coh = np.abs(Z.mean(axis=0))
    inc = np.abs(Z).mean(axis=0)
    return coh, inc, coh / (inc + eps)


def fine_order_spectrum(xa, spr=1024):
    """Single FFT over all whole revs: order resolution = 1/total_revs.
    Used for precise peak order estimation feeding cf_snap."""
    revs = len(xa) // spr
    L = revs * spr
    y = xa[:L] - xa[:L].mean()
    w = np.hanning(L)
    A = np.abs(np.fft.rfft(y * w)) / (w.sum() / 2)
    orders = np.arange(len(A)) / revs
    return A, orders


def peak_orders(A, orders, max_order=12.0, n_peaks=20, guard=3.0):
    """Local maxima above guard * local median, parabolic interp for
    sub-bin order estimate. Returns list of (order, amplitude, prominence)."""
    m = orders <= max_order
    A = A[m]
    o = orders[m]
    med = np.median(A) + 1e-15
    out = []
    for i in range(2, len(A) - 2):
        if A[i] > A[i - 1] and A[i] >= A[i + 1] and A[i] > guard * med:
            d = 0.5 * (A[i - 1] - A[i + 1]) / (A[i - 1] - 2 * A[i] + A[i + 1] + 1e-15)
            d = float(np.clip(d, -0.5, 0.5))
            out.append((float(o[i] + d * (o[1] - o[0])), float(A[i]), float(A[i] / med)))
    out.sort(key=lambda t: -t[1])
    return out[:n_peaks]


def cf_snap(order, tol=0.01, qmax=8):
    """Shor post-processing: snap noisy order to rational p/q, q<=qmax,
    else None. 0.4031 -> 2/5, 0.503 -> 1/2, 3.585 -> None (tol 0.01)."""
    fr = Fraction(order).limit_denominator(qmax)
    return fr if abs(float(fr) - order) <= tol else None


def snap_tol(total_revs, safety=2.0):
    """Sensible tol from record length: safety * order resolution."""
    return safety / total_revs


# ---------------- synthetic injection (H3) ----------------

def inject_locked(x, phase, order, amp, phi0=0.0):
    """Add tone phase-locked to the shaft (uses the SAME phase reference)."""
    return x + amp * np.cos(order * np.nan_to_num(phase) + phi0)


def inject_unlocked(x, fs, f_hz, amp, phi0=0.0):
    """Add fixed-Hz tone (models bearing-like slip: not shaft-locked)."""
    t = np.arange(len(x)) / fs
    return x + amp * np.cos(2 * np.pi * f_hz * t + phi0)


def amp_for_snr(x, snr_db):
    """Tone amplitude for target SNR vs total signal RMS (documented,
    crude but consistent definition used across H3)."""
    return float(np.sqrt(2) * np.std(x) * 10 ** (snr_db / 20))
