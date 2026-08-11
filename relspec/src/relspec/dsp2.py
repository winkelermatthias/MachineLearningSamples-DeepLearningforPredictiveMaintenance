"""Deeper signal processing, in service of the codec.

Every technique here earns its place through one of two currencies:

  DETECTION: it makes a fault line stand higher above the floor
             (kurtogram band selection, cepstral prewhitening), or
  BYTES:     it makes consecutive frames more alike so the residual coder
             pays less (Viterbi speed tracking, order-domain resampling).

The second currency is the point of this module. A spectrum codec's residual
is mostly ALIGNMENT error: a 0.5% speed error moves a 10th-harmonic line five
fine bins, and the coder pays for the departure and the arrival. Sharper speed
estimates and order-locked sampling attack the residual at its source, which
no amount of entropy-coder tuning can do.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

# ===================================================== 1. spectral kurtosis
def spectral_kurtosis(x, fs, nperseg=256):
    """SK(f) = <|X|^4>/<|X|^2>^2 - 2 over STFT frames. Impulsive content is
    heavy-tailed in every band it excites, so SK is large exactly where the
    fault's impacts ring and near zero where the signal is stationary."""
    n = len(x)
    hop = nperseg//2
    win = np.hanning(nperseg)
    nfr = max(1, (n-nperseg)//hop+1)
    S2 = np.zeros(nperseg//2+1); S4 = np.zeros(nperseg//2+1)
    for i in range(nfr):
        seg = x[i*hop:i*hop+nperseg]
        if len(seg) < nperseg: break
        X = np.abs(np.fft.rfft(seg*win))**2
        S2 += X; S4 += X**2
    S2 /= nfr; S4 /= nfr
    with np.errstate(divide='ignore', invalid='ignore'):
        sk = np.where(S2 > 1e-20, S4/np.maximum(S2**2, 1e-40)-2.0, 0.0)
    f = np.fft.rfftfreq(nperseg, 1/fs)
    return f, sk

def kurtogram_band(x, fs, levels=(1, 2, 3, 4), fmin_frac=0.04,
                   guard_frac=0.94, tie_frac=0.85):
    """Pick the demodulation band as the dyadic band maximising the kurtosis
    of its own envelope - the quantity envelope analysis actually consumes,
    rather than SK averaged over the band, which favours narrow noise spikes.

    v1 hardcoded 2-5 kHz. That is right for CWRU drive-end resonances and
    wrong for anything else: MFPT's outer-race ringing sits below 2 kHz at
    half the sample rate, and the SEU gearbox puts its energy near mesh
    harmonics. The band must come from the data.

    Two constraints matter as much as the score. Bands narrower than 1/16 of
    Nyquist are excluded outright: a modulated fault train needs bandwidth of
    several times its highest sideband spacing, and a too-narrow band shows
    beautiful kurtosis while amputating the sidebands the envelope spectrum
    is FOR. And among near-ties the WIDEST band wins, because extra bandwidth
    costs nothing when the impulsive content genuinely fills it."""
    n = len(x)
    X = np.fft.rfft(x)
    nyq = fs/2
    cands = []
    for L in levels:
        nb = 1 << L
        for b in range(nb):
            lo, hi = b/nb*nyq, (b+1)/nb*nyq
            if lo < fmin_frac*nyq: continue            # DC / shaft region
            if hi > guard_frac*nyq: hi = guard_frac*nyq
            if hi-lo < nyq/16: continue                # sideband-carrying minimum
            i0, i1 = int(lo/nyq*(len(X)-1)), int(hi/nyq*(len(X)-1))
            Y = np.zeros_like(X); Y[i0:i1] = X[i0:i1]
            env = np.abs(np.fft.irfft(Y, n))
            d = env-env.mean(); sd = d.std()
            if sd < 1e-15: continue
            kurt = float((d**4).mean()/sd**4)
            cands.append(((lo, hi), kurt))
    if not cands: return (0.3*nyq, 0.8*nyq), 3.0
    kmax = max(k for _, k in cands)
    near = [(b, k) for b, k in cands if k >= tie_frac*kmax]
    return max(near, key=lambda c: c[0][1]-c[0][0])

def envelope_banded(x, fs, band):
    """Envelope on an arbitrary band via FFT masking (zero-phase, no filter
    design constraints near Nyquist, works at any fs)."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1/fs)
    Y = np.where((f >= band[0]) & (f <= band[1]), X, 0.0)
    e = np.abs(hilbert(np.fft.irfft(Y, n)))
    return e-e.mean()

# ================================================ 2. cepstral prewhitening
def prewhiten(x):
    """Flatten the magnitude spectrum, keep the phase. Resonances and mesh
    harmonics carry their energy in |X|; impact TIMING lives in the phase.
    Setting |X|=1 removes everything periodic and leaves impulses standing on
    a flat floor - the cheapest fault enhancer there is, one FFT each way."""
    X = np.fft.rfft(x)
    return np.fft.irfft(X/np.maximum(np.abs(X), 1e-12), len(x))

# ============================================== 3. Viterbi speed tracking
def hps_grid(x, fs, lo=5.0, hi=90.0, nh=3, step=0.01, band=None):
    """Envelope-domain harmonic product score over a speed grid; the emission
    row for one acquisition. Same physics as pipeline.estimate_speed, kept
    separate so the tracker can consume the whole surface, not the argmax."""
    e = envelope_banded(x, fs, band) if band else _default_env(x, fs)
    N = 1 << int(np.ceil(np.log2(len(e))))
    A = np.abs(np.fft.rfft(e*np.hanning(len(e)), N))*2/len(e)
    f = np.fft.rfftfreq(N, 1/fs)
    grid = np.arange(lo, hi, step)
    hps = np.ones_like(grid)
    for h in range(1, nh+1):
        hps *= np.maximum(np.interp(grid*h, f, A), 1e-12)
    return grid, hps**(1/nh)

def acc_comb_surface(x, fs, fgrid, nh=6):
    """pipeline2.acc_comb_speed's score over an explicit grid: geometric mean
    of the full-record FFT magnitude over nh shaft harmonics. Kept as a
    surface (not an argmax) so a posterior can consume the whole row."""
    a = np.abs(np.fft.rfft(x*np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1/fs)
    sc = np.ones_like(fgrid)
    for h in range(1, nh+1):
        sc *= np.maximum(np.interp(fgrid*h, f, a), 1e-12)
    return sc**(1/nh)

def _default_env(x, fs):
    sos = butter(6, [min(2000., 0.35*fs/2)/(fs/2), min(5000., 0.8*fs/2)/(fs/2)],
                 btype='band', output='sos')
    e = np.abs(hilbert(sosfiltfilt(sos, x)))
    return e-e.mean()

def viterbi_track(emissions, grid, max_jump_pct=2.0, lam=8.0):
    """Maximum-a-posteriori speed path through a sequence of HPS surfaces.

    The per-frame argmax fails in a characteristic way: a frame where the true
    fundamental is weak hands the answer to a rival at 2x or a mount line, one
    frame reports garbage, and the codec then order-bins that frame against
    the wrong axis - a misalignment that costs hundreds of residual bytes.
    A machine's speed cannot teleport. Encoding that single physical fact as
    a quadratic transition penalty removes the teleports; the emission surface
    still decides everything within the plausible tube."""
    T = len(emissions); G = len(grid)
    if T == 0: return np.array([])
    loggrid = np.log(grid)
    dg = loggrid[1]-loggrid[0] if G > 1 else 1e-3
    W = max(1, int(np.log(1+max_jump_pct/100.0)/max(dg, 1e-9)))
    logem = [np.log(np.maximum(e, 1e-30)) for e in emissions]
    cost = logem[0].copy()
    back = np.zeros((T, G), dtype=np.int32)
    for t in range(1, T):
        best = np.full(G, -np.inf); arg = np.zeros(G, dtype=np.int32)
        for off in range(-W, W+1):
            pen = lam*(off*dg/np.log(1.02))**2
            src = np.arange(max(0, -off), min(G, G-off))
            dst = src+off
            c = cost[src]-pen
            upd = c > best[dst]
            best[dst[upd]] = c[upd]; arg[dst[upd]] = src[upd]
        cost = best+logem[t]; back[t] = arg
    path = np.zeros(T, dtype=np.int32)
    path[-1] = int(np.argmax(cost))
    for t in range(T-1, 0, -1): path[t-1] = back[t, path[t]]
    return grid[path]

# ============================================ 4. order-domain resampling
def instantaneous_speed(x, fs, fr_nominal, halfwidth_pct=12.0):
    """Phase of the analytic 1x tone. Bandpass around the nominal, unwrap,
    differentiate, smooth. Valid while the true 1x stays inside the band,
    which the Viterbi tracker guarantees at the acquisition level."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1/fs)
    lo, hi = fr_nominal*(1-halfwidth_pct/100), fr_nominal*(1+halfwidth_pct/100)
    Y = np.where((f >= lo) & (f <= hi), X, 0.0)
    z = hilbert(np.fft.irfft(Y, n))
    ph = np.unwrap(np.angle(z))
    fr_t = np.gradient(ph)*fs/(2*np.pi)
    k = max(8, int(fs/fr_nominal/2))
    ker = np.ones(k)/k
    fr_s = np.convolve(fr_t, ker, mode='same')
    fr_s[:k] = fr_s[k]; fr_s[-k:] = fr_s[-k-1]
    return np.clip(fr_s, 0.5*fr_nominal, 2.0*fr_nominal), ph

def order_spectrum(x, fs, fr_nominal, n_orders_max=200.0, pts_per_rev=None):
    """Angular resampling: sample the waveform at uniform SHAFT ANGLE, then
    one FFT gives a spectrum whose axis is orders exactly, independent of
    wander. Under steady speed this equals the Welch path; under wander it
    concentrates each line back into one bin - and a line that stays in its
    bin is a line the residual coder gets almost for free."""
    fr_t, _ = instantaneous_speed(x, fs, fr_nominal)
    theta = np.cumsum(fr_t)/fs                       # revolutions
    revs = theta[-1]-theta[0]
    if pts_per_rev is None:
        pts_per_rev = int(2*n_orders_max*1.28)
    n_out = int(revs*pts_per_rev)
    if n_out < 64: return None, None
    th_u = np.linspace(theta[0], theta[-1], n_out, endpoint=False)
    t_axis = np.arange(len(x))/fs
    xu = np.interp(th_u, theta, t_axis)              # time at uniform angle
    xr = np.interp(xu, t_axis, x)
    win = np.hanning(n_out)
    A = np.abs(np.fft.rfft(xr*win))*2/np.sum(win)
    orders = np.fft.rfftfreq(n_out, 1.0/pts_per_rev)
    m = orders <= n_orders_max
    return orders[m], A[m]

def order_welch(x, fs, fr_nominal, n_orders_max=200.0, revs_per_seg=16,
                pts_per_rev=None):
    """Welch in the ANGLE domain: angular resampling, then averaged
    periodograms over segments of whole revolutions.

    The single-FFT order spectrum wins line sharpness but loses at the codec:
    with no averaging every noise bin scatters chi-squared frame to frame and
    the residual coder pays to encode that scatter. Averaging in the angle
    domain keeps both properties - lines stay locked to integer orders while
    the floor smooths with sqrt(K) - and the codec sees the best of the two
    worlds. Sampling frequency handed to Welch is points-per-rev, so the
    returned axis is already in orders."""
    from scipy.signal import welch as _welch
    fr_t, _ = instantaneous_speed(x, fs, fr_nominal)
    theta = np.cumsum(fr_t)/fs
    revs = theta[-1]-theta[0]
    if pts_per_rev is None:
        pts_per_rev = int(2*n_orders_max*1.28)
    n_out = int(revs*pts_per_rev)
    if n_out < 64: return None, None
    th_u = np.linspace(theta[0], theta[-1], n_out, endpoint=False)
    t_axis = np.arange(len(x))/fs
    xu = np.interp(th_u, theta, t_axis)
    xr = np.interp(xu, t_axis, x)
    nps = 1 << int(np.round(np.log2(revs_per_seg*pts_per_rev)))
    nps = min(nps, n_out)
    o, p = _welch(xr, fs=pts_per_rev, nperseg=nps, noverlap=nps//2,
                  window='hann', scaling='spectrum', detrend='constant')
    A = np.sqrt(np.maximum(p, 0))*np.sqrt(2)
    m = o <= n_orders_max
    return o[m], A[m]

# =========================================================== 5. line SNR
def line_snr_db(f_or_o, amp, target, rel_tol=0.015, guard=4, noise_w=30):
    """SNR of the strongest line within rel_tol of target, against the local
    median floor. The common currency for judging every enhancement above."""
    idx = np.where(np.abs(f_or_o-target) <= rel_tol*target)[0]
    if idx.size == 0: return np.nan
    j = idx[int(np.argmax(amp[idx]))]
    lo = max(0, j-noise_w); hi = min(len(amp), j+noise_w+1)
    ring = np.r_[amp[lo:max(lo, j-guard)], amp[min(hi, j+guard+1):hi]]
    if ring.size == 0: return np.nan
    return float(20*np.log10(max(amp[j], 1e-12)/max(np.median(ring), 1e-12)))
