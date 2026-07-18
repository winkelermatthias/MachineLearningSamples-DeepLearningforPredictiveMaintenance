"""O1 blind speed: candidates -> rational-structure scoring -> top-3.
Implements PHASE2.md sections 2.0 to 2.3 (calibration deferred to real
dev data; confidence here is a softmax over structure scores).
"""
import numpy as np
from scipy.signal import welch, hilbert, butter, sosfiltfilt
from . import tacho as T
from . import spectra as S

GRID = np.geomspace(3.0, 100.0, 900)


def _spec(x, fs, nper=8192):
    f, P = welch(x, fs, nperseg=nper)
    A = np.sqrt(P)
    return f, A / (np.median(A) + 1e-15)


def comb_candidates(f, A, k=6, topn=8):
    df = f[1] - f[0]
    sc = np.zeros(len(GRID))
    for h in range(1, k + 1):
        b = np.clip(np.round(GRID * h / df).astype(int), 0, len(f) - 1)
        sc += np.log1p(A[b])
    out, used = [], np.zeros(len(GRID), bool)
    for i in np.argsort(-sc):
        if used[i]:
            continue
        out.append(float(GRID[i]))
        used[np.abs(GRID / GRID[i] - 1) < 0.015] = True
        if len(out) >= topn:
            break
    return out


def cepstrum_candidate(x, fs):
    n = len(x)
    logA = np.log(np.abs(np.fft.rfft(x * np.hanning(n))) + 1e-9)
    c = np.abs(np.fft.irfft(logA - logA.mean()))
    q = np.arange(len(c)) / fs
    m = (q > 1 / 100.0) & (q < 1 / 3.0)
    if not m.any():
        return []
    return [float(1.0 / q[m][np.argmax(c[m])])]


def envelope_candidate(x, fs):
    sos = butter(4, 1000 / (fs / 2), btype="high", output="sos")
    e = np.abs(hilbert(sosfiltfilt(sos, x)))
    f, A = _spec(e - e.mean(), fs, nper=8192)
    m = (f > 3) & (f < 100)
    if not m.any():
        return []
    return [float(f[m][np.argmax(A[m])])]


def twolf_ladder(x, fs, is_motor_point, slip_mid=0.015):
    """Detect 2LF via HF sideband-comb SPACING (autocorr of the HF
    magnitude spectrum), plus the low-band 2LF line; emit shaft
    candidates at 2LF/2p * (1-slip_mid), p = 1..3."""
    if not is_motor_point:
        return []
    f, A = _spec(x, fs)
    hf = (f > 1500) & (f < fs * 0.45)
    a = A[hf] - A[hf].mean()
    ac = np.correlate(a, a, "full")[len(a) - 1:]
    df = f[1] - f[0]
    lag = np.arange(len(ac)) * df
    m = (lag > 15) & (lag < 380)          # 2LF: slow VFD to fast
    cands = []
    if m.any() and ac[m].max() > 4 * np.std(ac[len(ac) // 2:]):
        two_lf = float(lag[m][np.argmax(ac[m])])
        cands.append(two_lf)
    # low-band direct line near 100/120-ish (or 2 f_e anywhere 60-260)
    lb = (f > 30) & (f < 380)
    pk = float(f[lb][np.argmax(A[lb])])
    if A[lb].max() > 6:
        cands.append(pk)
    out = []
    for two_lf in cands:
        for p in (1, 2, 3):
            out.append(two_lf / (2 * p) * (1 - slip_mid))
    return out


def fixed_lines(x, fs, lo=20.0, hi=400.0):
    """Crystal-narrow lines (drive/hum/neighbor tones): concentration of a
    full-resolution peak above 0.6. Bearing tones smear (typ. < 0.35)."""
    Af = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    ff = np.fft.rfftfreq(len(x), 1 / fs)
    m = (ff > lo) & (ff < hi)
    med = np.median(Af[m]) + 1e-15
    out = []
    for i in np.flatnonzero(m)[1:-1]:
        if Af[i] > 8 * med and Af[i] >= Af[i - 1] and Af[i] > Af[i + 1]:
            conc = Af[i - 1:i + 2].sum() / (Af[i - 12:i + 13].sum() + 1e-12)
            if conc > 0.6:
                out.append(float(ff[i]))
    return out


def structure_score(x, fs, c, flines=()):
    """Score candidate c by rational structure of the order spectrum
    under hypothesis c. Returns (score, evidence, refined_rate)."""
    try:
        ph, meta = T.phase_from_comb(x, fs, f_nom=c, prior_rel_sigma=0.008)
        c = float(meta["rate_hz"])            # refined candidate
        xa = S.angular_resample(x, ph, spr=128)
        A, o = S.fine_order_spectrum(xa, spr=128)
    except Exception:
        return -9.0, {}, c
    pk = S.peak_orders(A, o, max_order=12.0, n_peaks=25, guard=4.0)
    if not pk:
        return -9.0, {}, c
    tol = S.snap_tol(max(int(len(xa) / 128), 10))
    e_snap = e_tot = e_odd = 0.0
    n_evenint = n_int = 0
    for oo, aa, _ in pk:
        e_tot += aa
        fr = S.cf_snap(oo, tol=max(tol, 0.004), qmax=8)
        if fr is not None:
            e_snap += aa
            if fr.denominator == 1:
                n_int += 1
                if fr.numerator % 2 == 0:
                    n_evenint += 1
            if fr.numerator % 2 == 1:
                e_odd += aa
    floor = np.median(A) + 1e-15
    def aab(oo):
        bi = int(np.argmin(np.abs(o - oo)))
        return float(A[max(bi - 1, 0):bi + 2].max())
    a1_abs, a2_abs, a3_abs = aab(1.0), aab(2.0), aab(3.0)
    s1x = a1_abs / floor
    frac = e_snap / (e_tot + 1e-12)
    odd = e_odd / (e_snap + 1e-12)
    alias_pen = 1.0 if (n_int >= 3 and n_evenint == n_int and s1x < 5) else 0.0
    # electrical signature: on the f_e hypothesis, order 2 (= 2 f_e,
    # magnetic pull) dominates order 1 with no order-3 family behind it.
    # Misalignment also lifts order 2 but always brings a 3x companion.
    elec_pen = 1.0 if (a2_abs > 2.5 * a1_abs and a3_abs < 0.2 * a2_abs) else 0.0
    score = (2.2 * np.log1p(s1x) + 3.0 * frac + 1.2 * odd
             + 1.6 * np.log1p(3 * e_snap) - 2.5 * alias_pen - 3.0 * elec_pen)
    return float(score), {"s1x": round(s1x, 1), "snap_frac": round(frac, 2),
                          "odd": round(odd, 2), "alias_pen": alias_pen,
                          "elec_pen": elec_pen}, float(c)


def estimate_speed(x, fs, meta=None, use_ladder=True, use_structure=True,
                   top=3):
    """meta: {'component': 'motor'|'pump'} from asset names, optional."""
    f, A = _spec(x, fs)
    cands = comb_candidates(f, A)
    cands += cepstrum_candidate(x, fs)
    cands += envelope_candidate(x, fs)
    if use_ladder:
        is_motor = meta is None or meta.get("component") == "motor"
        cands += twolf_ladder(x, fs, is_motor)
    # octave neighbors, dedupe within 1.5%, plausibility clip
    full = []
    for c in cands:
        for mlt in (1.0, 2.0, 0.5, 1.5):
            v = c * mlt
            if 3.0 <= v <= 100.0 and all(abs(v / u - 1) > 0.015 for u in full):
                full.append(v)
    full = full[:18]
    if not use_structure:
        scored = [(0.0, c, {}) for c in full[:top]]
    else:
        fl = fixed_lines(x, fs)
        tmp = []
        for c in full:
            sc_, ev_, c_ref = structure_score(x, fs, c, flines=fl)
            # dedupe on the REFINED value, keep the best score
            dup = next((j for j, (s0, c0, _) in enumerate(tmp)
                        if abs(c_ref / c0 - 1) < 0.01), None)
            if dup is None:
                tmp.append((sc_, c_ref, ev_))
            elif sc_ > tmp[dup][0]:
                tmp[dup] = (sc_, c_ref, ev_)
        scored = sorted(tmp, key=lambda t: -t[0])
    sc = np.array([s for s, _, _ in scored[:max(top, 3)]])
    conf = np.exp(sc - sc.max()); conf = conf / conf.sum()
    return [{"hz": round(c, 3), "confidence": round(float(w), 3), "ev": ev}
            for (s, c, ev), w in zip(scored[:top], conf[:top])]
