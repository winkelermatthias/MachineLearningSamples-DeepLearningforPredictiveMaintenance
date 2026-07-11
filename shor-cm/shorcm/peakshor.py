"""Peak-Shor: period finding directly on the spectral peak set.

The Shor reading taken literally: the measured peak set is a noisy sample
of a rational lattice {(p/q) f0}. Continued-fraction rationalization of
PEAK-PAIR RATIOS recovers the lattice generator with NO speed hypothesis:
if f_i = m f0 and f_j = n f0 then f_j/f_i snaps to n/m and f_i/m votes
for f0. Applied greedily (remove explained peaks, repeat) it finds every
periodicity present — own shaft, neighbor machine, electrical family —
which is the skeleton of the O2 pattern ledger.

Differences from blindspeed.structure_score: scoring happens in the Hz
domain on the peak list (no angular resampling per candidate, ~20x
cheaper), so every candidate gets the full rational-structure treatment;
angular refinement is reserved for the final top-k. Alias signatures are
explicit NEGATIVE evidence in both directions (PHASE2 2.2):
 - half-integer (q=2) forest with a genuine sub-1 member  -> c = 2 f0
 - even-integer-only family with weak 1x                  -> c = f0 / 2
"""
import numpy as np
from fractions import Fraction

from . import tacho as T

GRID_HUM = (50.0, 60.0, 100.0, 120.0)
LO, HI = 3.0, 100.0


def spectral_peaks(x, fs, fmax=2000.0, n_peaks=45, guard=4.0, halfwin=30):
    """Full-resolution peak list: (freqs, amps, concentration).
    Local maxima above guard * local median, parabolic sub-bin freq."""
    x = np.asarray(x, float)
    w = np.hanning(len(x))
    A = np.abs(np.fft.rfft((x - x.mean()) * w)) / (w.sum() / 2)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    df = f[1] - f[0]
    imax = int(fmax / df)
    A = A[:imax]; f = f[:imax]
    cand = []
    for i in range(2, len(A) - 2):
        if A[i] > A[i - 1] and A[i] >= A[i + 1]:
            loc = np.median(A[max(i - halfwin, 0):i + halfwin]) + 1e-15
            if A[i] > guard * loc:
                d = 0.5 * (A[i - 1] - A[i + 1]) / (
                    A[i - 1] - 2 * A[i] + A[i + 1] + 1e-15)
                d = float(np.clip(d, -0.5, 0.5))
                conc = A[i - 1:i + 2].sum() / (A[i - 12:i + 13].sum() + 1e-12)
                cand.append((float(f[i] + d * df), float(A[i]), float(conc)))
    cand.sort(key=lambda t: -t[1])
    cand = cand[:n_peaks]
    if not cand:
        return np.array([]), np.array([]), np.array([])
    pf, pa, pc = map(np.array, zip(*sorted(cand)))
    return pf, pa, pc


def _is_hum(f, tol=1.5):
    return any(abs(f - h) < tol for h in GRID_HUM)


def pair_candidates(pf, pa, mmax=10, rtol=0.01, lo=LO, hi=HI, topn=10):
    """f0 candidates from CF-snapped peak-pair ratios, weighted vote.
    Each pair (i, j), f_i < f_j: snap f_j/f_i -> n/m (n, m <= mmax);
    implied fundamental f_i/m votes with weight sqrt(a_i a_j)."""
    votes = []                            # (f0, weight)
    n = len(pf)
    for i in range(n):
        if _is_hum(pf[i]):
            continue
        for j in range(i + 1, n):
            if _is_hum(pf[j]):
                continue
            r = pf[j] / pf[i]
            fr = Fraction(r).limit_denominator(mmax)
            if fr.numerator > mmax or fr.numerator == 0:
                continue
            if abs(float(fr) - r) > rtol * r:
                continue
            f0 = pf[i] / fr.denominator
            if lo <= f0 <= hi:
                votes.append((f0, float(np.sqrt(pa[i] * pa[j]))))
    if not votes:
        return []
    votes.sort()
    clusters = []                         # [f0_weighted, w_total]
    for f0, w in votes:
        if clusters and abs(f0 / (clusters[-1][0] / clusters[-1][1]) - 1) < 0.012:
            clusters[-1][0] += f0 * w
            clusters[-1][1] += w
        else:
            clusters.append([f0 * w, w])
    out = [(c[0] / c[1], c[1]) for c in clusters]
    out.sort(key=lambda t: -t[1])
    return [f for f, _ in out[:topn]]


def _amp_at_order(pf, pa, f0, order, rtol=0.012):
    m = np.abs(pf / f0 - order) < rtol * max(order, 1.0)
    return float(pa[m].max()) if m.any() else 0.0


def hz_structure_score(pf, pa, f0, qmax=8, max_order=12.0, rtol=0.01):
    """Rational-structure score of hypothesis f0 on the raw peak list.
    Same evidence vocabulary as blindspeed.structure_score plus the
    double-speed (q=2 forest) negative signature."""
    if len(pf) == 0:
        return -9.0, {}
    keep = ~np.array([_is_hum(f) for f in pf])
    pf_, pa_ = pf[keep], pa[keep]
    med = np.median(pa_) + 1e-15
    e_snap = e_tot = e_odd = 0.0
    n_int = n_evenint = 0
    for f, a in zip(pf_, pa_):
        o = f / f0
        if o > max_order:
            continue
        e_tot += a
        fr = Fraction(o).limit_denominator(qmax)
        if fr.numerator == 0:
            continue
        # C3 consequence: q >= 2 snaps also need small numerator; a peak
        # snapping to 11/7 is a false snap, not kinematics
        if fr.denominator >= 2 and fr.numerator > 10:
            fr = None
        if fr is not None and abs(float(fr) - o) <= max(rtol * o, 0.006):
            e_snap += a
            if fr.denominator == 1:
                n_int += 1
                if fr.numerator % 2 == 0:
                    n_evenint += 1
            if fr.numerator % 2 == 1:
                e_odd += a
    a05 = _amp_at_order(pf_, pa_, f0, 0.5)
    a1 = _amp_at_order(pf_, pa_, f0, 1.0)
    a15 = _amp_at_order(pf_, pa_, f0, 1.5)
    a2 = _amp_at_order(pf_, pa_, f0, 2.0)
    a25 = _amp_at_order(pf_, pa_, f0, 2.5)
    a3 = _amp_at_order(pf_, pa_, f0, 3.0)
    a35 = _amp_at_order(pf_, pa_, f0, 3.5)
    s1x = a1 / med
    frac = e_snap / (e_tot + 1e-12)
    odd = e_odd / (e_snap + 1e-12)
    # half-speed alias: even-integer family, order 1 weak
    alias_pen = 1.0 if (n_int >= 2 and n_evenint == n_int and s1x < 5) else 0.0
    # double-speed alias: 0.5/1.5 members without the 2.5/3.5 that real
    # looseness would bring (they would be true 5x/7x, absent for aliases)
    q2_pen = 1.0 if (a05 > 0.2 * a1 and a15 > 0.2 * a1
                     and (a25 + a35) < 0.15 * max(a1, a05)) else 0.0
    # electrical: order 2 dominant with no order-3 companion
    elec_pen = 1.0 if (a2 > 2.5 * a1 and a3 < 0.2 * a2) else 0.0
    score = (2.2 * np.log1p(s1x) + 3.0 * frac + 1.2 * odd
             + 1.6 * np.log1p(3 * e_snap / med / 10)
             - 2.5 * alias_pen - 2.5 * q2_pen - 3.0 * elec_pen)
    return float(score), {"s1x": round(s1x, 1), "snap_frac": round(frac, 2),
                          "odd": round(odd, 2), "alias_pen": alias_pen,
                          "q2_pen": q2_pen, "elec_pen": elec_pen}


def estimate_speed_shor(x, fs, meta=None, top=3, refine=True,
                        extra_candidates=()):
    """Blind speed via peak-Shor. Returns the same contract as
    blindspeed.estimate_speed: list of {hz, confidence, ev}."""
    pf, pa, pc = spectral_peaks(x, fs)
    cands = list(pair_candidates(pf, pa))
    cands += [c for c in extra_candidates if LO <= c <= HI]
    if not cands:                          # drowned spectrum: fall back to
        from . import blindspeed as BS     # the comb grid, then give up
        f, A = BS._spec(x, fs)             # gracefully with a null answer
        cands = BS.comb_candidates(f, A, topn=4)
    if not cands:
        return [{"hz": float("nan"), "confidence": 0.0, "ev": {}}]
    # octave/harmonic neighbors so alias partners always compete
    full = []
    for c in cands:
        for mlt in (1.0, 0.5, 2.0, 2 / 3, 1.5):
            v = c * mlt
            if LO <= v <= HI and all(abs(v / u - 1) > 0.012 for u in full):
                full.append(v)
    scored = []
    for c in full[:30]:
        sc, ev = hz_structure_score(pf, pa, c)
        scored.append((sc, c, ev))
    scored.sort(key=lambda t: -t[0])
    out = scored[:max(top, 3)]
    if refine:
        ref = []
        for sc, c, ev in out:
            try:
                _, mtaa = T.phase_from_comb(x, fs, f_nom=c,
                                            prior_rel_sigma=0.008)
                c = float(mtaa["rate_hz"])
            except Exception:
                pass
            dup = next((k for k, (_, c0, _) in enumerate(ref)
                        if abs(c / c0 - 1) < 0.008), None)
            if dup is None:
                ref.append((sc, c, ev))
        out = ref
    sc = np.array([s for s, _, _ in out])
    conf = np.exp(sc - sc.max()); conf = conf / conf.sum()
    return [{"hz": round(c, 3), "confidence": round(float(w), 3), "ev": ev}
            for (s, c, ev), w in zip(out[:top], conf[:top])]


# ---------------- multi-periodicity pattern ledger ----------------

def pattern_ledger_peaks(pf, pa, pc, f0, qmax=8, rtol=0.012):
    """Greedy assignment of every peak to a named pattern family.
    Families: SHAFT (integer orders of f0), HALF (q=2 ladder), RATIONAL
    (other small-q), HUM (grid values), NEARRAT (unsnapped 1.8..9),
    NEIGHBOR (integer lattice of a second fundamental found by re-running
    pair_candidates on leftovers), OTHER."""
    n = len(pf)
    label = np.array(["OTHER"] * n, dtype=object)
    order = pf / f0
    # NOTE: no width-clustering here. In the raw Hz domain the whole
    # shaft lattice is wander-smeared, so width does not discriminate
    # drifting tones the way it does after angular resampling. Bearing
    # (drifting) patterns are the ORDER-domain ledger's job
    # (simforge_corpus.ledger); this Hz ledger covers lattices and
    # fixed lines.
    for i in range(n):
        if _is_hum(pf[i]):
            label[i] = "HUM"
            continue
        o = order[i]
        if o > 12.5:
            continue
        fr = Fraction(o).limit_denominator(qmax)
        small = fr.numerator and (fr.denominator == 1 or fr.numerator <= 10)
        if small and abs(float(fr) - o) <= max(rtol * o, 0.006):
            if fr.denominator == 1:
                label[i] = "SHAFT"
            elif fr.denominator == 2:
                label[i] = "HALF"
            else:
                label[i] = "RATIONAL"
        elif 1.8 < o < 9.0:
            label[i] = "NEARRAT"
    # second periodicity on leftovers -> NEIGHBOR. Candidates come from
    # OTHER + NEARRAT peaks, but only STABLE peaks (high concentration)
    # may be relabeled: a bearing tone's phase walk smears it, a neighbor
    # machine's line does not. This is the NEARRAT-vs-NEIGHBOR
    # discriminator at zero extra cost.
    rest = (label == "OTHER") | (label == "NEARRAT")
    if rest.sum() >= 2:
        c2 = pair_candidates(pf[rest], pa[rest], topn=1, lo=2.0, hi=120.0)
        if c2:
            f1 = c2[0]
            for i in np.flatnonzero(rest):
                o = pf[i] / f1
                if (abs(o - round(o)) <= max(rtol * o, 0.01)
                        and round(o) >= 1 and pc[i] >= 0.45):
                    label[i] = "NEIGHBOR"
    e_tot = pa.sum() + 1e-12
    fams = {}
    for i in range(n):
        fams.setdefault(label[i], []).append(i)
    ledger = []
    for fam, idx in sorted(fams.items()):
        idx = np.array(idx)
        ledger.append({"family": fam, "n": int(len(idx)),
                       "energy_share": float(pa[idx].sum() / e_tot),
                       "freqs_hz": [round(float(v), 2) for v in pf[idx]],
                       "amps": [round(float(v), 3) for v in pa[idx]]})
    ledger.sort(key=lambda d: -d["energy_share"])
    return ledger
