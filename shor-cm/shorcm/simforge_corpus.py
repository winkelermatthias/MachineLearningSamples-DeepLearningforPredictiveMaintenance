"""SimForge corpus v1.1: massive, seed-regeneratable synthetic population.

Additive over simforge_lite. Two rules kept sacred:
 1. NEVER fitted to MAFAULDA (physics + standards knowledge only).
 2. Domain randomization is widened ONE-SIDED (harder only): extra
    broadband noise, extra resonance floor, extra neighbor tones are
    ADDED on top of the pinned v1 generator, never removed. Any result
    on v1.1 is therefore a lower bound on v1 performance.

Every run is regeneratable from (BASE_SEED, run index) alone; the corpus
is stored as a manifest of ground truth, not waveforms.

Also home of the shared O3 physics-level feature ledger (`ledger`) and
the transparent rules arm (`rules_from_ledger`), ported verbatim in
logic from scripts/94_eval_blindspeed.py so the rules and ML arms score
the same evidence.
"""
import numpy as np
from scipy.signal import sosfilt, butter

from . import simforge_lite as SF
from . import tacho as T
from . import spectra as S
from . import blindspeed as BS

BASE_SEED = 20260709


def rng_for_run(i):
    """Independent, reproducible stream per run index."""
    return np.random.default_rng(np.random.SeedSequence((BASE_SEED, i)))


def sample_run(i, truth=None):
    """(machine dict, waveform) for corpus index i. v1 machine + one-sided
    hardening: extra white noise 0..0.25, extra shaped floor 0..1.5,
    0 or 1 extra neighbor tone family. Pass truth=[] to capture the exact
    component composition (see simforge_lite.synth_run)."""
    rng = rng_for_run(i)
    m = SF.sample_machine(rng)
    x = SF.synth_run(m, rng, truth=truth)
    extra_noise = float(rng.uniform(0.0, 0.25))
    x = x + extra_noise * rng.standard_normal(len(x))
    extra_floor = float(rng.uniform(0.0, 1.5))
    if extra_floor > 0.05:
        fr = rng.uniform(600, 7000)
        lo = max(fr * 0.85, 50) / (SF.FS / 2)
        hi = min(fr * 1.15, 0.47 * SF.FS) / (SF.FS / 2)
        sos = butter(2, [lo, hi], btype="band", output="sos")
        x = x + extra_floor * sosfilt(sos, rng.standard_normal(len(x)))
    n_extra_neighbor = int(rng.random() < 0.35)
    if n_extra_neighbor:
        t = np.arange(len(x)) / SF.FS
        fn = rng.uniform(5, 90)
        for k in (1, 2):
            x += 0.10 / k * np.cos(2 * np.pi * k * fn * t + rng.uniform(0, 6.28))
        if truth is not None:
            truth.append({"family": "NEIGHBOR", "freqs_hz": [float(fn),
                          float(2 * fn)], "amps": [0.10, 0.05],
                          "drifting": False})
    m = dict(m)
    m["run_id"] = i
    m["extra_noise"] = extra_noise
    m["extra_floor"] = extra_floor
    m["extra_neighbor"] = n_extra_neighbor
    return m, x


# ---------------- shared O3 evidence ledger ----------------

def _fixed_bases(x, fs, f_hat):
    """FIXEDHZ mask bases: grid hum, non-shaft comb families, narrow
    (drive-stable) low-band lines. Logic from 94_eval PLUS the
    concentration gate on comb bases: a severe bearing's harmonic family
    can win the comb score, but its phase walk SMEARS the line, while a
    real fixed-Hz confuser is crystal-narrow. Without the gate the mask
    deletes the bearing's own evidence (found via the O4 paired-sweep
    guardrail failure: driver falling monotonically with severity)."""
    fz, Az = BS._spec(x, fs)
    Af = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    ff = np.fft.rfftfreq(len(x), 1 / fs)

    def conc_at(hz):
        i = int(round(hz * len(x) / fs))
        if i < 13 or i > len(Af) - 14:
            return 0.0
        j = i - 3 + int(np.argmax(Af[i - 3:i + 4]))
        return float(Af[j - 1:j + 2].sum() / (Af[j - 12:j + 13].sum() + 1e-12))

    bases = [50.0, 60.0, 100.0, 120.0]
    for c in BS.comb_candidates(fz, Az, topn=4):
        if all(abs(c / (f_hat * r) - 1) > 0.03 for r in (0.5, 1, 2, 3)) \
                and conc_at(c) > 0.5:
            bases.append(c)
    lbm = (ff > 30) & (ff < 380)
    med = np.median(Af[lbm]) + 1e-15
    idx = np.flatnonzero(lbm)
    for i in idx[1:-1]:
        if Af[i] > 8 * med and Af[i] >= Af[i - 1] and Af[i] > Af[i + 1]:
            conc = Af[i - 1:i + 2].sum() / (Af[i - 12:i + 13].sum() + 1e-12)
            if conc > 0.6:
                bases += [float(ff[i]), float(ff[i]) / 2.0]
    return bases


def ledger(x, fs, f_hat):
    """Physics-level evidence ledger under speed hypothesis f_hat.
    Returns dict of scalar features (None on phase-extraction failure).
    Never raw bins: orders, families, snap fractions, coherence ratios."""
    try:
        ph, meta = T.phase_from_comb(x, fs, f_nom=f_hat, prior_rel_sigma=0.008)
        xa = S.angular_resample(x, ph, 128)
        A, o = S.fine_order_spectrum(xa, 128)
        Z, ob = S.block_spectra(xa, 128, revs=5)
        _, _, ratio_b = S.coherent_split(Z)
    except Exception:
        return None
    bases = _fixed_bases(x, fs, f_hat)

    def masked(order):
        hz = order * f_hat
        return any(abs(hz - k * b) < 1.5 for b in bases for k in range(1, 6))

    def aabs(oo):
        bi = int(np.argmin(np.abs(o - oo)))
        return float(A[max(bi - 1, 0):bi + 2].max())

    def rat(oo):
        kb = int(round(oo * 5))
        return float(ratio_b[kb]) if kb < len(ratio_b) else 0.0

    a1, a2, a3 = aabs(1), aabs(2), aabs(3)
    halves = [aabs(k) for k in (0.5, 1.5, 2.5, 3.5)]
    n_half = int(sum(h > 0.10 for h in halves))
    e_half = float(sum(halves))
    pk = S.peak_orders(A, o, 9.0, 25, guard=5.0)
    tol = max(S.snap_tol(max(int(len(xa) / 128), 10)), 0.004)
    do = o[1] - o[0]

    def narrow(oo, a):
        i = int(round(oo / do))
        loc = np.median(A[max(i - 40, 0):i + 40]) + 1e-15
        return a > 4.0 * loc

    # cluster adjacent peaks: a drifting tone (bearing) smears into a
    # GROUP of sub-peaks whose members false-snap individually (C3) and
    # whose own sidebands defeat the local-floor test. Cluster width is
    # the drift discriminator: locked tones are resolution-narrow, and a
    # wide cluster cannot be a fixed line, so it is exempt from masking.
    e_snap = e_tot = e_odd = 0.0
    uns = 0.0
    uns_best = (0.0, 0.0)                 # (amp, order) of top drifting tone
    clusters = []
    for oo, aa, _ in sorted(pk):
        if clusters and oo - clusters[-1][-1][0] < 0.008 * oo:
            clusters[-1].append((oo, aa))
        else:
            clusters.append([(oo, aa)])
    for cl in clusters:
        a_c = sum(a for _, a in cl)
        o_c = sum(o * a for o, a in cl) / (a_c + 1e-15)
        width = (cl[-1][0] - cl[0][0]) / o_c if len(cl) > 1 else 0.0
        e_tot += a_c
        drifting = width > 0.004 or len(cl) >= 3
        if drifting:
            if 1.8 < o_c < 9.0:
                uns += a_c
                if a_c > uns_best[0]:
                    uns_best = (a_c, o_c)
            continue
        fr = S.cf_snap(o_c, tol=tol, qmax=8)
        # C3 rule: q >= 2 snaps need a small numerator too
        if fr is not None and fr.denominator >= 2 and fr.numerator > 10:
            fr = None
        if fr is not None:
            e_snap += a_c
            if fr.numerator % 2 == 1:
                e_odd += a_c
        elif 1.8 < o_c < 9.0 and not masked(o_c) and narrow(o_c, a_c):
            uns += a_c
            if a_c > uns_best[0]:
                uns_best = (a_c, o_c)
    tot = e_tot + 1e-12
    led = {"a1": a1, "a2": a2, "a3": a3, "a2_over_a1": a2 / (a1 + 1e-9),
           "n_half": n_half, "e_half": e_half,
           "snap_frac": e_snap / tot, "odd_frac": e_odd / (e_snap + 1e-12),
           "uns_frac": uns / tot, "uns_abs": uns, "e_tot": float(tot),
           "uns_top_amp": uns_best[0],
           "uns_top_order": uns_best[1],
           "ratio_1": rat(1.0), "ratio_2": rat(2.0), "ratio_3": rat(3.0),
           "ratio_half": rat(0.5),
           "n_peaks": len(pk), "floor": float(np.median(A))}
    # PPA z at the strongest unsnapped tone: phase-walk continuity evidence
    if uns_best[0] > 0:
        from . import amplify as AMP
        zb = AMP.zoom_order_dft            # noqa: F841 (doc pointer)
        kb = uns_best[1]
        i_bin = int(round(kb * 5))
        if 0 < i_bin < Z.shape[1]:
            led["ppa_z_uns"] = float(AMP.ppa_zscore(Z[:, i_bin])[0])
        else:
            led["ppa_z_uns"] = 0.0
    else:
        led["ppa_z_uns"] = 0.0
    return led


def rules_from_ledger(led):
    """Transparent O3 rules arm. Misalignment rule per Matthias
    (2026-07-11): the 1x must be there when the 2x is — a bare 2x-ish
    line (which in the field is the electrical 2LF at order 2/(1-s), a
    small slip off exact 2) is NEVER misalignment. Guards: (a) a1 must
    be genuinely present, (b) the electrical shape (2x dominant with no
    3x companion) is excluded even when 1x exists."""
    if led is None:
        return "unknown"
    if led["n_half"] >= 2:
        return "looseness"
    if led["uns_frac"] > 0.15:
        return "bearing"
    elec_like = led["a2"] > 2.5 * led["a1"] and led["a3"] < 0.2 * led["a2"]
    if (led["a2"] > 0.35 and led["a2"] > 0.6 * led["a1"]
            and led["a1"] > 0.12 and not elec_like):
        return "misalignment"
    if led["a1"] > 0.5:
        return "imbalance"
    return "healthy"


LEDGER_FEATURES = ["a1", "a2", "a3", "a2_over_a1", "n_half", "e_half",
                   "snap_frac", "odd_frac", "uns_frac", "uns_abs", "e_tot",
                   "uns_top_amp", "uns_top_order", "ratio_1", "ratio_2",
                   "ratio_3", "ratio_half", "n_peaks", "floor", "ppa_z_uns"]
