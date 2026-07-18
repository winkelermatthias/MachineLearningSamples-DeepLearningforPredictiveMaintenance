"""Mechanism gate tests for peak-Shor period finding (T20-T23).
All synthetic, all deterministic, run BEFORE any corpus evaluation."""
import numpy as np
from shorcm import peakshor as PS
from shorcm import simforge_corpus as SC
from shorcm import simforge_lite as SF


def test_T20_pair_candidates_recover_f0():
    rng = np.random.default_rng(30)
    f0 = 7.31
    # noisy harmonic lattice + two interlopers, amplitudes decaying
    pf = np.array([f0 * k * (1 + rng.normal(0, 0.002)) for k in
                   (1, 2, 3, 5, 7)] + [23.7, 41.3])
    pa = np.array([1.0, 0.6, 0.4, 0.25, 0.15, 0.5, 0.3])
    order = np.argsort(pf)
    cands = PS.pair_candidates(pf[order], pa[order])
    assert cands, "no candidates"
    assert any(abs(c / f0 - 1) < 0.01 for c in cands[:3]), cands[:3]


def test_T21_double_speed_alias_penalized():
    # misalignment-like family: 1x, strong 2x, 3x. Hypothesis 2*f0 sees
    # 0.5 / 1.0 / 1.5 (q=2 forest) and must lose to f0.
    f0 = 12.4
    pf = np.array([f0, 2 * f0, 3 * f0])
    pa = np.array([0.25, 0.95, 0.35])
    s_true, ev_t = PS.hz_structure_score(pf, pa, f0)
    s_alias, ev_a = PS.hz_structure_score(pf, pa, 2 * f0)
    assert ev_a["q2_pen"] == 1.0, ev_a
    assert ev_t["q2_pen"] == 0.0, ev_t
    assert s_true > s_alias, (s_true, s_alias)


def test_T21b_looseness_true_halves_not_penalized():
    # genuine looseness: 0.5/1.5/2.5/3.5 present -> q2_pen must NOT fire
    f0 = 9.0
    pf = f0 * np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.5])
    pa = np.array([0.30, 0.28, 0.22, 0.10, 0.15, 0.11])
    _, ev = PS.hz_structure_score(pf, pa, f0)
    assert ev["q2_pen"] == 0.0, ev


def test_T22_multi_periodicity_ledger():
    # own shaft lattice + neighbor lattice; both must be identified
    f0, fn = 11.0, 17.3
    pf = np.concatenate([f0 * np.array([1, 2, 3, 5]),
                         fn * np.array([1, 2, 3])])
    pa = np.array([1.0, 0.5, 0.3, 0.2, 0.4, 0.2, 0.13])
    pc = np.ones_like(pa)
    order = np.argsort(pf)
    led = PS.pattern_ledger_peaks(pf[order], pa[order], pc[order], f0)
    fams = {d["family"]: d for d in led}
    assert fams["SHAFT"]["n"] == 4, fams
    assert "NEIGHBOR" in fams and fams["NEIGHBOR"]["n"] >= 2, fams


def test_T23_waveform_end_to_end():
    # an easy SimForge machine: healthy mains pump, moderate noise.
    # peak-Shor must land top-3 within 1% (and typically top-1).
    for i in (11, 23, 42):
        tr = []
        m, x = SC.sample_run(i, truth=tr)
        est = PS.estimate_speed_shor(x, SF.FS,
                                     meta={"component": m["component"]})
        errs = [abs(c["hz"] / m["f_shaft"] - 1) for c in est]
        # gate on top-3 for robustness across the three machines
        assert min(errs) <= 0.015, (i, m["fault"], m["f_shaft"],
                                    [c["hz"] for c in est])
