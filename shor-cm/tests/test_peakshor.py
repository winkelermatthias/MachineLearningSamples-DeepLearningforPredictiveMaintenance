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


def test_T25_misalignment_requires_1x():
    """Matthias's rule: 2x without 1x is never misalignment; 2x with a
    slip offset is electrical, not mechanical."""
    base = {"n_half": 0, "uns_frac": 0.0, "a3": 0.0}
    # electrical masquerade: strong exact-2 line, weak 1x (f_e capture)
    led = base | {"a1": 0.10, "a2": 0.55}
    assert SC.rules_from_ledger(led) != "misalignment"
    # electrical shape even with 1x present: 2x dominant, no 3x family
    led = base | {"a1": 0.15, "a2": 0.60}
    assert SC.rules_from_ledger(led) != "misalignment"
    # true misalignment: 1x present, strong 2x WITH its 3x companion
    led = base | {"a1": 0.30, "a2": 0.70, "a3": 0.25}
    assert SC.rules_from_ledger(led) == "misalignment"


def test_T38_sheet_conditioned_speed_recovers_input_shaft():
    """Gearbox-like peak set: weak input 1x, strong driven lattice and
    mesh. The plain estimator is drawn to the driven shaft; with the
    kinematic sheet (known ratio, mesh, vanes) the input frame wins."""
    import numpy as np
    f0, r = 24.0, 19 / 43           # input Hz, ratio z1/z2
    f2 = f0 * r
    pf = np.array(sorted([f0 * 1.0, f0 * 2.0,                 # weak input
                          f2, 2 * f2, 3 * f2, 6 * f2,         # driven+vane
                          19 * f0, 19 * f0 + f0, 19 * f0 - f0]))  # mesh
    amp = {f0: 0.10, 2 * f0: 0.05, f2: 0.5, 2 * f2: 0.25, 3 * f2: 0.15,
           6 * f2: 0.4, 19 * f0: 0.5, 19 * f0 + f0: 0.2,
           19 * f0 - f0: 0.2}
    pa = np.array([amp[f] for f in pf])
    sheet = {"ratio": r, "mesh": 19, "passage": 6}
    exp = PS.sheet_expected_orders(sheet)
    m_true = PS.sheet_match(pf, pa, f0, exp)
    m_driven = PS.sheet_match(pf, pa, f2, exp)
    assert m_true > m_driven + 0.2, (m_true, m_driven)
    # derived-candidate path: driven-lattice candidate / ratio -> input
    assert abs((f2 / r) / f0 - 1) < 1e-9


def test_T40_sheet_octave_arbitration_fixes_looseness_frame():
    """A looseness machine whose half-order lattice pulls the estimator
    into the f0/2 frame: with the kinematic sheet (known passage), the
    octave arbitration must return the TRUE frame on most records."""
    import numpy as np
    from shorcm import simforge_v2 as V2
    rng0 = V2.rng_for_run(30_000_016)
    m = V2.sample_machine(rng0)          # machine 16: gearbox, looseness
    assert m["fault"] == "looseness"
    ok = 0
    for t in range(6):
        rng = V2.rng_for_run((40_000_016, t))
        sc = float(np.clip(1 + 0.18 * np.sin(1.7 * t + 16)
                           + rng.normal(0, 0.03), 0.75, 1.25))
        mm = dict(m); mm["severity"] = 0.7
        mm["f_shaft"] = m["f_shaft"] * sc
        mm["f_e"] = mm["f_shaft"] * mm["pole_pairs"] / (1 - mm["slip"])
        for k in ("f2", "f3", "fc", "gmf", "gmf2"):
            if k in mm:
                mm[k] = mm[k] * sc
        x = V2.synth_run(mm, rng, omega1x=sc ** 2)
        est = PS.estimate_speed_sheet(x, V2.FS, V2.kinematic_sheet(m),
                                      meta={"component": "motor"})
        if abs(est[0]["hz"] / mm["f_shaft"] - 1) < 0.01:
            ok += 1
    assert ok >= 4, f"true frame on only {ok}/6 records"
