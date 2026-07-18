"""Mechanism gate tests for pattern-energy tracking (T32-T34)."""
import numpy as np
from shorcm import tracker as TK
from shorcm import simforge_v2 as V2


def _seq_tracker(logs, key="SHAFT_1", f=30.0):
    tr = TK.PatternTracker(f_ref=f)
    for t, e in enumerate(logs):
        tr.update(t, {key: e} if key.startswith("SHAFT") or key == "HALF"
                  else {key: (e, 3.1)}, f)
    return tr


def test_T32_growth_alarms_stationary_does_not():
    rng = np.random.default_rng(40)
    # growing: 12 dB rise over 16 records + noise
    grow = [1e-4 * 10 ** (1.2 * t / 16) * (1 + 0.15 * rng.standard_normal())
            for t in range(16)]
    tr = _seq_tracker(grow)
    td = tr.trends()["SHAFT_1"]
    assert td["alarm"], td
    # stationary fault: present, fluctuating, NOT growing -> indicator,
    # never an alarm
    stat = [3e-3 * (1 + 0.2 * rng.standard_normal()) for _ in range(16)]
    tr2 = _seq_tracker(stat)
    td2 = tr2.trends()["SHAFT_1"]
    assert not td2["alarm"], td2
    # online delay: alarm must fire before the end on the growing series
    assert tr.first_alarm("SHAFT_1") is not None


def test_T33_omega2_normalization_kills_speed_ramp_false_alarm():
    """A VFD ramp 25 -> 40 Hz raises raw 1x energy by (40/25)^4 in E
    units (amp ~ omega^2 in force -> here amp ~ omega^2 modeled via
    omega1x). Normalized SHAFT_1 must NOT alarm."""
    tr = TK.PatternTracker(f_ref=25.0)
    rng = np.random.default_rng(41)
    for t in range(16):
        f = 25.0 + 15.0 * t / 15
        amp = 0.2 * (f / 25.0) ** 2 * (1 + 0.05 * rng.standard_normal())
        tr.update(t, {"SHAFT_1": amp ** 2 / 2}, f)
    td = tr.trends()["SHAFT_1"]
    assert not td["alarm"], td


def test_T34_order_invariant_tracking_under_variable_speed():
    """Same machine, same severity, speed varying +/-20% record to
    record: tracked SHAFT_2 and NEARRAT energies must stay flat (no
    trend), and NEARRAT association must survive the speed changes."""
    rng0 = V2.rng_for_run(777)
    m = V2.sample_machine(rng0)
    m["fault"], m["subtype"], m["fault_shaft"] = "bearing", "BPFO", "in"
    m["severity"] = 0.8
    m["archetype"], m["population"] = "pump_direct", "vfd"
    m["f2"] = m["f_shaft"]
    m.setdefault("vanes", 6)
    f_base = min(max(m["f_shaft"], 20.0), 45.0)
    tr = TK.PatternTracker(f_ref=f_base)
    got = 0
    for t in range(8):
        sc = 1.0 + 0.2 * np.sin(2.2 * t)
        mm = dict(m)
        mm["f_shaft"] = f_base * sc
        mm["f_e"] = mm["f_shaft"] * mm["pole_pairs"] / (1 - mm["slip"])
        rng = V2.rng_for_run(778 + t)
        x = V2.synth_run(mm, rng)
        en = TK.pattern_energies(x, V2.FS, mm["f_shaft"])
        assert en is not None
        if en["NEARRAT"][0] > 0:
            got += 1
        tr.update(t, en, mm["f_shaft"])
    assert got >= 6, f"NEARRAT found in only {got}/8 records"
    td = tr.trends(min_n=6).get("NEARRAT")
    assert td is not None and not td["alarm"], td


def test_T35_band_energy_calibration():
    """A phase-walking tone of known amplitude: band-integrated NEARRAT
    energy must land near a^2/2 (within ~2.5 dB) and grow monotonically
    with amplitude — the peak-tip sum underestimated it badly."""
    fs, dur, f0 = 16384, 4.0, 33.0
    n = int(dur * fs)
    rng = np.random.default_rng(50)
    t = np.arange(n) / fs
    ph = 2 * np.pi * f0 * t
    es = []
    for amp in (0.3, 0.6, 1.2):
        rw = np.cumsum(rng.normal(0, 0.6 / np.sqrt(fs / f0), n))
        x = 0.3 * rng.standard_normal(n) \
            + 0.25 * np.cos(ph + 0.1) \
            + amp * np.cos(3.57 * ph + rw)
        en = TK.pattern_energies(x, fs, f0)
        assert en is not None and en["NEARRAT"][0] > 0
        e, o_c = en["NEARRAT"]
        assert abs(o_c - 3.57) < 0.08, o_c
        err_db = 10 * np.log10(e / (amp ** 2 / 2))
        assert abs(err_db) < 2.5, (amp, e, err_db)
        es.append(e)
    assert es[0] < es[1] < es[2]


def test_T36_temporal_speed_lock_suppresses_octave_flips():
    cands_good = [{"hz": 30.0, "confidence": 0.5},
                  {"hz": 60.0, "confidence": 0.3},
                  {"hz": 15.0, "confidence": 0.2}]
    # estimator flips to the octave with higher confidence; history says 30
    cands_flip = [{"hz": 60.0, "confidence": 0.55},
                  {"hz": 30.1, "confidence": 0.35},
                  {"hz": 15.0, "confidence": 0.10}]
    assert abs(TK.select_speed(cands_good, None) - 30.0) < 1e-9
    assert abs(TK.select_speed(cands_flip, 30.0) - 30.1) < 1e-9
    # genuine large operating move: history must not veto a strong,
    # clearly better candidate forever (band is wide)
    cands_move = [{"hz": 39.0, "confidence": 0.85},
                  {"hz": 78.0, "confidence": 0.15}]
    assert abs(TK.select_speed(cands_move, 30.0) - 39.0) < 1e-9


def test_T37_frame_selector_consistency_beats_flips_and_first_error():
    """Records of one machine at wandering speed; candidate lists contain
    the true frame and octave aliases. Confidence prefers the alias on
    some records INCLUDING record 1. FrameSelector must recover the true
    frame nearly everywhere; v1 select_speed locks onto record-1's error."""
    rng = np.random.default_rng(60)
    prof = {1.0: 1.0, 2.0: 0.45, 3.0: 0.3, 5.0: 0.22, 6.0: 0.35}
    T = 10
    sel = TK.FrameSelector(warmup=5)
    chosen, truth = [], []
    for t in range(T):
        f = 30.0 * (1 + 0.12 * np.sin(1.3 * t))
        truth.append(f)
        pf = np.array([k * f * (1 + rng.normal(0, 0.002)) for k in prof])
        pa = np.array([v * (1 + 0.1 * rng.normal()) for v in prof.values()])
        # confidence flips to the octave on even records (incl. t=0)
        if t % 2 == 0:
            cands = [{"hz": 2 * f, "confidence": 0.5},
                     {"hz": f, "confidence": 0.35},
                     {"hz": f / 2, "confidence": 0.15}]
        else:
            cands = [{"hz": f, "confidence": 0.5},
                     {"hz": 2 * f, "confidence": 0.3},
                     {"hz": f / 2, "confidence": 0.2}]
        chosen.append(sel.observe(cands, pf, pa))
    # apply retroactive warmup choices
    chosen[:5] = sel.warmup_choices
    ok = sum(abs(c / f - 1) < 0.02 for c, f in zip(chosen, truth))
    assert ok >= 9, (ok, [round(c / f, 2) for c, f in zip(chosen, truth)])
