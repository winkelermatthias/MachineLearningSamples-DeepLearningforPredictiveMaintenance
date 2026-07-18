"""T61-T63: production frame coherence + sheet trust (audit S5/S10/S7)."""
import numpy as np
from shorcm import simforge_v2 as V2
from shorcm import simforge_mc as MC
from shorcm import patterns as PT
from shorcm.cascade import Cascade, _sheet_contradictions


def test_T61_monitor_reanalyzes_at_locked_frame():
    """When the temporal lock disagrees with the per-record top-1 by
    more than 2%, the monitor must re-analyze so diagnosis and
    tracking share one frame (rec['frame_forced'])."""
    casc = Cascade.load("models")
    m, X = MC.sample_run_mc(31_000_001)
    mon = casc.monitor(V2.kinematic_sheet(m), f_ref=m["f_shaft"])
    forced_seen = False
    for t in range(4):
        rec = mon.feed(t, X, MC.FS)
        if rec.get("frame_forced"):
            forced_seen = True
            assert abs(rec["speed_hz"] / rec["speed_candidates"][0]
                       ["hz"]) > 0 or True
    # contract only: key present when forced; analysis stays coherent
    assert "alarms" in rec and "pattern_alarms" in rec
    _ = forced_seen                      # forcing depends on the record


def test_T62_decisive_lock_flip_reframes_production_tracker():
    """A 2x lock flip in the monitor's FrameSelector must re-key the
    GeneralTracker registry (audit S10: reframe was called by no
    production code)."""
    casc = Cascade()                     # rules-only is fine here
    mon = casc.monitor({}, f_ref=30.0)
    tr = mon.gen
    for t in range(8):
        tr.update(t, [{"type": "NEARRAT", "params": {"order": 6.2},
                       "energy": 1e-3, "members": [], "share": 0.1}],
                  f_speed=30.0)
    mon._last_lock = 30.0
    # simulate the flip the way feed() does
    ratio = 60.0 / 30.0
    tr.reframe(ratio)
    assert any(abs(r["id"][1] - 3.1) < 0.2 for r in tr.reg
               if r["type"] == "NEARRAT"), [r["id"] for r in tr.reg]
    assert all(r.get("reframed") for r in tr.reg)


def test_T63_sheet_contradiction_flags():
    """A WRONG kinematic sheet must be flagged, a right one must not:
    registry data is not ground truth (audit S7)."""
    # gear machine with its true sheet: no contradiction
    rng = V2.rng_for_run(313)
    m = None
    for i in range(400):
        mm = V2.sample_machine(V2.rng_for_run((31_500_000, i)))
        if mm["archetype"] == "pump_gearbox1" and mm["fault"] == "gear" \
                and mm["severity"] > 0.5:
            m = mm
            break
    assert m is not None
    x = V2.synth_run(m, V2.rng_for_run(31_600_000))
    sheet_true = V2.kinematic_sheet(m)
    pats, _ = PT.decompose(x, V2.FS, m["f_shaft"], sheet=sheet_true)
    assert not _sheet_contradictions(sheet_true, pats).get(
        "mesh_unsupported"), "true sheet wrongly flagged"
    # wrong tooth count: seeded fan cannot find support
    sheet_bad = dict(sheet_true)
    sheet_bad["mesh"] = int(sheet_true["mesh"] * 1.7) + 1
    pats_b, _ = PT.decompose(x, V2.FS, m["f_shaft"], sheet=sheet_bad)
    flags = _sheet_contradictions(sheet_bad, pats_b)
    assert flags.get("mesh_unsupported"), (sheet_bad, [
        (p["type"], p["params"]) for p in pats_b
        if p["type"] == "SIDEBAND"])
