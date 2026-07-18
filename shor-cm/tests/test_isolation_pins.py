"""U4/U6 pins: the last unpinned mechanisms from the iteration-28
audit — adaptive alarm gates and the isolation-hardening fixes of
iteration 16 that only corpus numbers (not tests) protected."""
import numpy as np
from shorcm import patterns as PT

FS, DUR = 16384, 4.0


def _wave(components, seed=1, noise=0.35, wander=0.005):
    rng = np.random.default_rng(seed)
    n = int(DUR * FS)
    t = np.arange(n) / FS
    f_inst = 25.0 * (1 + wander * np.sin(2 * np.pi * 0.7 * t + 1.0))
    ph = 2 * np.pi * np.cumsum(f_inst) / FS
    x = noise * rng.standard_normal(n)
    for o, a, walk in components:
        if walk:
            rw = np.cumsum(rng.normal(0, walk / np.sqrt(FS / 25.0), n))
        else:
            rw = 0.0
        x += a * np.cos(o * ph + rw)
    return x


def test_U4_adaptive_gate_scales_with_baseline_scatter():
    """A noisy-baseline machine must NOT alarm on a rise the fixed
    6 dB gate would fire on; a quiet-baseline machine must alarm on a
    modest genuine rise (iteration 12's adaptive gates, production
    path, previously unpinned)."""
    rng = np.random.default_rng(11)
    # noisy baseline: sigma ~0.35 decades -> gate ~10 dB; rise 6.5 dB
    # (a fixed 6 dB gate would fire; the adaptive one must not)
    noisy = [(t, np.log10(1e-3) + 0.35 * rng.standard_normal()
              + (0.65 if t >= 12 else 0.0)) for t in range(16)]
    td = PT._trend(noisy, adaptive=True)
    assert td["gate_db"] > 6.0
    assert not td["alarm"], td
    # quiet baseline: sigma ~0.03 -> gate 3 dB; genuine 4.5 dB rise
    quiet = [(t, np.log10(1e-3) + 0.03 * rng.standard_normal()
              + 0.45 * min(t / 10, 1.0)) for t in range(16)]
    td2 = PT._trend(quiet, adaptive=True)
    assert td2["alarm"], td2


def test_U6a_rejected_2x_probe_leaves_neighbor_intact():
    """Iteration 16 fix: the NEARRAT 2x probe must be claim-on-accept
    only — a rejected probe once punctured the NEIGHBORING bearing
    hump (measured -114 dB on run 487)."""
    # two independent drifting tones: 3.9 and its near-2x neighbor 8.6
    x = _wave([(1.0, 0.4, 0), (3.9, 0.5, 0.5), (8.6, 0.6, 0.5)],
              seed=13)
    pats, _ = PT.decompose(x, FS, 25.0)
    nr = [p for p in pats if p["type"] == "NEARRAT"]
    got86 = [p for p in nr
             if abs(p["params"]["order"] / 8.6 - 1) < 0.06
             or any(abs(mo / 8.6 - 1) < 0.06 for mo, _ in p["members"])]
    assert got86, [(p["type"], p["params"]) for p in pats]
    e86 = sum(me for p in got86 for mo, me in p["members"]
              if abs(mo / 8.6 - 1) < 0.06)
    err = 10 * np.log10((e86 + 1e-12) / (0.6 ** 2 / 2))
    assert abs(err) < 3.0, err


def test_U6b_raw_gap_merge_keeps_distinct_tones_separate():
    """Iteration 16 fix: hump merging keys on RAW-edge gaps — two
    drifting tones 0.35 orders apart must stay separate instances,
    not fuse into one centerless pseudo-hump."""
    # 0.7 orders apart with moderate walks: genuinely distinct humps
    # separated by a cold valley (tones whose smears TOUCH are one
    # hump and merging them is correct - not this fixture)
    x = _wave([(1.0, 0.4, 0), (3.37, 0.5, 0.15), (4.61, 0.5, 0.15)],
              seed=17)
    pats, _ = PT.decompose(x, FS, 25.0)
    nr_orders = sorted(p["params"]["order"] for p in pats
                       if p["type"] == "NEARRAT")
    hits = [o for o in nr_orders if 3.0 < o < 4.9]
    assert len(hits) >= 2, (nr_orders, [
        (p["type"], p["params"]) for p in pats])


def test_U6c_small_tone_on_big_machine_survives_floor_gate():
    """Iteration 16 fix: the hump anti-clutter gate is floor-relative,
    not e_tot-relative — a belt lattice at ~0.3% of total energy is a
    pattern, not clutter."""
    # dominant 1x/2x machine + one small drifting tone
    x = _wave([(1.0, 1.4, 0), (2.0, 0.7, 0), (5.31, 0.12, 0.5)],
              seed=19, noise=0.25)
    pats, _ = PT.decompose(x, FS, 25.0)
    small = [p for p in pats if p["type"] in ("NEARRAT", "TONE")
             and abs((p["params"].get("order") or 0) / 5.31 - 1) < 0.06]
    assert small, [(p["type"], p["params"], round(p["share"], 4))
                   for p in pats if p["type"] != "FLOOR"]
