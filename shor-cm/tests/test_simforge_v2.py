"""Mechanism gate tests for SimForge v2 (T26-T29): kinematic identities,
spectral presence of generated families, bearing-size physics."""
import numpy as np
from shorcm import simforge_v2 as V2


def _machines(n=200, seed=1):
    rng = np.random.default_rng(seed)
    return [V2.sample_machine(rng) for _ in range(n)]


def test_T26_kinematic_identities():
    for m in _machines():
        if m["archetype"] == "pump_planetary":
            # carrier: fc (Zs + Zr) = fs Zs ; GMF = fc Zr
            assert abs(m["fc"] * (m["Zs"] + m["Zr"])
                       - m["f_shaft"] * m["Zs"]) < 1e-9
            assert abs(m["gmf"] - m["fc"] * m["Zr"]) < 1e-9
            assert m["Zr"] == m["Zs"] + 2 * m["Zp"]
        if m["archetype"] == "pump_gearbox1":
            assert abs(m["f2"] * m["z2"] - m["f_shaft"] * m["z1"]) < 1e-9
            assert abs(m["gmf"] - m["f_shaft"] * m["z1"]) < 1e-9
        if m["archetype"] == "fan_gearbox2":
            assert abs(m["gmf2"] - m["f2"] * m["z3"]) < 1e-9
        if m["archetype"] == "fan_belt":
            r = m["f2"] / (m["f_shaft"] * m["belt_ratio"])
            assert 0.97 < r < 1.0            # creep slip only slows


def test_T27_bearing_physics():
    rng = np.random.default_rng(2)
    for big in (True, False):
        for _ in range(60):
            b = V2.bearing_orders(rng, big)
            # BPFO + BPFI = Z exactly (kinematic identity)
            assert abs(b["BPFO"] + b["BPFI"] - b["Z"]) < 1e-9
            assert 0.35 < b["FTF"] < 0.5
            if big:
                assert b["Z"] >= 12 and b["BPFO"] > 4.5   # big & slow
            else:
                assert b["Z"] <= 12
    # size follows speed in the population
    ms = _machines(400, seed=3)
    slow = [m for m in ms if m["low_speed"]]
    fast = [m for m in ms if not m["low_speed"]]
    assert np.mean([m["brg_in"]["Z"] for m in slow]) > \
        np.mean([m["brg_in"]["Z"] for m in fast]) + 4


def test_T28_generated_families_present_in_spectrum():
    """For a gearbox and a planetary machine, the GMF line must actually
    exist in the FFT at the truth frequency; same for a roots blower
    lobe pass and a belt fan's second lattice."""
    rng = np.random.default_rng(4)
    seen = set()
    for i in range(400):
        tr = []
        m, x = V2.sample_run(i + 50_000, truth=tr)
        want = {"pump_gearbox1": "GMF", "pump_planetary": "GMF",
                "blower_roots": "PASSAGE", "fan_belt": "SHAFT2"}
        fam = want.get(m["archetype"])
        if fam is None or fam in seen or m["fault"] != "healthy":
            continue
        A = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        fr = np.fft.rfftfreq(len(x), 1 / V2.FS)
        t = next(t for t in tr if t["family"] == fam)
        f_t = t["freqs_hz"][0]
        if not (3 < f_t < 7000):
            continue
        i0 = int(round(f_t * len(x) / V2.FS))
        med = np.median(A[max(i0 - 400, 1):i0 + 400]) + 1e-12
        assert A[max(i0 - 3, 0):i0 + 4].max() > 4 * med, \
            (m["archetype"], fam, f_t)
        seen.add(fam)
        if len(seen) == 4:
            return
    assert len(seen) >= 3, f"only saw {seen}"


def test_T29_masquerade_population_exists():
    """v2 must contain motor points whose residual 1x is weak while the
    electrical 2LF line is strong — the population that punishes a
    misalignment rule without the 1x requirement."""
    n_masq = 0
    for m in _machines(300, seed=5):
        if m["component"] == "motor" and m["resid_1x"] < 0.12 \
                and m["fault"] == "healthy":
            n_masq += 1
    assert n_masq > 5, n_masq


def test_T30_spacing_candidates_recover_generator():
    from shorcm import peakshor as PS
    # gear-mesh comb: 700 Hz carrier with sidebands spaced 23.7 Hz, plus
    # unrelated peaks; the spacing must appear as a candidate
    f_s = 23.7
    pf = np.array([700 - 2 * f_s, 700 - f_s, 700.0, 700 + f_s,
                   700 + 2 * f_s, 111.0, 421.0])
    pa = np.array([0.2, 0.4, 1.0, 0.4, 0.2, 0.3, 0.25])
    cands = PS.spacing_candidates(pf, pa)
    assert any(abs(c / f_s - 1) < 0.02 for c in cands), cands


def test_T31_gear_rule_and_ledger_features():
    """A v2 gearbox run with a gear fault must produce GMF evidence in
    the ledger at TRUE speed, and the rules arm must call gear on a
    strong-sideband fixture but not on a bare mesh."""
    from shorcm import simforge_corpus as SC
    rng = np.random.default_rng(9)
    for _ in range(200):
        m = V2.sample_machine(rng)
        if m["archetype"] == "pump_gearbox1" and m["fault"] == "gear" \
                and m["severity"] > 0.7:
            x = V2.synth_run(m, rng)
            led = SC.ledger(x, V2.FS, m["f_shaft"], spr=256, uns_hi=16.0)
            assert led is not None
            assert led["gmf_rel"] > 5, led["gmf_rel"]
            assert abs(led["gmf_order"] - m["z1"]) < 1.5 \
                or led["gmf_sb_count"] >= 2, \
                (led["gmf_order"], m["z1"], led["gmf_sb_count"])
            break
    else:
        raise AssertionError("no severe gearbox gear fault sampled")
    base = {"n_half": 0, "uns_frac": 0.0, "a1": 0.2, "a2": 0.05,
            "a3": 0.0}
    strong = base | {"gmf_rel": 12, "gmf_sb_count": 4, "gmf_sb_energy": 0.9}
    bare = base | {"gmf_rel": 12, "gmf_sb_count": 1, "gmf_sb_energy": 0.1}
    assert SC.rules_from_ledger(strong) == "gear"
    assert SC.rules_from_ledger(bare) != "gear"
