"""Mechanism gate tests for the general pattern decomposer (T46-T49)."""
import numpy as np
from shorcm import patterns as PT
from shorcm import simforge_v2 as V2

FS = 16384
DUR = 4.0


def _mk(f0=25.0, seed=1, harm=None, half=None, sb=None, near=None,
        hum=0.10, noise=0.35, carrier_visible=True):
    """Constructed waveform with EXACTLY known composition. Returns
    (x, truth_energies dict)."""
    rng = np.random.default_rng(seed)
    n = int(DUR * FS)
    t = np.arange(n) / FS
    # real machines always wander; a wander-free shaft is genuinely
    # indistinguishable from a fixed-Hz source
    f_inst = f0 * (1 + 0.005 * np.sin(2 * np.pi * 0.7 * t + 1.0))
    ph = 2 * np.pi * np.cumsum(f_inst) / FS
    x = noise * rng.standard_normal(n)
    tr = {}
    if harm:
        e = 0.0
        for k, a in harm.items():
            x += a * np.cos(k * ph + rng.uniform(0, 6.28))
            e += a ** 2 / 2
        tr["HARM"] = e
    if half:
        e = 0.0
        for k, a in half.items():
            x += a * np.cos(k * ph + rng.uniform(0, 6.28))
            e += a ** 2 / 2
        tr["HALFHARM"] = e
    if sb:
        c, d, a_c, a_s = sb
        e = 0.0
        if carrier_visible:
            x += a_c * np.cos(c * ph + rng.uniform(0, 6.28))
            e += a_c ** 2 / 2
        for k in (1, 2):
            for sgn in (-1, 1):
                x += (a_s / k) * np.cos((c + sgn * k * d) * ph
                                        + rng.uniform(0, 6.28))
                e += (a_s / k) ** 2 / 2
        tr["SIDEBAND"] = e
    if near:
        o_b, a_b = near[0], near[1]
        walk = near[2] if len(near) > 2 else 0.6
        rw = np.cumsum(rng.normal(0, walk / np.sqrt(FS / f0), n))
        x += a_b * np.cos(o_b * ph + rw)
        tr["NEARRAT"] = a_b ** 2 / 2
    if hum:
        # 60 Hz: order 2.4 at f0=25 — NOT shaft-coincident (100 Hz would
        # be exactly order 4, a genuinely undecidable collision)
        x += hum * np.cos(2 * np.pi * 60.0 * t + rng.uniform(0, 6.28))
        tr["FIXEDHZ"] = hum ** 2 / 2
    return x, tr


def _shares(pats):
    d = {}
    for p in pats:
        d[p["type"]] = d.get(p["type"], 0.0) + p["energy"]
    return d


def test_T46_isolation_and_attribution():
    """Mixed composition: every family recovered with attribution error
    < 10% of its true energy... measured as dB error < 1.5, shares +
    FLOOR sum to 1, no double counting."""
    x, tr = _mk(harm={1: 0.5, 2: 0.25, 3: 0.15},
                half={0.5: 0.25, 1.5: 0.18, 2.5: 0.12},
                sb=(37.0, 1.0, 0.4, 0.2),
                near=(3.57, 0.5))
    pats, e_tot = PT.decompose(x, FS, 25.0)
    got = _shares(pats)
    for fam, e_true in tr.items():
        assert fam in got, (fam, list(got))
        err = 10 * np.log10((got[fam] + 1e-12) / e_true)
        assert abs(err) < 1.5, (fam, e_true, got[fam], err)
    share_sum = sum(p["share"] for p in pats)
    assert abs(share_sum - 1.0) < 1e-6, share_sum


def test_T47_suppressed_carrier_sideband():
    """A modulation fan WITHOUT its carrier must still be identified as
    SIDEBAND with the right spacing and carrier position."""
    x, tr = _mk(harm={1: 0.4, 2: 0.1}, sb=(41.0, 1.0, 0.0, 0.35),
                carrier_visible=False, near=None, half=None)
    pats, _ = PT.decompose(x, FS, 25.0)
    sbs = [p for p in pats if p["type"] == "SIDEBAND"]
    assert sbs, [p["type"] for p in pats]
    p = max(sbs, key=lambda q: q["energy"])
    assert abs(p["params"]["carrier"] - 41.0) < 0.5, p["params"]
    assert abs(p["params"]["spacing"] - 1.0) < 0.1, p["params"]
    assert not p["params"]["carrier_visible"]
    err = 10 * np.log10(p["energy"] / tr["SIDEBAND"])
    assert abs(err) < 2.0, err


def test_T48_half_record_stability():
    """PHASE2 metric: shares from the two halves of one record drift
    < 5 percentage points per family."""
    # moderate diffusion: stability is judged where detection holds in
    # each half; the extreme-walk detection limit is T35/T46 territory
    x, _ = _mk(harm={1: 0.5, 2: 0.25}, near=(3.57, 0.5, 0.35),
               sb=(37.0, 1.0, 0.4, 0.2))
    n2 = len(x) // 2
    s1 = _shares(PT.decompose(x[:n2], FS, 25.0)[0])
    s2 = _shares(PT.decompose(x[n2:], FS, 25.0)[0])
    e1 = sum(v for k, v in s1.items() if k != "FLOOR") + 1e-12
    e2 = sum(v for k, v in s2.items() if k != "FLOOR") + 1e-12
    for fam in ("HARM", "SIDEBAND", "NEARRAT"):
        d = abs(s1.get(fam, 0) / e1 - s2.get(fam, 0) / e2)
        assert d < 0.05, (fam, s1.get(fam), s2.get(fam), d)


def test_T50_crystal_hum_on_half_integer_recovered_shaft_not_stolen():
    """A mains line landing EXACTLY on a shaft half-integer order must
    still be recovered as FIXEDHZ (it is crystal-narrow in Hz; a
    coincident shaft harmonic would be smeared by wander). Converse: a
    genuine wandering shaft harmonic at the same position must NOT be
    stolen from the HARM family."""
    # hum at 87.5 Hz = order 3.5 at f0=25 — inside the old blanket
    # exemption window; wander 1.2% so a shaft-locked line there WOULD
    # smear over >3 Hz bins (the resolvability gate's territory)
    rng = np.random.default_rng(7)
    n = int(DUR * FS)
    t = np.arange(n) / FS
    f_inst = 25.0 * (1 + 0.012 * np.sin(2 * np.pi * 0.7 * t + 1.0))
    ph = 2 * np.pi * np.cumsum(f_inst) / FS
    x = 0.35 * rng.standard_normal(n) \
        + 0.5 * np.cos(ph) + 0.25 * np.cos(2 * ph) \
        + 0.14 * np.cos(2 * np.pi * 87.5 * t + 0.3)
    pats, _ = PT.decompose(x, FS, 25.0)
    fx = [p for p in pats if p["type"] == "FIXEDHZ"
          and abs(p["params"]["hz"] - 87.5) < 1.5]
    assert fx, [(p["type"], p["params"]) for p in pats]
    err = 10 * np.log10(fx[0]["energy"] / (0.14 ** 2 / 2))
    assert abs(err) < 2.5, err
    # converse: genuine wandering shaft harmonics at the same wander,
    # no hum anywhere — the HARM family must keep its energy
    rng2 = np.random.default_rng(8)
    x2 = 0.35 * rng2.standard_normal(n) \
        + 0.5 * np.cos(ph + 0.5) + 0.25 * np.cos(2 * ph + 1.1) \
        + 0.3 * np.cos(3 * ph + 2.0)
    e_true = (0.5 ** 2 + 0.25 ** 2 + 0.3 ** 2) / 2
    pats2, _ = PT.decompose(x2, FS, 25.0)
    got2 = _shares(pats2)
    err2 = 10 * np.log10((got2.get("HARM", 0) + 1e-12) / e_true)
    assert abs(err2) < 1.5, (err2, [(p["type"], p["params"])
                                    for p in pats2])


def test_T51_group_alarm_bearing_growing_in_its_fan():
    """A bearing whose energy grows in its modulation FAN while its
    tone sits still: instance-level NEARRAT stays quiet, but the
    kinematic group (NEARRAT + co-carrier SIDEBAND) must alarm."""
    rng = np.random.default_rng(9)
    tr = PT.GeneralTracker()
    for t in range(14):
        e_tone = 3e-3 * (1 + 0.1 * rng.standard_normal())
        e_fan = 2e-4 * 10 ** (1.3 * t / 13) \
            * (1 + 0.1 * rng.standard_normal())
        pats = [
            {"type": "NEARRAT", "params": {"order": 6.2}, "energy": e_tone,
             "members": [(6.2, e_tone)], "share": 0.1},
            {"type": "SIDEBAND",
             "params": {"carrier": 6.2, "spacing": 1.0,
                        "carrier_visible": True},
             "energy": e_fan, "members": [(6.2, e_fan)], "share": 0.1},
            {"type": "HARM", "params": {"base": 1.0}, "energy": 0.12,
             "members": [(1.0, 0.12)], "share": 0.5},
        ]
        tr.update(t, pats)
    tds = tr.trends()
    nr = {k: d for k, d in tds.items() if k.startswith("('NEARRAT'")}
    assert nr and not any(d["alarm"] for d in nr.values()), nr
    gps = tr.trends_grouped()
    bearing_gp = [g for g in gps if "NEARRAT" in g["types"]]
    assert bearing_gp, gps
    assert any(g["alarm"] for g in bearing_gp), bearing_gp
    assert "SIDEBAND" in bearing_gp[0]["types"], bearing_gp
    harm_gp = [g for g in gps if g["types"] == ["HARM"]]
    assert harm_gp and not any(g["alarm"] for g in harm_gp), gps


def test_T52_sheet_seeded_mesh_fan_under_heavy_wander():
    """A gearbox mesh fan at order 37 under 1.2% wander: the smear
    (0.44 orders) fuses carrier and sidebands. The kinematic sheet
    knows z = 37 — seeded extraction must recover the mesh energy
    within 2.5 dB and label it SIDEBAND at the right carrier."""
    rng = np.random.default_rng(11)
    n = int(DUR * FS)
    t = np.arange(n) / FS
    f_inst = 25.0 * (1 + 0.012 * np.sin(2 * np.pi * 0.7 * t + 1.0))
    ph = 2 * np.pi * np.cumsum(f_inst) / FS
    x = 0.35 * rng.standard_normal(n) \
        + 0.5 * np.cos(ph) + 0.25 * np.cos(2 * ph)
    e_true = 0.0
    for o_m, a in ((37.0, 0.5), (36.0, 0.25), (38.0, 0.25),
                   (35.0, 0.12), (39.0, 0.12)):
        x += a * np.cos(o_m * ph + rng.uniform(0, 6.28))
        e_true += a ** 2 / 2
    pats, _ = PT.decompose(x, FS, 25.0, sheet={"mesh": 37})
    sb = [p for p in pats if p["type"] == "SIDEBAND"
          and p["params"].get("seeded")
          and abs(p["params"]["carrier"] - 37.0) < 0.5]
    assert sb, [(p["type"], p["params"]) for p in pats]
    e_got = sum(p["energy"] for p in sb)
    err = 10 * np.log10(e_got / e_true)
    assert abs(err) < 2.5, err
    # control: healthy-ish record, no mesh energy — the seed must NOT
    # invent a pattern out of floor
    x2 = 0.35 * np.random.default_rng(12).standard_normal(n) \
        + 0.5 * np.cos(ph)
    pats2, _ = PT.decompose(x2, FS, 25.0, sheet={"mesh": 37})
    sb2 = [p for p in pats2 if p["type"] == "SIDEBAND"
           and p["params"].get("seeded")]
    assert not sb2 or sum(p["energy"] for p in sb2) < 0.1 * e_true, sb2


def _feed(tr, t, e):
    tr.update(t, [{"type": "NEARRAT", "params": {"order": 4.3},
                   "energy": e, "members": [(4.3, e)], "share": 0.2}])


def test_T53_cusum_recall_channel():
    """The supplementary CUSUM channel: fires on a genuine 14 dB
    growth (its value is RECALL, not earliness — the early-warning
    hypothesis was killed on dev: delay is set by detectability, not
    by the trend test's 6-point need) and stays quiet on a stationary
    noisy instance."""
    rng = np.random.default_rng(21)
    tr = PT.GeneralTracker()
    for t in range(16):
        sev = max(t - 3, 0) / 12
        e = 1e-3 * 10 ** (1.4 * sev) * (1 + 0.12 * rng.standard_normal())
        _feed(tr, t, max(e, 1e-6))
    assert any(d["alarm"] for d in tr.cusum().values()), tr.cusum()
    # stationary: fluctuating +/- but flat -> quiet
    tr2 = PT.GeneralTracker()
    for t in range(16):
        e = 3e-3 * (1 + 0.25 * rng.standard_normal())
        _feed(tr2, t, max(e, 1e-6))
    assert not any(d["alarm"] for d in tr2.cusum().values()), tr2.cusum()


def _env_near(pats, order, tol=0.06):
    targets = (order, 2 * order)
    for p in pats:
        cand = None
        if p["type"] in ("NEARRAT", "TONE"):
            cand = p["params"]["order"]
        elif p["type"] == "SIDEBAND":
            cand = p["params"]["carrier"]
        elif p["type"] == "HARM" and p["params"]["base"] > 1.5:
            cand = p["params"]["base"]
        if cand is not None and any(abs(cand / o - 1) < tol
                                    for o in targets):
            return True
    return False


def test_T54_impulsive_bearing_lives_in_the_envelope():
    """A SimForge impulsive bearing: the raw order spectrum carries
    little at the defect order (that is the measured MFPT physics),
    while the envelope channel captures it. Healthy control: the
    envelope must not invent the defect order."""
    rng = np.random.default_rng(0)
    found = 0
    for i in range(400):
        m = V2.sample_machine(V2.rng_for_run((70_000_000, i)))
        if m["fault"] != "bearing" or m["fault_shaft"] != "in":
            continue
        h = int(m["f_shaft"] * 1e6)
        if not (h % 100) / 100 < (0.75 - 0.35 * m["severity"]):
            continue
        if m["severity"] < 0.5:
            continue
        tr = []
        x = V2.synth_run(m, V2.rng_for_run((70_500_000, i)), truth=tr)
        tb = next(t for t in tr if t["family"] == "BEARING")
        assert tb["impulsive"]
        bo = tb["freqs_hz"][0] / m["f_shaft"]
        pats_env, _, band = PT.decompose_envelope(x, V2.FS, m["f_shaft"])
        assert _env_near(pats_env, bo), (bo, [
            (p["type"], p["params"]) for p in pats_env
            if p["type"] != "FLOOR"][:6])
        found += 1
        if found >= 3:
            break
    assert found >= 2, f"only {found} impulsive bearing cases sampled"
    # healthy control
    for i in range(200):
        m = V2.sample_machine(V2.rng_for_run((71_000_000, i)))
        if m["fault"] != "healthy":
            continue
        x = V2.synth_run(m, V2.rng_for_run((71_500_000, i)))
        pats_env, _, _ = PT.decompose_envelope(x, V2.FS, m["f_shaft"])
        assert not _env_near(pats_env, 3.245) \
            or not _env_near(pats_env, 4.755)
        break


def test_T49_general_tracking_under_varying_speed():
    """A growing SIDEBAND fan and a constant HARM family, speed moving
    +/-15% record to record: identities persist, the sideband instance
    alarms, the harmonic family stays quiet."""
    tr = PT.GeneralTracker()
    for t in range(14):
        f0 = 25.0 * (1 + 0.15 * np.sin(1.1 * t))
        a_s = 0.06 * 10 ** (0.9 * t / 13)
        x, _ = _mk(f0=f0, seed=200 + t, harm={1: 0.5, 2: 0.25},
                   sb=(37.0, 1.0, 0.3, a_s), near=None, half=None)
        pats, _ = PT.decompose(x, FS, f0)
        tr.update(t, pats)
    tds = tr.trends()
    sb_keys = [k for k in tds if k.startswith("('SIDEBAND'")]
    h_keys = [k for k in tds if k.startswith("('HARM', 1.0")]
    assert sb_keys and h_keys, list(tds)
    assert any(tds[k]["alarm"] for k in sb_keys), \
        {k: tds[k] for k in sb_keys}
    assert not any(tds[k]["alarm"] for k in h_keys), \
        {k: tds[k] for k in h_keys}
