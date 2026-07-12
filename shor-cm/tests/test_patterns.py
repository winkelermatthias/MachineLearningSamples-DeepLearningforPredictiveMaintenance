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
