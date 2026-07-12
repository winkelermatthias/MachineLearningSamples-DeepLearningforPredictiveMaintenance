"""Robustness pins for the decomposer against field-data pathologies
the loop has not met yet but deployment will: DC-offset + clipped
sensors, NaN dropouts, dead channels, short records, within-record
speed ramps, simultaneous faults, velocity-integrated channels, and
saturated impulsive bearings (the envelope channel is where those
live, per the MFPT ledger mechanism).

House rules: exact-truth fixtures like tests/test_patterns._mk, fixed
seeds, tolerance asserts in dB, every test fast.
"""
import numpy as np
import pytest

from shorcm import patterns as PT
from shorcm import simforge_v2 as V2

FS = 16384
DUR = 4.0


def _mk_harm(f0=25.0, seed=1, harm=None, noise=0.35, n=None, ramp=0.0):
    """Wandering-shaft harmonic ladder with EXACTLY known energy.
    ramp: total linear relative speed sweep across the record (the
    record's MEAN speed stays f0)."""
    harm = harm or {1: 0.5, 2: 0.25, 3: 0.15}
    rng = np.random.default_rng(seed)
    n = n or int(DUR * FS)
    t = np.arange(n) / FS
    if ramp:
        f_inst = f0 * (1 - ramp / 2 + ramp * t / t[-1])
    else:
        f_inst = f0 * (1 + 0.005 * np.sin(2 * np.pi * 0.7 * t + 1.0))
    ph = 2 * np.pi * np.cumsum(f_inst) / FS
    x = noise * rng.standard_normal(n)
    e = 0.0
    for k, a in harm.items():
        x += a * np.cos(k * ph + rng.uniform(0, 6.28))
        e += a ** 2 / 2
    return x, e, ph, rng


def _shares(pats):
    d = {}
    for p in pats:
        d[p["type"]] = d.get(p["type"], 0.0) + p["energy"]
    return d


def _db(got, true):
    return 10 * np.log10((got + 1e-12) / true)


def test_R1_dc_offset_and_clipping_tolerance():
    """A field sensor with a +3.0 DC offset and hard clipping at 80%
    of peak: the HARM ladder must still be recovered within 2 dB, the
    share closure must hold, and the DC step must not be reported as a
    FIXEDHZ line."""
    x, e_true, _, _ = _mk_harm(seed=4)
    pk = float(np.max(np.abs(x)))
    xc = np.clip(x, -0.8 * pk, 0.8 * pk) + 3.0
    pats, _ = PT.decompose(xc, FS, 25.0)
    got = _shares(pats)
    assert abs(_db(got.get("HARM", 0.0), e_true)) < 2.0, got
    assert abs(sum(p["share"] for p in pats) - 1.0) < 1e-6
    lows = [p for p in pats if p["type"] == "FIXEDHZ"
            and p["params"]["hz"] < 5.0]
    assert not lows, lows


def test_R2_nan_dropout_raises_no_silent_propagation():
    """Pin of CURRENT behavior: a NaN dropout burst makes decompose
    (and the envelope path) raise ValueError. The contract this pins
    is 'no silent NaN propagation into pattern energies' — if the
    library later chooses to handle NaNs, this pin must be replaced by
    a finiteness assertion, never deleted."""
    x, _, _, _ = _mk_harm(seed=5)
    xn = x.copy()
    xn[1000:1010] = np.nan
    with pytest.raises(ValueError):
        PT.decompose(xn, FS, 25.0)
    with pytest.raises(ValueError):
        PT.decompose_envelope(xn, FS, 25.0)


def test_R3_dead_channel_all_zeros_raises_cleanly():
    """Pin of CURRENT behavior: a dead (all-zero) channel raises
    ValueError instead of fabricating patterns from numerical noise.
    Deployment-relevant: historian gaps arrive as zero-filled blocks."""
    with pytest.raises(ValueError):
        PT.decompose(np.zeros(int(DUR * FS)), FS, 25.0)


def test_R4_short_record_1p5_seconds():
    """A 1.5 s record (37 revs at 25 Hz): decompose must work — HARM
    within 2 dB, closure intact. Pins that the spr-rev block machinery
    degrades gracefully instead of dying below the standard 4 s."""
    x, e_true, _, _ = _mk_harm(seed=6, n=int(1.5 * FS))
    pats, e_tot = PT.decompose(x, FS, 25.0)
    assert np.isfinite(e_tot) and e_tot > 0
    got = _shares(pats)
    assert abs(_db(got.get("HARM", 0.0), e_true)) < 2.0, got
    assert abs(sum(p["share"] for p in pats) - 1.0) < 1e-6


def test_R5_within_record_speed_ramp_at_mean_speed():
    """+8% linear speed ramp inside one record, decomposed at the MEAN
    speed: the 1x ladder smears ~0.08 orders at base and 0.24 at 3x,
    but block-local demodulation must still capture HARM within 2 dB.
    This is the coast-down/load-step case every plant record has."""
    x, e_true, _, _ = _mk_harm(seed=3, ramp=0.08)
    pats, _ = PT.decompose(x, FS, 25.0)
    got = _shares(pats)
    assert abs(_db(got.get("HARM", 0.0), e_true)) < 2.0, \
        (got, _db(got.get("HARM", 0.0), e_true))


def test_R6_two_simultaneous_faults_isolated():
    """Imbalance (1x ladder) + tonal bearing (near-rational 3.57x with
    random-walk phase) in ONE record: both families must be isolated
    with < 2.5 dB attribution error each — neither steals the other.
    Field machines rarely have the decency to fail one way at a time."""
    x, e_harm, ph, rng = _mk_harm(seed=5, harm={1: 0.6, 2: 0.2, 3: 0.1})
    rw = np.cumsum(rng.normal(0, 0.6 / np.sqrt(FS / 25.0), len(x)))
    x = x + 0.45 * np.cos(3.57 * ph + rw)
    e_near = 0.45 ** 2 / 2
    pats, _ = PT.decompose(x, FS, 25.0)
    got = _shares(pats)
    assert "HARM" in got and "NEARRAT" in got, list(got)
    assert abs(_db(got["HARM"], e_harm)) < 2.5, _db(got["HARM"], e_harm)
    assert abs(_db(got["NEARRAT"], e_near)) < 2.5, \
        _db(got["NEARRAT"], e_near)


def test_R7_velocity_integrated_signal_harm1_survives():
    """Some historians deliver velocity, not acceleration. Integrating
    (cumsum / fs, linear detrend) scales harmonic k by 1/(2 pi k f0)
    and tilts the floor 1/f — HARM(1) must still be found, dominate
    all other families, and match the analytically integrated truth
    within 2.5 dB."""
    harm = {1: 0.5, 2: 0.25, 3: 0.15}
    f0 = 25.0
    x, _, _, _ = _mk_harm(f0=f0, seed=6, harm=harm)
    v = np.cumsum(x) / FS
    idx = np.arange(len(v))
    v = v - np.polyval(np.polyfit(idx, v, 1), idx)
    e_true = sum((a / (2 * np.pi * k * f0)) ** 2 / 2
                 for k, a in harm.items())
    pats, _ = PT.decompose(v, FS, f0)
    got = _shares(pats)
    h1 = [p for p in pats if p["type"] == "HARM"
          and abs(p["params"].get("base", 0) - 1.0) < 0.1]
    assert h1, [(p["type"], p["params"]) for p in pats]
    assert abs(_db(got.get("HARM", 0.0), e_true)) < 2.5, \
        (got.get("HARM"), e_true)
    others = {k: v_ for k, v_ in got.items() if k not in ("HARM", "FLOOR")}
    assert all(got["HARM"] > v_ for v_ in others.values()), got


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


def test_R8_saturated_impulsive_bearing_envelope_survives():
    """Hard sensor saturation (symmetric clip at 60% of peak) on
    SimForge impulsive bearings: clipping flattens exactly the impulse
    tips the HF-resonance envelope feeds on, yet the defect order must
    still be captured in the envelope for every sampled case. Same
    deterministic candidate scan as test_T54 (seeds fixed)."""
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
        pk = float(np.max(np.abs(x)))
        xc = np.clip(x, -0.6 * pk, 0.6 * pk)
        pats_env, _, _ = PT.decompose_envelope(xc, V2.FS, m["f_shaft"])
        assert _env_near(pats_env, bo), (i, bo, [
            (p["type"], p["params"]) for p in pats_env
            if p["type"] != "FLOOR"][:6])
        found += 1
        if found >= 3:
            break
    assert found >= 3, f"only {found} impulsive bearing cases sampled"
