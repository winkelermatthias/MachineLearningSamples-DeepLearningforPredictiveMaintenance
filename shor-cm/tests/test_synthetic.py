"""G2 gate: five synthetic unit tests. NO real-data feature extraction is
allowed until `pytest -q` passes. These calibrate the whole method."""
import numpy as np
import pytest
from shorcm import tacho as T
from shorcm import spectra as S

FS = 50_000
RNG = np.random.default_rng(20260709)


def synth(f0=30.0, dur=5.0, wander=0.02, wander_hz=0.5, tones=(),
          noise=0.5, fs=FS, seed=0):
    """tones: list of (order, amp, locked: bool, slip)."""
    rng = np.random.default_rng(seed)
    n = int(dur * fs)
    t = np.arange(n) / fs
    f_inst = f0 * (1 + wander * np.sin(2 * np.pi * wander_hz * t))
    phase = 2 * np.pi * np.cumsum(f_inst) / fs
    # tacho: narrow pulse each rev
    tach = ((phase % (2 * np.pi)) < (2 * np.pi * 0.02)).astype(float)
    x = noise * rng.standard_normal(n)
    for order, amp, locked, slip in tones:
        if locked:
            x = x + amp * np.cos(order * phase + rng.uniform(0, 2 * np.pi))
        else:
            f = order * f0 * (1 + slip)
            x = x + amp * np.cos(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return tach, x, phase


def _pipeline(tach, x, revs=5):
    phase, meta = T.phase_from_tacho(tach, FS)
    xa = S.angular_resample(x, phase, spr=1024)
    Z, orders = S.block_spectra(xa, 1024, revs)
    coh, inc, ratio = S.coherent_split(Z)
    return Z, orders, coh, inc, ratio, meta


def bin_of(orders, o):
    return int(np.argmin(np.abs(orders - o)))


def test_T1_angular_resampling_accuracy():
    tach, x, _ = synth(tones=[(1.0, 1.0, True, 0)], noise=0.0)
    _, orders, coh, _, _, _ = _pipeline(tach, x)
    b = bin_of(orders, 1.0)
    assert abs(coh[b] - 1.0) < 0.01, f"1x coherent amp {coh[b]:.4f}, want 1.0 +/- 1%"


def test_T2_sqrtN_gain_law():
    # locked 0.4x tone at -20 dB vs noise; SNR in coherent spectrum must
    # scale ~ sqrt(N_blocks): floor drops 1/sqrt(N), tone amplitude constant.
    tach, x, _ = synth(dur=12.0, tones=[(0.4, 0.05, True, 0)], noise=0.5, seed=1)
    phase, _ = T.phase_from_tacho(tach, FS)
    xa = S.angular_resample(x, phase, 1024)
    snrs, Ns = [], [4, 9, 16, 36, 64]
    for N in Ns:
        Z, orders = S.block_spectra(xa[: N * 5 * 1024], 1024, 5)
        coh, _, _ = S.coherent_split(Z)
        b = bin_of(orders, 0.4)
        floor = np.median(np.delete(coh[1:200], b - 1))
        snrs.append(coh[b] / floor)
    slope = np.polyfit(np.log(Ns), np.log(snrs), 1)[0]
    assert 0.35 < slope < 0.65, f"gain exponent {slope:.2f}, theory 0.5"


def test_T3_unlocked_tone_low_ratio():
    tach, x, _ = synth(tones=[(3.585, 0.4, False, 0.015)], noise=0.3, seed=2)
    _, orders, coh, inc, ratio, _ = _pipeline(tach, x)
    b = bin_of(orders, 3.6)
    assert inc[b] > 5 * np.median(inc[1:200]), "unlocked peak missing in incoherent"
    assert ratio[b] < 0.25, f"unlocked ratio {ratio[b]:.2f}, want < 0.25"


def test_T4_cf_snap():
    from fractions import Fraction
    assert S.cf_snap(0.4031, tol=0.01) == Fraction(2, 5)
    assert S.cf_snap(0.503, tol=0.01) == Fraction(1, 2)
    assert S.cf_snap(3.585, tol=0.01) is None
    assert S.cf_snap(2.001, tol=0.01) == Fraction(2, 1)


def test_T5_locked_and_unlocked_discrimination():
    tach, x, _ = synth(dur=10.0, noise=0.3, seed=3,
                       tones=[(3.4, 0.3, True, 0), (3.6, 0.3, False, 0.015)])
    _, orders, coh, inc, ratio, _ = _pipeline(tach, x)
    assert ratio[bin_of(orders, 3.4)] > 0.8
    assert ratio[bin_of(orders, 3.6)] < 0.25


def test_T6_no_tacho_variants_recover_locked_tone():
    # onex + comb variants: locked 2x tone must stay coherent even though
    # phase comes from the signal itself (via its own 1x).
    tach, x, true_phase = synth(f0=30.0, dur=8.0, noise=0.3, seed=4,
                                tones=[(1.0, 0.6, True, 0), (2.0, 0.3, True, 0),
                                       (3.585, 0.3, False, 0.015)])
    run = np.zeros((len(x), 8)); run[:, 0] = tach; run[:, 2] = x
    for variant in ("onex", "comb"):
        phase, meta = T.get_phase(variant, run, FS, f_nom=30.0, ref_channel=2)
        xa = S.angular_resample(x, phase, 1024)
        Z, orders = S.block_spectra(xa, 1024, 5)
        _, _, ratio = S.coherent_split(Z)
        r2 = ratio[bin_of(orders, 2.0)]
        rb = ratio[bin_of(orders, 3.6)]
        assert r2 > 0.7, f"{variant}: 2x ratio {r2:.2f}"
        assert rb < 0.35, f"{variant}: bearing-like ratio {rb:.2f}"


def test_T7_nominal_variant_degrades_with_wander():
    # constant-speed assumption: 2% wander must smear a locked tone
    # (this is the caveat H3-extra quantifies on real data)
    tach, x, _ = synth(f0=30.0, dur=8.0, wander=0.02, noise=0.1, seed=5,
                       tones=[(2.0, 0.5, True, 0)])
    run = np.zeros((len(x), 8)); run[:, 0] = tach; run[:, 2] = x
    phase_n, _ = T.get_phase("nominal", run, FS, 30.0, ref_channel=2)
    phase_t, _ = T.get_phase("tacho", run, FS, 30.0)
    r = {}
    for nm, ph in (("nominal", phase_n), ("tacho", phase_t)):
        xa = S.angular_resample(x, ph, 1024)
        Z, orders = S.block_spectra(xa, 1024, 5)
        coh, _, _ = S.coherent_split(Z)
        r[nm] = coh[bin_of(orders, 2.0)]
    assert r["nominal"] < 0.6 * r["tacho"], (
        f"nominal {r['nominal']:.3f} vs tacho {r['tacho']:.3f}: "
        "expected clear degradation under 2% wander")


def test_T8_comb_reference_capture_guard():
    # Regression: a DOMINANT slipping tone (bearing-like, 2.998x at 1.5%
    # slip, stronger than the 1x) must NOT hijack the comb phase reference
    # and thereby appear coherent. Found via smoke test; fixed with
    # log-compressed scoring + f_nom prior + lowest-prominent-harmonic
    # refinement.
    tach, x, _ = synth(f0=30.0, dur=8.0, noise=0.2, seed=6,
                       tones=[(1.0, 0.5, True, 0), (2.998, 0.9, False, 0.015)])
    run = np.zeros((len(x), 8)); run[:, 0] = tach; run[:, 2] = x
    phase, meta = T.get_phase("comb", run, FS, f_nom=30.0, ref_channel=2)
    xa = S.angular_resample(x, phase, 1024)
    Z, orders = S.block_spectra(xa, 1024, 5)
    _, _, ratio = S.coherent_split(Z)
    assert ratio[bin_of(orders, 3.0)] < 0.35, (
        f"comb captured by bearing tone: ratio@3.0 = "
        f"{ratio[bin_of(orders, 3.0)]:.2f}")
    assert ratio[bin_of(orders, 1.0)] > 0.7, "comb lost the true 1x"
