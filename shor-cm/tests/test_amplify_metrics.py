"""Gate tests for H4 primitives and the loop machinery (T9-T13)."""
import numpy as np
from shorcm import amplify as AM
from shorcm import metrics as M

RNG = np.random.default_rng(7)


def test_T9_slip_scan_recovers_slip():
    # angular-domain signal: bearing tone at kinematic order * (1+slip)
    spr, revs, slip_true = 256, 200, 0.012
    n = spr * revs
    phi = 2 * np.pi * np.arange(n) / spr
    x = 0.4 * np.cos(2.998 * (1 + slip_true) * phi + 1.3) \
        + 0.6 * RNG.standard_normal(n)
    r = AM.slip_scan(x, spr, 2.998, span=0.03)
    assert abs(r["slip"] - slip_true) < 1e-3, r["slip"]
    assert r["sharpness"] > 5


def test_T10_ppa_amplifies_drifting_tone():
    # per-block complex values: amplitude 1, phase random-walk step 0.8 rad
    # (wanders over the full circle across 60 blocks, so naive coherent
    # averaging collapses while block-to-block phase stays predictable)
    rng = np.random.default_rng(7)
    N = 60
    theta = np.cumsum(rng.normal(0, 0.8, N))
    z = np.exp(1j * theta) + 0.5 * (rng.standard_normal(N)
                                    + 1j * rng.standard_normal(N))
    zs, obs, mu, sd = AM.ppa_zscore(z)
    naive = np.abs(z.mean())
    assert zs > 5, f"PPA z {zs:.1f}"
    assert obs > 1.5 * naive, "PPA should beat naive coherent mean on drift"


def test_T11_ppa_no_false_gain_on_noise():
    rng = np.random.default_rng(11)
    z = rng.standard_normal(60) + 1j * rng.standard_normal(60)
    zs, *_ = AM.ppa_zscore(z)
    assert abs(zs) < 3.5, f"PPA false positive on noise, z {zs:.1f}"


def test_T12_vault_is_deterministic_and_sized():
    ids = [f"run_{i}" for i in range(5000)]
    m1 = M.vault_mask(ids)
    m2 = M.vault_mask(ids)
    assert (m1 == m2).all()
    assert 0.17 < m1.mean() < 0.23


def test_T13_promotion_gate_tightens_with_attempts():
    # deltas at bootstrap-of-the-mean scale: true gain 0.01, spread 0.004
    # (i.e. clearly real at alpha=0.05, not at alpha=0.05/200)
    deltas = np.random.default_rng(13).normal(0.01, 0.004, 4000)
    early = M.promotion_gate(deltas, True, attempts=1)
    late = M.promotion_gate(deltas, True, attempts=200)
    assert early["promote"] is True
    assert late["promote"] is False
    assert M.promotion_gate(deltas, False, attempts=1)["promote"] is False
