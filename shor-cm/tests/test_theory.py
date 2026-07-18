"""SYNTHVAL gate tests: Monte Carlo must match closed-form theory inside
pre-registered tolerance bands. A divergence is a finding, not noise."""
import numpy as np
from shorcm import theory as TH
from shorcm import spectra as S


def blocks(A, sigma, N, s, n_mc, rng):
    """(n_mc, N) complex block values for tone amp A, noise sigma,
    random-walk step s (0 = locked)."""
    th0 = rng.uniform(0, 2 * np.pi, (n_mc, 1))
    steps = rng.normal(0, s, (n_mc, N)).cumsum(axis=1) if s > 0 else 0.0
    z = A * np.exp(1j * (th0 + steps))
    z = z + sigma * (rng.standard_normal((n_mc, N))
                     + 1j * rng.standard_normal((n_mc, N)))
    return z


def test_T14_ratio_noise_null():
    rng = np.random.default_rng(21)
    for N in (8, 16, 64):
        z = blocks(0.0, 1.0, N, 0, 20000, rng)
        r = np.abs(z.mean(1)) / np.abs(z).mean(1)
        assert abs(r.mean() - TH.ratio_noise(N)) < 0.05 * TH.ratio_noise(N) + 0.01


def test_T15_ratio_locked_rice():
    rng = np.random.default_rng(22)
    for A, sig, N in [(1.0, 1.0, 16), (0.5, 1.0, 32), (2.0, 0.7, 8)]:
        z = blocks(A, sig, N, 0, 20000, rng)
        r = (np.abs(z.mean(1)) / np.abs(z).mean(1)).mean()
        assert abs(r - TH.ratio_locked(A, sig, N)) < 0.03, (A, sig, N, r)


def test_T16_drift_factor_and_ratio():
    rng = np.random.default_rng(23)
    for s, N in [(0.4, 32), (0.8, 32), (0.8, 64)]:
        z = blocks(1.0, 0.8, N, s, 20000, rng)
        r = (np.abs(z.mean(1)) / np.abs(z).mean(1)).mean()
        assert abs(r - TH.ratio_drift(1.0, 0.8, N, s)) < 0.04, (s, N, r)


def test_T17_cf_false_snap_exact_vs_mc():
    rng = np.random.default_rng(24)
    tol, qmax, lo, hi = 0.01, 8, 0.2, 10.0
    o = rng.uniform(lo, hi, 40000)
    hits = np.fromiter((S.cf_snap(v, tol=tol, qmax=qmax) is not None
                        for v in o), bool).mean()
    ex = TH.cf_false_snap_rate(tol, qmax, lo, hi)
    assert abs(hits - ex) < 0.01, (hits, ex)
    # the finding itself, pinned: false-snap risk per random peak is LARGE
    assert 0.25 < ex < 0.45, f"expected ~1/3 false-snap zone, got {ex:.3f}"


def test_T18_decoherence_gaussian_in_order():
    rng = np.random.default_rng(25)
    sig_phi, N, n_mc = 0.15, 64, 4000
    for k in (1, 3, 6):
        eps = rng.normal(0, sig_phi, (n_mc, N))
        z = np.exp(1j * k * eps)
        loss = np.abs(z.mean(1)).mean()
        assert abs(loss - TH.decoherence_factor(k, sig_phi)) < 0.02, (k, loss)


def test_T19_coh_mean_decohered_limits_and_mc():
    # limit 1: sigma_phi = 0 reduces exactly to the locked Rice mean
    for A, sig, N in [(1.0, 1.0, 16), (2.0, 0.5, 64)]:
        assert abs(TH.coh_mean_decohered(A, sig, N, 3, 0.0)
                   - TH.coh_mean_locked(A, sig, N)) < 1e-12
    # limit 2: fully decohered high order -> Rayleigh floor of the
    # tone-scatter + noise mixture, sqrt(pi/2 * (A^2/2 + sig^2) / N)
    A, sig, N = 1.0, 0.5, 64
    fl = TH.coh_mean_decohered(A, sig, N, 40, 1.0)
    assert abs(fl - np.sqrt(np.pi / 2 * (A**2 / 2 + sig**2) / N)) < 0.01
    # MC agreement across the interesting middle
    rng = np.random.default_rng(26)
    n_mc = 6000
    for k, sp in [(1, 0.15), (3, 0.15), (6, 0.15), (2, 0.4)]:
        eps = rng.normal(0, sp, (n_mc, N))
        z = A * np.exp(1j * k * eps) + sig * (
            rng.standard_normal((n_mc, N)) + 1j * rng.standard_normal((n_mc, N)))
        mc = np.abs(z.mean(1)).mean()
        th = TH.coh_mean_decohered(A, sig, N, k, sp)
        assert abs(mc - th) < 0.02, (k, sp, mc, th)
