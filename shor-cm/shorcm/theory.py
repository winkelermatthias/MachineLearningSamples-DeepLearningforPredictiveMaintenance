"""Closed-form theory for the synthetic validation program (SYNTHVAL).

Every empirical claim in the pipeline has a derivable prediction here.
Monte Carlo exists to VERIFY these, and divergence beyond tolerance is a
finding, never a thing to average away. Model: at one order bin, block
spectra are z_k = A e^{i theta_k} + w_k, w_k complex Gaussian with
per-quadrature std sigma, k = 1..N.

Cases:
  locked  : theta_k = const
  drifting: theta_k random walk, step std s per block
  noise   : A = 0
"""
import numpy as np
from fractions import Fraction
from scipy.special import ive  # exponentially scaled Bessel I

SQRT_PI_2 = np.sqrt(np.pi / 2)


def rice_mean(nu, sigma):
    """E|nu + w|, w complex Gaussian per-quadrature std sigma (Rice mean).
    Uses exponentially scaled Bessels for numerical stability."""
    if sigma == 0:
        return abs(nu)
    x = nu**2 / (4 * sigma**2)
    # L_{1/2}(-2x) = e^{-x} [(1+2x) I0(x) + 2x I1(x)]  with scaled ive
    l_half = (1 + 2 * x) * ive(0, x) + 2 * x * ive(1, x)
    return sigma * SQRT_PI_2 * l_half


def coh_mean_locked(A, sigma, N):
    """E|mean z_k| for a locked tone: Rice with noise std sigma/sqrt(N)."""
    return rice_mean(A, sigma / np.sqrt(N))


def inc_mean(A, sigma):
    """E mean|z_k|: Rice with full per-block noise."""
    return rice_mean(A, sigma)


def ratio_locked(A, sigma, N):
    """Ratio of means (close to mean of ratio for N >= 8)."""
    return coh_mean_locked(A, sigma, N) / inc_mean(A, sigma)


def ratio_noise(N):
    """Pure noise: E coh / E inc = 1/sqrt(N) exactly (both Rayleigh means,
    sigma cancels)."""
    return 1.0 / np.sqrt(N)


def drift_coherent_factor(s, N):
    """|mean e^{i theta_k}| for a random walk with step std s:
    E|sum|^2 = sum_{j,k} rho^{|j-k|}, rho = exp(-s^2/2). Closed geometric
    sum, exact for finite N."""
    rho = np.exp(-s**2 / 2)
    if rho >= 1 - 1e-12:
        return 1.0
    S = N + 2 * (rho * (N - 1) - rho**2 * (1 - rho**(N - 1)) / (1 - rho)) \
        / (1 - rho)
    return float(np.sqrt(max(S, 0)) / N)


def ratio_drift(A, sigma, N, s):
    """Drifting tone: coherent part shrinks by drift_coherent_factor;
    noise part as usual. Approximation: tone and noise powers add."""
    eff = A * drift_coherent_factor(s, N)
    coh = np.sqrt(eff**2 + (np.pi / 2) * (sigma / np.sqrt(N))**2)
    return float(coh / inc_mean(A, sigma))


def decoherence_factor(order, sigma_phi):
    """Phase-reference error epsilon per block, std sigma_phi rad at the
    SHAFT: at order k the tone is multiplied by e^{i k epsilon}, so the
    coherent amplitude shrinks by exp(-k^2 sigma_phi^2 / 2). Distinctive,
    testable prediction: decoherence is GAUSSIAN IN ORDER. High orders die
    first under a bad phase reference (the nominal variant's disease)."""
    return np.exp(-(order**2) * (sigma_phi**2) / 2)


def cf_false_snap_rate(tol, qmax, o_lo, o_hi):
    """EXACT probability that a uniform random order in [o_lo, o_hi]
    snaps to some p/q, q <= qmax, within tol: measure of the union of
    acceptance windows (windows overlap near dense Farey regions, so
    counting fractions x 2 tol overestimates; we merge intervals)."""
    pts = set()
    for q in range(1, qmax + 1):
        p0 = int(np.floor(o_lo * q))
        p1 = int(np.ceil(o_hi * q))
        for p in range(max(p0, 1), p1 + 1):
            v = p / q
            if o_lo - tol <= v <= o_hi + tol:
                pts.add(v)
    iv = sorted((v - tol, v + tol) for v in pts)
    merged, total = [], 0.0
    for a, b in iv:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    for a, b in merged:
        total += min(b, o_hi) - max(a, o_lo)
    return max(0.0, min(1.0, total / (o_hi - o_lo)))


def coh_mean_decohered(A, sigma, N, k, sigma_phi):
    """E|mean z_k| for a locked tone observed at order k through a phase
    reference with per-block error std sigma_phi rad at the shaft:
    z_j = A e^{i k eps_j} + w_j, eps_j iid N(0, sigma_phi^2).
    The tone mean shrinks to A D with D = exp(-k^2 sigma_phi^2 / 2); the
    scatter of e^{i k eps} about its mean adds A^2 (1 - D^2) / 2 per
    quadrature on top of the noise, both divided by N after averaging.
    Semi-analytic: Rice mean with nu = A D,
    s^2 = (A^2 (1 - D^2) / 2 + sigma^2) / N.
    Limits: sigma_phi = 0 -> coh_mean_locked; k sigma_phi -> inf ->
    Rayleigh floor sqrt(pi/2) * sqrt((A^2/2 + sigma^2)/N)."""
    D = decoherence_factor(k, sigma_phi)
    s2 = (A**2 * (1 - D**2) / 2 + sigma**2) / N
    return rice_mean(A * D, np.sqrt(s2))


def rice_var(nu, sigma):
    """Var|nu + w| = nu^2 + 2 sigma^2 - rice_mean^2."""
    return max(nu**2 + 2 * sigma**2 - rice_mean(nu, sigma)**2, 1e-15)


def _ratio_conditional(A, sigma, N, C):
    """Ratio moments conditional on walk resultant C (locked: C = 1).
    General delta method WITH correlation: numerator Rice(A C, sigma/rtN),
    denominator mean of N Rices, correlation rho = C * w where
    w = A^2/(A^2+sigma^2) gates whether the tone dominates per-block
    magnitudes (cancellation regime) or not (plain delta regime). Limits:
    high per-block SNR, C=1 -> full cancellation + second-order floor;
    low per-block SNR -> plain delta with Rice(A C, sigma/rtN) numerator,
    which keeps the post-averaging gain that large N buys."""
    sN = sigma / np.sqrt(N)
    m_c = rice_mean(A * C, sN)
    v_c = rice_var(A * C, sN)
    m_i = rice_mean(A, sigma)
    v_i = rice_var(A, sigma) / N
    w = A**2 / (A**2 + sigma**2)
    rho = C * w
    m = m_c / m_i
    v = (v_c + (m_c / m_i)**2 * v_i
         - 2 * rho * (m_c / m_i) * np.sqrt(v_c * v_i)) / m_i**2
    v += w * C**2 * sigma**4 / (2 * max(A, 1e-9)**4) * (1 / N - 1 / N**2)
    return m, max(v, 1e-12)


def _ratio_moments_locked(A, sigma, N):
    return _ratio_conditional(A, sigma, N, 1.0)


_WALK_DRAWS = {}


def _walk_coherence_draws(s, N, n=2000, seed=4242):
    """Deterministic quadrature draws of C = |mean e^{i theta_k}| for the
    analytically intractable random-walk latent. Fixed seed, independent
    of any experiment seed: semi-analytic theory, pre-registrable."""
    key = (round(s, 6), N, n, seed)
    if key not in _WALK_DRAWS:
        rng = np.random.default_rng(seed)
        th = rng.normal(0, s, (n, N)).cumsum(1)
        _WALK_DRAWS[key] = np.abs(np.exp(1j * th).mean(1))
    return _WALK_DRAWS[key]


def ratio_auc_locked_vs_drift(A, sigma, N, s, n_mc=0):
    """Exact-conditional semi-analytic AUC. Conditional on walk resultant
    C, and treating the denominator as concentrated (Var ~ 1/N, verified
    small), the ratio is a scaled Rice: locked ~ Rice(A, sigma/rtN),
    drift ~ Rice(A C, sigma/rtN), same scale, scale cancels in the
    comparison. AUC(C) = P(R_locked > R_drift) computed by deterministic
    1D quadrature of pdf_drift * SF_locked; final AUC = mean over the
    fixed walk-quadrature draws of C. No normal approximation anywhere.
    History: two moment-matched normal models diverged from MC at small N
    (skewed mixture) and at low SNR x large N (regime blend keyed on
    per-block instead of post-averaging SNR); this replaced them."""
    from scipy.stats import rice as _rice
    sN = sigma / np.sqrt(N)
    C = _walk_coherence_draws(s, N)
    hi = (A + 6 * sN)
    grid = np.linspace(0, hi, 900)
    sf1 = _rice.sf(grid, A / sN, scale=sN)          # locked survival
    pdf0 = _rice.pdf(grid[None, :], (A * C[:, None]) / sN, scale=sN)
    auc_c = np.trapezoid(pdf0 * sf1[None, :], grid, axis=1)
    return float(np.mean(auc_c))
