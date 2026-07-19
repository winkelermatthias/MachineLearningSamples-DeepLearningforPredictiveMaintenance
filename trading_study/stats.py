"""Statistical rigor for backtests: bootstrap Sharpe confidence intervals and
the Deflated Sharpe Ratio (multiple-testing correction).

References
----------
- Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting and Non-Normality".
- Politis & Romano (1994), moving block bootstrap.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252
EULER_MASCHERONI = 0.5772156649015329


def block_bootstrap_sharpe_ci(ret: pd.Series, n_boot: int = 2000,
                              block: int = 21, ci: float = 0.95,
                              seed: int = 42) -> tuple[float, float]:
    """Moving-block bootstrap CI for the annualized Sharpe ratio.

    Blocks preserve short-range autocorrelation that i.i.d. resampling would
    destroy (and which biases naive Sharpe standard errors).
    """
    x = ret.dropna().to_numpy()
    n = len(x)
    if n < 2 * block:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block, size=(n_boot, n_blocks))
    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        sample = np.concatenate([x[s:s + block] for s in starts[b]])[:n]
        sd = sample.std()
        sharpes[b] = sample.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else 0.0
    lo, hi = np.quantile(sharpes, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def deflated_sharpe_ratio(ret: pd.Series, n_trials: int,
                          trial_sr_std: float | None = None) -> dict:
    """Probability that the observed Sharpe is not a fluke of trying
    ``n_trials`` configurations.

    Computes the expected maximum Sharpe under the null of zero skill across
    ``n_trials`` independent trials (SR0), then evaluates the Probabilistic
    Sharpe Ratio of the observed (non-annualized, per-period) Sharpe against
    that benchmark, adjusting for skewness and kurtosis of returns.

    Returns dict with observed annualized SR, the annualized hurdle SR0, and
    DSR = P(true SR > SR0). DSR > 0.95 is the usual bar for "likely real".
    """
    x = ret.dropna()
    t = len(x)
    sr = float(x.mean() / x.std())  # per-period Sharpe
    skew = float(x.skew())
    kurt = float(x.kurtosis()) + 3.0  # scipy-style raw kurtosis

    # Cross-trial std of per-period Sharpe under the null. If not supplied,
    # use the estimator std of a zero-mean SR over t observations.
    if trial_sr_std is None:
        trial_sr_std = np.sqrt(1.0 / t)
    n = max(int(n_trials), 2)
    sr0 = trial_sr_std * ((1 - EULER_MASCHERONI) * norm.ppf(1 - 1.0 / n)
                          + EULER_MASCHERONI * norm.ppf(1 - 1.0 / (n * np.e)))
    denom = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2))
    z = (sr - sr0) * np.sqrt(t - 1) / denom
    return {
        "SR_ann": sr * np.sqrt(TRADING_DAYS),
        "SR0_ann_hurdle": float(sr0 * np.sqrt(TRADING_DAYS)),
        "n_trials": n,
        "DSR": float(norm.cdf(z)),
    }
