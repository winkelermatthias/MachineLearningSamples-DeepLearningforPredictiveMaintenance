"""Data loading for the trading strategy study.

Data source: real daily prices bundled inside the `skfolio` PyPI package
(no network access required):

- ``load_sp500_dataset``: daily prices of 20 current S&P 500 constituents,
  1990-01-02 to 2022-12-28.
- ``load_sp500_index``: daily prices of the S&P 500 index (benchmark).

IMPORTANT CAVEAT — survivorship bias: the 20 stocks are *current* index
members that survived the whole 1990-2022 period. Long-only results on this
universe are biased upward versus a point-in-time universe. Long-short and
relative (cross-sectional) results are less affected but not immune.
"""

import pandas as pd
from skfolio.datasets import load_sp500_dataset, load_sp500_index

# Out-of-sample boundary: data before this date may be used for parameter
# selection; data from this date on is touched only for final evaluation.
OOS_START = "2010-01-01"


def load_prices() -> pd.DataFrame:
    """Daily prices of the 20-stock universe, sorted, forward-filled."""
    prices = load_sp500_dataset().sort_index()
    return prices.ffill().dropna()


def load_benchmark() -> pd.Series:
    """Daily S&P 500 index level as a Series."""
    idx = load_sp500_index().sort_index()
    return idx.iloc[:, 0].rename("SP500")


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns."""
    return prices.pct_change().iloc[1:]


def split_is_oos(obj, oos_start: str = OOS_START):
    """Split a Series/DataFrame into (in-sample, out-of-sample) parts."""
    ts = pd.Timestamp(oos_start)
    return obj.loc[obj.index < ts], obj.loc[obj.index >= ts]
