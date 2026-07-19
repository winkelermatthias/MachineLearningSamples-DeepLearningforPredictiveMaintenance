"""Strategy families from the empirical asset-pricing literature.

Every function returns a DataFrame of *target weights* indexed by decision
date (close of that day). Only information up to the decision date is used —
all rolling statistics end at the decision date, never after it. The engine
then applies the weights with a one-day lag.

Implemented families and their canonical references:

- Cross-sectional momentum        — Jegadeesh & Titman (1993)
- Time-series momentum            — Moskowitz, Ooi & Pedersen (2012)
- Short-term reversal             — Jegadeesh (1990), Lehmann (1990)
- Moving-average trend following  — Brock, Lakonishok & LeBaron (1992)
- Volatility targeting overlay    — Moreira & Muir (2017)
- Inverse-volatility (risk parity)— Asness, Frazzini & Pedersen (2012)
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each month present in the index."""
    s = pd.Series(index=index, data=np.arange(len(index)))
    return index[s.groupby([index.year, index.month]).max().values]


def equal_weight(returns: pd.DataFrame) -> pd.DataFrame:
    """1/N portfolio, rebalanced monthly (benchmark)."""
    rb = _month_ends(returns.index)
    n = returns.shape[1]
    return pd.DataFrame(1.0 / n, index=rb, columns=returns.columns)


def xs_momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21,
                top_frac: float = 0.25, long_short: bool = False) -> pd.DataFrame:
    """Cross-sectional (relative) momentum, monthly rebalance.

    Rank assets by their return from t-lookback to t-skip (skipping the most
    recent `skip` days to avoid the short-term reversal effect). Long the top
    fraction; optionally short the bottom fraction (each leg gross 0.5).
    """
    mom = prices.shift(skip) / prices.shift(lookback) - 1.0
    rb = _month_ends(prices.index)
    mom = mom.loc[rb].dropna(how="all")
    k = max(1, int(round(top_frac * prices.shape[1])))
    ranks = mom.rank(axis=1, ascending=False)
    long_w = ranks.le(k).astype(float)
    long_w = long_w.div(long_w.sum(axis=1), axis=0)
    if not long_short:
        return long_w
    short_ranks = mom.rank(axis=1, ascending=True)
    short_w = short_ranks.le(k).astype(float)
    short_w = short_w.div(short_w.sum(axis=1), axis=0)
    return 0.5 * long_w - 0.5 * short_w


def ts_momentum(prices: pd.DataFrame, lookback: int = 252,
                vol_lookback: int = 60) -> pd.DataFrame:
    """Time-series momentum: sign of each asset's own past return.

    Positions are inverse-volatility scaled (so risk is spread across assets)
    and normalized to gross exposure 1. Weekly rebalance.
    """
    rets = prices.pct_change()
    sig = np.sign(prices / prices.shift(lookback) - 1.0)
    vol = rets.rolling(vol_lookback).std() * np.sqrt(TRADING_DAYS)
    raw = sig / vol.clip(lower=0.05)
    rb = prices.index[::5]  # every 5th trading day
    raw = raw.loc[rb].dropna(how="all")
    gross = raw.abs().sum(axis=1)
    return raw.div(gross.where(gross > 0, 1.0), axis=0)


def short_term_reversal(prices: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Dollar-neutral short-term reversal, daily rebalance.

    Weight is minus the cross-sectionally demeaned past `lookback`-day return,
    scaled to gross exposure 1. Deliberately high-turnover: it demonstrates
    how transaction costs kill paper alpha.
    """
    r = prices.pct_change(lookback)
    demeaned = -(r.sub(r.mean(axis=1), axis=0))
    gross = demeaned.abs().sum(axis=1)
    return demeaned.div(gross.where(gross > 0, np.nan), axis=0).dropna(how="all")


def ma_crossover(prices: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """Long-only trend following: hold 1/N in each asset whose fast MA is
    above its slow MA, else cash. Weekly rebalance."""
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    sig = (fast_ma > slow_ma).astype(float)
    rb = prices.index[::5]
    sig = sig.loc[rb]
    sig = sig[slow_ma.loc[rb].notna().any(axis=1)]
    return sig / prices.shape[1]


def inverse_vol(returns: pd.DataFrame, vol_lookback: int = 60) -> pd.DataFrame:
    """Naive risk parity: monthly rebalance, weights proportional to 1/vol."""
    vol = returns.rolling(vol_lookback).std()
    rb = _month_ends(returns.index)
    iv = (1.0 / vol.loc[rb]).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return iv.div(iv.sum(axis=1), axis=0)


def vol_target_overlay(weights: pd.DataFrame, returns: pd.DataFrame,
                       target_vol: float = 0.10, vol_lookback: int = 20,
                       max_leverage: float = 1.0) -> pd.DataFrame:
    """Volatility-targeting overlay (Moreira & Muir 2017).

    Scales the base portfolio by target_vol / trailing realized vol, capped at
    ``max_leverage``. With max_leverage=1 this only de-risks (no borrowing),
    which is implementable by anyone.
    """
    w = weights.reindex(returns.index).ffill().fillna(0.0)
    port_ret = (w.shift(1) * returns).sum(axis=1)
    realized = port_ret.rolling(vol_lookback).std() * np.sqrt(TRADING_DAYS)
    scale = (target_vol / realized).clip(upper=max_leverage).fillna(0.0)
    return w.mul(scale, axis=0)
