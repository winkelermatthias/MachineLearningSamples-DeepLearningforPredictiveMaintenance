"""Vectorized daily backtest engine with strict no-lookahead conventions.

Conventions
-----------
- A strategy produces *target weights* ``w_t`` using information available up
  to and including the close of day ``t``.
- The engine applies ``w_{t-1}`` to the return of day ``t`` (``w.shift(1)``),
  i.e. a decision made at today's close earns tomorrow's return. This one-day
  implementation lag is what most naive backtests forget, and omitting it is
  the single most common source of fake alpha in daily-frequency studies.
- Transaction costs are charged on turnover: ``cost_t = tc * sum_i |w_{t,i} -
  w_{t-1,i}|`` (one-way cost per unit of traded notional).

Known simplification: turnover is computed from changes in *target* weights,
ignoring the intra-period drift of held weights with prices. This slightly
understates turnover for slow-rebalancing portfolios (e.g. a constant-target
equal-weight book shows ~zero turnover) and is the standard vectorized
approximation; the cost-sensitivity table bounds its impact.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    name: str
    gross: pd.Series      # daily gross returns
    net: pd.Series        # daily net-of-cost returns
    turnover: pd.Series   # daily one-way turnover (sum |dw|)
    weights: pd.DataFrame


def run_backtest(weights: pd.DataFrame, returns: pd.DataFrame,
                 tc_bps: float = 5.0, name: str = "strategy") -> BacktestResult:
    """Backtest target weights against asset returns.

    Parameters
    ----------
    weights : target weights decided at the close of each date (rows may be a
        subset of ``returns.index``; they are forward-filled between rebalances).
    returns : simple daily asset returns.
    tc_bps : one-way transaction cost in basis points of traded notional.
    """
    w = weights.reindex(returns.index).ffill().fillna(0.0)
    pos = w.shift(1).fillna(0.0)                # held during day t
    gross = (pos * returns).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()    # initial position build
    net = gross - (tc_bps / 1e4) * turnover
    return BacktestResult(name=name, gross=gross, net=net,
                          turnover=turnover, weights=w)


def equity_curve(ret: pd.Series) -> pd.Series:
    return (1.0 + ret).cumprod()


def drawdown(ret: pd.Series) -> pd.Series:
    eq = equity_curve(ret)
    return eq / eq.cummax() - 1.0


def summary_stats(ret: pd.Series, turnover: pd.Series | None = None) -> dict:
    """Standard performance metrics on a daily return series."""
    ret = ret.dropna()
    n = len(ret)
    if n == 0 or ret.std() == 0:
        return {k: np.nan for k in ["CAGR", "AnnVol", "Sharpe", "Sortino",
                                    "MaxDD", "Calmar", "Skew", "Kurtosis",
                                    "AnnTurnover"]}
    years = n / TRADING_DAYS
    total = float((1 + ret).prod())
    cagr = total ** (1 / years) - 1 if total > 0 else -1.0
    vol = float(ret.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(ret.mean() / ret.std() * np.sqrt(TRADING_DAYS))
    downside = ret[ret < 0].std()
    sortino = float(ret.mean() / downside * np.sqrt(TRADING_DAYS)) if downside and downside > 0 else np.nan
    mdd = float(drawdown(ret).min())
    return {
        "CAGR": cagr,
        "AnnVol": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": mdd,
        "Calmar": cagr / abs(mdd) if mdd else np.nan,
        "Skew": float(ret.skew()),
        "Kurtosis": float(ret.kurtosis()),
        "AnnTurnover": float(turnover.reindex(ret.index).sum() / years) if turnover is not None else np.nan,
    }
