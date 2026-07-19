"""Run the full trading-strategy study.

Protocol
--------
1. Build every strategy configuration on the full sample (all signals are
   causal: statistics at date t use only data through t, and the engine adds a
   one-day execution lag).
2. Parameter selection uses ONLY the in-sample period (1990-2009), by net
   Sharpe. The out-of-sample period (2010-2022) is touched exactly once per
   family, for final evaluation.
3. A walk-forward track re-selects each family's parameters every ~2 years
   using an expanding window, mimicking what a live investor could have done.
4. Statistics: moving-block bootstrap CIs for Sharpe; Deflated Sharpe Ratio
   charging each family for the total number of configurations tried in the
   whole study.
5. Transaction-cost sensitivity at 0/5/10/20 bps one-way.

Usage: python3 run_study.py   (writes tables, figures and REPORT.md to results/)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import OOS_START, load_benchmark, load_prices, to_returns
from engine import BacktestResult, drawdown, equity_curve, run_backtest, summary_stats
from stats import block_bootstrap_sharpe_ci, deflated_sharpe_ratio
import strategies as st

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TC_BPS = 5.0
WF_START = "2000-01-01"
WF_STEP = 504  # refit every ~2 years


def build_grids(prices, returns):
    """family -> {config label -> weights DataFrame}."""
    grids = {}
    grids["EqualWeight"] = {"EW monthly": st.equal_weight(returns)}
    grids["XSMOM long-only"] = {
        f"lb={lb},skip={sk}": st.xs_momentum(prices, lb, sk, 0.25, False)
        for lb in (126, 252) for sk in (0, 21)}
    grids["XSMOM long-short"] = {
        f"lb={lb},skip={sk}": st.xs_momentum(prices, lb, sk, 0.25, True)
        for lb in (126, 252) for sk in (0, 21)}
    grids["TSMOM"] = {f"lb={lb}": st.ts_momentum(prices, lb)
                      for lb in (63, 126, 252)}
    grids["ShortTermReversal"] = {f"lb={lb}": st.short_term_reversal(prices, lb)
                                  for lb in (3, 5, 10)}
    grids["MA crossover"] = {f"{f}/{s}": st.ma_crossover(prices, f, s)
                             for f, s in ((10, 50), (20, 100), (50, 200))}
    grids["RiskParity"] = {f"vol_lb={v}": st.inverse_vol(returns, v)
                           for v in (30, 60, 120)}
    ew = st.equal_weight(returns)
    grids["VolTarget EW"] = {
        f"target={t:.0%}": st.vol_target_overlay(ew, returns, t)
        for t in (0.08, 0.10, 0.12)}
    return grids


def main():
    os.makedirs(RESULTS, exist_ok=True)
    prices = load_prices()
    returns = to_returns(prices)
    bench = load_benchmark().pct_change().reindex(returns.index).dropna()
    oos = pd.Timestamp(OOS_START)

    grids = build_grids(prices, returns)
    n_trials_total = sum(len(g) for g in grids.values())
    print(f"Universe: {prices.shape[1]} stocks, {prices.index[0]:%Y-%m-%d} to "
          f"{prices.index[-1]:%Y-%m-%d}; {n_trials_total} configurations total")

    # Backtest every configuration once on the full sample.
    results: dict[str, dict[str, BacktestResult]] = {}
    for fam, cfgs in grids.items():
        results[fam] = {lbl: run_backtest(w, returns, TC_BPS, f"{fam} [{lbl}]")
                        for lbl, w in cfgs.items()}

    def net_sharpe(res, lo=None, hi=None):
        r = res.net
        if lo is not None:
            r = r.loc[r.index >= lo]
        if hi is not None:
            r = r.loc[r.index < hi]
        if (r != 0).any():
            r = r.loc[r.ne(0).idxmax():]  # trim warm-up period before first position
        sd = r.std()
        return float(r.mean() / sd * np.sqrt(252)) if sd and sd > 0 else np.nan

    # ---- 1) In-sample selection -> single untouched OOS evaluation ----
    rows, picked = [], {}
    for fam, cfgs in results.items():
        is_sharpes = {lbl: net_sharpe(res, hi=oos) for lbl, res in cfgs.items()}
        best = max(is_sharpes, key=lambda k: (np.nan_to_num(is_sharpes[k], nan=-9)))
        picked[fam] = best
        res = cfgs[best]
        net_oos = res.net.loc[res.net.index >= oos]
        stats = summary_stats(net_oos, res.turnover)
        ci = block_bootstrap_sharpe_ci(net_oos)
        dsr = deflated_sharpe_ratio(net_oos, n_trials=n_trials_total)
        rows.append({
            "Family": fam, "Config (chosen in-sample)": best,
            "IS Sharpe": is_sharpes[best], **stats,
            "Sharpe 95% CI lo": ci[0], "Sharpe 95% CI hi": ci[1],
            "DSR": dsr["DSR"], "SR hurdle (ann)": dsr["SR0_ann_hurdle"],
        })
    bench_oos = bench.loc[bench.index >= oos]
    b_stats = summary_stats(bench_oos)
    b_ci = block_bootstrap_sharpe_ci(bench_oos)
    rows.append({"Family": "S&P 500 index (buy&hold)",
                 "Config (chosen in-sample)": "-",
                 "IS Sharpe": net_sharpe_series(bench.loc[bench.index < oos]),
                 **b_stats, "Sharpe 95% CI lo": b_ci[0],
                 "Sharpe 95% CI hi": b_ci[1], "DSR": np.nan,
                 "SR hurdle (ann)": np.nan})
    summary = pd.DataFrame(rows).set_index("Family")
    summary.to_csv(os.path.join(RESULTS, "summary_oos.csv"))
    print("\nOut-of-sample (2010-2022) summary:\n",
          summary[["Config (chosen in-sample)", "IS Sharpe", "Sharpe", "CAGR",
                   "MaxDD", "DSR"]].round(3).to_string())

    # ---- 2) Walk-forward selection track ----
    wf_returns = {}
    idx = returns.index
    start_pos = idx.searchsorted(pd.Timestamp(WF_START))
    for fam, cfgs in results.items():
        pieces = []
        for p in range(start_pos, len(idx), WF_STEP):
            train_hi = idx[p]
            sharpes = {lbl: net_sharpe(res, hi=train_hi)
                       for lbl, res in cfgs.items()}
            best = max(sharpes, key=lambda k: np.nan_to_num(sharpes[k], nan=-9))
            seg = results[fam][best].net.iloc[p:p + WF_STEP]
            pieces.append(seg)
        wf_returns[fam] = pd.concat(pieces)
    wf_df = pd.DataFrame(wf_returns)
    wf_stats = wf_df.apply(lambda c: pd.Series(summary_stats(c)))
    wf_stats.T.to_csv(os.path.join(RESULTS, "summary_walkforward.csv"))

    # ---- 3) Cost sensitivity on the IS-picked configs ----
    cost_rows = {}
    for fam in results:
        w = grids[fam][picked[fam]]
        cost_rows[fam] = {
            bps: net_sharpe_series(
                run_backtest(w, returns, bps).net.loc[returns.index >= oos])
            for bps in (0.0, 5.0, 10.0, 20.0)}
    cost_df = pd.DataFrame(cost_rows).T
    cost_df.columns = [f"{int(c)} bps" for c in cost_df.columns]
    cost_df.to_csv(os.path.join(RESULTS, "cost_sensitivity.csv"))

    make_plots(results, picked, wf_df, cost_df, bench, oos)
    write_report(summary, wf_stats.T, cost_df, picked, n_trials_total,
                 prices, returns)
    print(f"\nDone. Outputs in {RESULTS}/")


def net_sharpe_series(r: pd.Series) -> float:
    r = r.dropna()
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(252)) if sd and sd > 0 else np.nan


def make_plots(results, picked, wf_df, cost_df, bench, oos):
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9})

    # OOS equity curves (net of 5 bps)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for fam, cfgs in results.items():
        r = cfgs[picked[fam]].net
        r = r.loc[r.index >= oos]
        ax.plot(equity_curve(r), label=f"{fam} [{picked[fam]}]", lw=1.2)
    ax.plot(equity_curve(bench.loc[bench.index >= oos]), "k--", lw=1.5,
            label="S&P 500 buy&hold")
    ax.set_yscale("log")
    ax.set_title("Out-of-sample growth of $1 (2010-2022, net of 5 bps costs, "
                 "params chosen in-sample only)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "equity_curves_oos.png"))
    plt.close(fig)

    # Walk-forward equity curves
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for fam in wf_df:
        ax.plot(equity_curve(wf_df[fam].dropna()), label=fam, lw=1.2)
    b = bench.loc[bench.index >= wf_df.index[0]]
    ax.plot(equity_curve(b), "k--", lw=1.5, label="S&P 500 buy&hold")
    ax.set_yscale("log")
    ax.set_title("Walk-forward growth of $1 (2000-2022): parameters re-selected "
                 "every 2 years on expanding window, net of 5 bps")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "walkforward_equity.png"))
    plt.close(fig)

    # OOS drawdowns
    fig, ax = plt.subplots(figsize=(10, 4))
    for fam, cfgs in results.items():
        r = cfgs[picked[fam]].net
        ax.plot(drawdown(r.loc[r.index >= oos]), lw=1.0, label=fam)
    ax.set_title("Out-of-sample drawdowns (net)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "drawdowns_oos.png"))
    plt.close(fig)

    # Cost sensitivity
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for fam in cost_df.index:
        ax.plot([0, 5, 10, 20], cost_df.loc[fam].values, marker="o", label=fam)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("one-way transaction cost (bps)")
    ax.set_ylabel("OOS annualized Sharpe (net)")
    ax.set_title("Transaction costs decide who survives")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "cost_sensitivity.png"))
    plt.close(fig)


def write_report(summary, wf_stats, cost_df, picked, n_trials, prices, returns):
    def tbl(df):
        return df.round(3).to_markdown()

    lines = [
        "# Trading strategy study: what survives a correct backtest",
        "",
        f"*Generated by `run_study.py`. Universe: {prices.shape[1]} US large-cap "
        f"stocks (bundled real daily prices from the `skfolio` package), "
        f"{prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}. "
        f"Benchmark: S&P 500 index.*",
        "",
        "## Protocol (the point of this study)",
        "",
        "- **No lookahead**: every signal at date *t* uses data through *t* only; "
        "the engine applies weights with a **one-day execution lag**.",
        "- **Costs**: 5 bps one-way on all turnover (sensitivity at 0/5/10/20 bps below).",
        "- **In-sample / out-of-sample**: parameters chosen on 1990-2009 by net "
        "Sharpe; 2010-2022 evaluated exactly once.",
        "- **Walk-forward**: parameters re-selected every ~2 years on an expanding "
        "window from 2000 — what a live investor could actually have done.",
        f"- **Multiple testing**: {n_trials} configurations were tried in total; the "
        "Deflated Sharpe Ratio (DSR) charges each result for all of them. "
        "DSR > 0.95 = unlikely to be selection luck.",
        "- **Bootstrap**: 95% CIs for Sharpe from a 21-day moving-block bootstrap.",
        "",
        "## Out-of-sample results (2010-2022, net of 5 bps)",
        "",
        tbl(summary[["Config (chosen in-sample)", "IS Sharpe", "Sharpe",
                     "Sharpe 95% CI lo", "Sharpe 95% CI hi", "CAGR", "AnnVol",
                     "MaxDD", "AnnTurnover", "DSR"]]),
        "",
        "![OOS equity curves](equity_curves_oos.png)",
        "",
        "## Walk-forward results (2000-2022, net of 5 bps)",
        "",
        tbl(wf_stats[["Sharpe", "CAGR", "AnnVol", "MaxDD"]]),
        "",
        "![Walk-forward equity](walkforward_equity.png)",
        "",
        "## Transaction-cost sensitivity (OOS Sharpe)",
        "",
        tbl(cost_df),
        "",
        "![Cost sensitivity](cost_sensitivity.png)",
        "",
        "![Drawdowns](drawdowns_oos.png)",
        "",
        "## Caveats (read before believing anything above)",
        "",
        "- **Survivorship bias**: the 20 stocks are *current* S&P 500 members that "
        "existed through 1990-2022. Long-only absolute returns are inflated; "
        "relative comparisons between strategies are more trustworthy than levels.",
        "- **Small universe**: 20 names is thin for cross-sectional strategies; "
        "published momentum/reversal studies use thousands of stocks.",
        "- **Execution model**: close-to-close fills, linear costs, no market "
        "impact, no borrow fees on shorts, no taxes.",
        "- **No order-book data**: every market-data host is blocked by this "
        "environment's network policy, so the order-book signal module "
        "(`orderbook_signals.py`: queue imbalance, microprice, Cont-Kukanov-"
        "Stoikov OFI) ships verified code but no empirical results. Backtesting "
        "microstructure alpha on synthetic books would be self-deception.",
        "- **There are no guaranteed 'winning' strategies.** The honest output of "
        "a correct backtest is a small set of risk premia (momentum, trend, "
        "vol-targeting) with modest Sharpe and fat left tails — not a money machine.",
        "",
    ]
    with open(os.path.join(RESULTS, "REPORT.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
