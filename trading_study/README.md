# Trading strategy study: advanced strategies, backtested correctly

A self-contained, reproducible study of the classic "winning" strategy
families from the empirical finance literature, with a backtesting protocol
whose main purpose is to **not lie to you**. The headline lesson is the
methodology, not any single equity curve.

## Data

Real daily prices of 20 US large-cap stocks (1990-2022) plus the S&P 500
index, bundled inside the [`skfolio`](https://pypi.org/project/skfolio/) PyPI
package — chosen because this execution environment's network policy blocks
every market-data host (Yahoo, Stooq, Binance, LOBSTER), while PyPI is
allowed. **Known bias:** these are surviving current index members, so
long-only *levels* are inflated; relative comparisons are more trustworthy.

## Strategy families

| Family | Reference |
|---|---|
| Cross-sectional momentum (long-only and long-short) | Jegadeesh & Titman (1993) |
| Time-series momentum | Moskowitz, Ooi & Pedersen (2012) |
| Short-term reversal | Jegadeesh (1990) |
| Moving-average trend following | Brock, Lakonishok & LeBaron (1992) |
| Volatility targeting | Moreira & Muir (2017) |
| Inverse-vol risk parity | Asness, Frazzini & Pedersen (2012) |
| Equal weight / index buy & hold | benchmarks |

`orderbook_signals.py` additionally implements the modern microstructure
signals (queue imbalance, Stoikov microprice, Cont-Kukanov-Stoikov order-flow
imbalance) with hand-verified self-tests, ready for real L2 data — no
empirical claims are made for them because no order-book data is reachable
here, and backtesting them on synthetic books would be self-deception.

## What "backtested correctly" means here

1. **No lookahead** — signals at *t* use data through *t*; execution is lagged
   one day (`weights.shift(1)`), the omission that creates most fake alpha.
2. **Transaction costs** — 5 bps one-way on turnover, sensitivity at 0/5/10/20.
3. **Untouched out-of-sample** — parameters picked on 1990-2009 by net Sharpe;
   2010-2022 evaluated exactly once.
4. **Walk-forward** — parameters re-selected every ~2 years on an expanding
   window from 2000, mimicking a live investor.
5. **Multiple-testing correction** — the Deflated Sharpe Ratio (Bailey & Lopez
   de Prado 2014) charges every result for all 24 configurations tried.
6. **Block-bootstrap** 95% confidence intervals for Sharpe ratios.

## Run it

```bash
pip install -r requirements.txt
python3 run_study.py        # writes tables, figures, REPORT.md to results/
python3 orderbook_signals.py  # microstructure signal self-tests
```

## Headline findings (see `results/REPORT.md` for full tables)

- **Short-term reversal is the cautionary tale**: mildly positive in-sample,
  Sharpe **-0.29 out-of-sample** after 5 bps, and **-1.47 at 20 bps** — paper
  alpha destroyed by turnover, exactly as the literature predicts post-2000.
- **Volatility-targeted equal weight** was the only strategy whose Deflated
  Sharpe cleared 0.95 (OOS Sharpe 1.03 at half the benchmark's drawdown) —
  and it is a *risk management* overlay, not a return predictor.
- Long-only **cross-sectional momentum** had the best OOS Sharpe (0.97) and
  CAGR (21%), but on a survivorship-biased 20-name universe its DSR (0.93)
  stays below the bar; long-short momentum on 20 names is statistically
  indistinguishable from zero (DSR 0.18).
- Every positive-Sharpe strategy's walk-forward Sharpe (2000-2022, a harder
  period including two crashes) is **lower** than its one-shot OOS Sharpe —
  even honest parameter selection degrades when done repeatedly in real time.
- **No strategy here is a money machine.** Correct backtesting reliably turns
  "winning strategies" into a shortlist of modest, cost-sensitive risk premia.
