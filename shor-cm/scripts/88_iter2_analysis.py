#!/usr/bin/env python3
"""Iteration-2 analysis on the triple-judgment results.

 1. SPEED scoreboard: baseline(pinned) vs baseline2(ladder-fixed) vs
    peakshor vs union, with per-fault breakdown.
 2. O1 confidence calibration: isotonic fitted on EVEN speed bands,
    ECE reported on ODD bands only (union arm).
 3. FAULT ML arm on the evidence ledger, two conditions: true speed
    ("separately") and estimated speed ("together"), GroupKFold by
    speed band; rules arm same conditions; paired promotion gate
    ML-over-rules under attempts from the loop ledger.
 4. O4 severity: observational Spearman of the promoted drivers
    (bearing driver now uns_abs) in both conditions.
 5. JOINT: end-to-end all-stage success rates.

Outputs: experiments/peakshor/iter2_findings.json + figures.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import simforge_lite as SF               # noqa: E402
from shorcm import simforge_corpus as SC             # noqa: E402
from shorcm import splits as SP                      # noqa: E402
from shorcm import metrics as M                      # noqa: E402

OUT = Path("experiments/peakshor")
SEED = 20260709
FAULTS = SF.FAULTS
DRIVERS = {"imbalance": "a1", "misalignment": "a2_over_a1",
           "looseness": "e_half", "bearing": "uns_abs"}


def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(conf)
    for i in range(bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1] + (i == bins - 1))
        if m.sum():
            e += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def speed_board(df):
    base = pd.read_parquet("experiments/massive/o1_results.parquet")
    base = base[base.ablation == "full"]
    board = {"baseline_pinned": {
        "top1": round(float(base.top1.mean()), 4),
        "top3": round(float(base.top3.mean()), 4),
        "octave": round(float(base.octave.mean()), 4), "n": len(base)}}
    for arm in ("baseline2", "peakshor", "union"):
        d = df.dropna(subset=[f"{arm}_top1"])
        board[arm] = {k: round(float(d[f"{arm}_{k}"].mean()), 4)
                      for k in ("top1", "top3", "octave")} | {"n": len(d)}
    by_fault = df.groupby("fault")[["union_top1", "union_top3",
                                    "union_octave"]].mean().round(3)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    arms = list(board)
    x = np.arange(len(arms))
    ax.bar(x - 0.2, [board[a]["top1"] for a in arms], 0.35,
           label="top-1", color="#0FB5A6")
    ax.bar(x + 0.2, [board[a]["top3"] for a in arms], 0.35,
           label="top-3", color="#666")
    ax.set_xticks(x, arms, fontsize=8)
    ax.set_ylabel("rate @1%"); ax.legend()
    ax.set_title("O1 blind speed by arm, n=2000 (baseline2: n=400)",
                 fontsize=10)
    fig.savefig(OUT / "speed_arms.png", dpi=140, bbox_inches="tight")
    return board, by_fault


def calibration(df):
    band = SP.speed_band(df.f_true.values)
    fit = band % 2 == 0
    conf = df.union_conf.values
    ok = df.union_top1.values.astype(float)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(conf[fit], ok[fit])
    cal = iso.predict(conf[~fit])
    res = {"ece_raw_holdout": round(ece(conf[~fit], ok[~fit]), 4),
           "ece_calibrated_holdout": round(ece(cal, ok[~fit]), 4),
           "n_fit": int(fit.sum()), "n_holdout": int((~fit).sum())}
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot([0, 1], [0, 1], "--", c="gray", lw=1)
    for c, o, lab, col in ((conf[~fit], ok[~fit], "raw", "#999"),
                           (cal, ok[~fit], "isotonic", "#0FB5A6")):
        bins = np.linspace(0, 1, 11)
        xs, ys = [], []
        for i in range(10):
            m = (c >= bins[i]) & (c < bins[i + 1] + (i == 9))
            if m.sum() > 5:
                xs.append(c[m].mean()); ys.append(o[m].mean())
        ax.plot(xs, ys, "o-", label=lab, color=col)
    ax.set_xlabel("stated confidence"); ax.set_ylabel("observed top-1")
    ax.set_title("O1 union reliability, odd-band holdout", fontsize=10)
    ax.legend()
    fig.savefig(OUT / "calibration.png", dpi=140, bbox_inches="tight")
    return res


def fault_arms(df):
    import lightgbm as lgb
    res, oof_store = {}, {}
    for cond, pre in (("true_speed", "lt"), ("est_speed", "le")):
        feats = [f"{pre}_{f}" for f in SC.LEDGER_FEATURES]
        d = df.dropna(subset=feats).reset_index(drop=True)
        X = d[feats].values
        y = pd.Categorical(d.fault, categories=FAULTS).codes
        groups = SP.speed_band(d.f_true.values)
        oof = np.full(len(d), -1)
        margin = np.zeros(len(d))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            mdl = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                     num_leaves=31, random_state=SEED,
                                     verbose=-1)
            mdl.fit(X[tr], y[tr])
            p = mdl.predict_proba(X[te])
            oof[te] = p.argmax(1)
            ps = np.sort(p, 1)
            margin[te] = ps[:, -1] - ps[:, -2]
        ml = np.array(FAULTS)[oof]
        rules = d[f"rules_{'true' if pre == 'lt' else 'est'}"].values
        yn = d.fault.values

        def mf1(a, b):
            return f1_score(a, b, average="macro", labels=FAULTS,
                            zero_division=0)

        deltas = M.paired_bootstrap_delta(yn, rules, ml, mf1, n=2000)
        gate = M.promotion_gate(deltas, guardrails_ok=True)
        res[cond] = {"n": len(d),
                     "rules_acc": round(float((rules == yn).mean()), 4),
                     "ml_acc": round(float((ml == yn).mean()), 4),
                     "rules_macro_f1": round(float(mf1(yn, rules)), 4),
                     "ml_macro_f1": round(float(mf1(yn, ml)), 4),
                     "gate_ml_over_rules": gate}
        oof_store[cond] = (d, ml, margin, yn)
    # coverage-accuracy, est-speed condition
    d, ml, margin, yn = oof_store["est_speed"]
    order = np.argsort(-margin)
    curve = []
    for c in np.linspace(0.2, 1.0, 33):
        k = max(int(c * len(d)), 1)
        s = order[:k]
        curve.append((float(c), float((ml[s] == yn[s]).mean())))
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(*zip(*curve), c="#0FB5A6")
    ax.set_xlabel("coverage"); ax.set_ylabel("accuracy")
    ax.set_title("O3 ML arm (est speed), margin abstention", fontsize=10)
    fig.savefig(OUT / "o3_coverage_iter2.png", dpi=140, bbox_inches="tight")
    res["coverage_090"] = round(float(
        [a for c, a in curve if abs(c - 0.9) < 0.02][0]), 4)
    return res


def severity(df):
    out = {}
    for cond, pre in (("true_speed", "lt"), ("est_speed", "le")):
        obs = {}
        for f, drv in DRIVERS.items():
            col = f"{pre}_{drv}"
            d = df[(df.fault == f)].dropna(subset=[col])
            if len(d) > 10:
                obs[f] = {"driver": drv, "n": int(len(d)),
                          "rho": round(float(
                              spearmanr(d.severity, d[col]).statistic), 3)}
        out[cond] = obs
    return out


def main():
    t0 = time.time()
    df = pd.read_parquet(OUT / "results.parquet")
    board, by_fault = speed_board(df)
    find = {"speed": board,
            "speed_union_by_fault": json.loads(by_fault.to_json()),
            "calibration": calibration(df),
            "fault": fault_arms(df),
            "severity": severity(df)}
    j = {}
    j["speed_top1_and_rules_fault"] = round(float(
        (df.union_top1 & (df.rules_est == df.fault)).mean()), 4)
    pat_cols = [c for c in df.columns if c.startswith(("pt_", "pe_"))]
    j["patterns_mean"] = {c: round(float(df[c].mean()), 3)
                          for c in sorted(pat_cols)}
    bd = df[df.fault == "bearing"]
    j["bearing_orderdomain_recall"] = {
        "true_speed": round(float(((np.abs(bd.lt_uns_top_order
                                    / bd.bear_order_true - 1) < 0.03)
                                   & (bd.lt_uns_abs > 0.1)).mean()), 3),
        "est_speed": round(float(((np.abs(bd.le_uns_top_order
                                   / bd.bear_order_true - 1) < 0.03)
                                  & (bd.le_uns_abs > 0.1)).mean()), 3)}
    find["joint"] = j
    find["wall_s"] = round(time.time() - t0, 1)
    (OUT / "iter2_findings.json").write_text(
        json.dumps(find, indent=1, default=str))
    print(json.dumps(find, indent=1, default=str))


if __name__ == "__main__":
    main()
