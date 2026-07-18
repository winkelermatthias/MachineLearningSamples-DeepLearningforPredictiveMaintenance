#!/usr/bin/env python3
"""V2 analysis: ML arm, per-archetype scoreboard, joint rates, figures.
Consumes experiments/v2/results.parquet."""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import simforge_v2 as V2                 # noqa: E402
from shorcm import simforge_corpus as SC             # noqa: E402
from shorcm import splits as SP                      # noqa: E402
from shorcm import metrics as M                      # noqa: E402

OUT = Path("experiments/v2")
SEED = 20260709
FAULTS = V2.FAULTS6


def main():
    t0 = time.time()
    df = pd.read_parquet(OUT / "results.parquet")
    find = {"n_runs": len(df)}

    # ---- speed ----
    sp = {}
    for arm in ("pure", "union", "sheet"):
        sp[arm] = {k: round(float(df[f"{arm}_{k}"].mean()), 4)
                   for k in ("top1", "top3", "octave", "shaft2")}
    sp["union_plus_shaft2_upper_bound"] = round(float(
        (df.union_top1 | df.union_shaft2).mean()), 4)
    find["speed"] = sp
    find["speed_by_archetype"] = json.loads(
        df.groupby("archetype")[["union_top1", "sheet_top1",
                                 "sheet_top3"]].mean().round(3).to_json())
    find["speed_by_speed_class"] = json.loads(
        df.groupby("low_speed")[["sheet_top1", "sheet_top3"]]
        .mean().round(3).to_json())

    # ---- fault: rules + ML, both conditions ----
    import lightgbm as lgb
    fault = {}
    ml_est_pred = None
    for cond, pre, rc in (("true_speed", "lt", "rules_true"),
                          ("est_speed", "le", "rules_est")):
        feats = [f"{pre}_{f}" for f in SC.LEDGER_FEATURES_V2]
        d = df.dropna(subset=feats).reset_index(drop=True)
        X = d[feats].values
        y = pd.Categorical(d.fault, categories=FAULTS).codes
        groups = SP.speed_band(d.f_true.values)
        oof = np.full(len(d), -1)
        for tr, te in GroupKFold(5).split(X, y, groups):
            mdl = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                     num_leaves=31, random_state=SEED,
                                     verbose=-1)
            mdl.fit(X[tr], y[tr])
            oof[te] = mdl.predict_proba(X[te]).argmax(1)
        ml = np.array(FAULTS)[oof]
        yn, rules = d.fault.values, d[rc].values

        def mf1(a, b):
            return f1_score(a, b, average="macro", labels=FAULTS,
                            zero_division=0)

        gate = M.promotion_gate(
            M.paired_bootstrap_delta(yn, rules, ml, mf1, n=2000),
            guardrails_ok=True)
        fault[cond] = {
            "n": len(d),
            "rules_acc": round(float((rules == yn).mean()), 4),
            "rules_macro_f1": round(float(mf1(yn, rules)), 4),
            "ml_acc": round(float((ml == yn).mean()), 4),
            "ml_macro_f1": round(float(mf1(yn, ml)), 4),
            "gate_promote": bool(gate["promote"]),
            "gate_delta_lo": round(float(gate["delta_lo_alpha_adj"]), 4)}
        if cond == "est_speed":
            ml_est_pred = pd.Series(ml, index=d.run_id)
            cm = confusion_matrix(yn, ml, labels=FAULTS)
            fig, ax = plt.subplots(figsize=(5.2, 4.2))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(FAULTS)), FAULTS, rotation=30,
                          fontsize=8)
            ax.set_yticks(range(len(FAULTS)), FAULTS, fontsize=8)
            for a in range(len(FAULTS)):
                for b in range(len(FAULTS)):
                    ax.text(b, a, cm[a, b], ha="center", va="center",
                            fontsize=8, color="white"
                            if cm[a, b] > cm.max() / 2 else "black")
            ax.set_xlabel("ML predicted (est speed)")
            ax.set_ylabel("true")
            ax.set_title("V2 6-way fault, end-to-end blind", fontsize=10)
            fig.colorbar(im, shrink=0.8)
            fig.savefig(OUT / "v2_confusion.png", dpi=140,
                        bbox_inches="tight")
        else:
            d_arch = d.assign(ml=ml)
            find["fault_ml_by_archetype_true_speed"] = {
                a: round(float((g.ml == g.fault).mean()), 3)
                for a, g in d_arch.groupby("archetype")}
    find["fault"] = fault

    # ---- joint ----
    dj = df.set_index("run_id")
    ml_ok = ml_est_pred.reindex(dj.index)
    find["joint"] = {
        "top1_and_rules": round(float(
            (dj.sheet_top1 & (dj.rules_est == dj.fault)).mean()), 4),
        "top1_and_ml": round(float(
            (dj.sheet_top1 & (ml_ok == dj.fault)).mean()), 4),
        "kinsheet_and_ml": round(float(
            ((dj.sheet_top1 | dj.sheet_shaft2) & (ml_ok == dj.fault))
            .mean()), 4)}

    # ---- patterns ----
    find["patterns"] = {c: round(float(df[c].mean()), 3)
                        for c in sorted(df.columns)
                        if c.startswith(("pt_", "pe_"))}
    bd = df[df.fault == "bearing"].dropna(subset=["lt_uns_top_order"])
    bo_true = bd.bear_hz_true / bd.f_true
    find["bearing_orderdomain_recall_true_speed"] = round(float(
        ((np.abs(bd.lt_uns_top_order / bo_true - 1) < 0.03)
         & (bd.lt_uns_abs > 0.1)).mean()), 3)

    # speed figure by archetype
    g = df.groupby("archetype")[["union_top1", "sheet_top1",
                                 "sheet_top3"]].mean()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(g))
    ax.bar(x - 0.25, g.union_top1, 0.25, label="plain top-1", color="#999")
    ax.bar(x, g.sheet_top1, 0.25, label="sheet top-1", color="#0FB5A6")
    ax.bar(x + 0.25, g.sheet_top3, 0.25, label="sheet top-3", color="#666")
    ax.set_xticks(x, g.index, rotation=20, fontsize=7)
    ax.legend(fontsize=8); ax.set_ylabel("rate @1%")
    ax.set_title("V2 blind speed by archetype (union arm)", fontsize=10)
    fig.savefig(OUT / "v2_speed_archetype.png", dpi=140,
                bbox_inches="tight")

    find["wall_s"] = round(time.time() - t0, 1)
    (OUT / "v2_findings.json").write_text(json.dumps(find, indent=1,
                                                     default=str))
    print(json.dumps(find, indent=1, default=str))


if __name__ == "__main__":
    main()
