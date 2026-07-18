#!/usr/bin/env python3
"""H1: coherence ratio separates shaft-locked faults from bearing faults.
Runs the WHOLE experiment once per phase variant (tacho, onex, comb, nominal)
so the headline table is: variant x feature-set x split -> AUC (95% CI).

Feature sets:
  RATIO : coherence-ratio columns only (the hypothesis)
  HZ    : baseline, incoherent order amps mapped back to Hz-binned features
          (approximated here by inc_* columns WITHOUT order normalization
          being available to the model: we add f_nom so the model must learn
          speed itself, mimicking a plain FFT pipeline)
  INC   : incoherent order amps (order tracking, no coherence)
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm.splits import grouped_folds, extreme_speed_split   # noqa: E402
from shorcm.features import SELF_REF_MASK                      # noqa: E402

RNG = np.random.default_rng(20260709)
OUTD = Path("experiments/h1"); OUTD.mkdir(parents=True, exist_ok=True)

LOCKED = {"imbalance", "horizontal-misalignment", "vertical-misalignment"}
BEARING = {"underhang", "overhang"}


def boot_auc(y, p, n=1000):
    aucs = []
    idx = np.arange(len(y))
    for _ in range(n):
        s = RNG.choice(idx, len(idx))
        if len(np.unique(y[s])) == 2:
            aucs.append(roc_auc_score(y[s], p[s]))
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), \
        float(np.percentile(aucs, 97.5))


def featset(df, name, variant):
    drop = set(SELF_REF_MASK.get(variant, []))
    if name == "RATIO":
        cols = [c for c in df if c.startswith(("ratio_", "bear_ratio_"))
                and c not in drop]
    elif name == "INC":
        cols = [c for c in df if c.startswith(("inc_", "bear_inc_"))]
    elif name == "HZ":
        cols = [c for c in df if c.startswith("inc_")] + ["f_nom"]
    return df[cols].fillna(0).values


def eval_split(df, tr, te, fs_name, variant):
    X = featset(df, fs_name, variant)
    y = df["y"].values
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])[:, 1]
    return boot_auc(y[te], p)


def main():
    all_f = pd.read_parquet("features/tabular/features.parquet")
    all_f = all_f[all_f["error"] == ""]
    results = []
    for variant in ["tacho", "onex", "comb", "nominal"]:
        df = all_f[all_f.variant == variant].reset_index(drop=True)
        df = df[df.cls.isin(LOCKED | BEARING)].reset_index(drop=True)
        df["y"] = df.cls.isin(BEARING).astype(int)
        for fs_name in ["RATIO", "INC", "HZ"]:
            # grouped 5-fold
            for k, (tr, te) in enumerate(grouped_folds(df)):
                auc, lo, hi = eval_split(df, tr, te, fs_name, variant)
                results.append(dict(variant=variant, fset=fs_name,
                                    split=f"gkf{k}", auc=auc, lo=lo, hi=hi,
                                    n_test=len(te)))
            # extreme-speed hold-out
            tr, te = extreme_speed_split(df)
            if len(te) > 20:
                auc, lo, hi = eval_split(df, tr, te, fs_name, variant)
                results.append(dict(variant=variant, fset=fs_name,
                                    split="extreme_speed", auc=auc, lo=lo,
                                    hi=hi, n_test=len(te)))
        # severity monotonicity (imbalance mass vs 1x coherent amp)
        imb = df[df.cls == "imbalance"].copy()
        imb["mass"] = pd.to_numeric(imb.sev.str.replace("g", ""), errors="coerce")
        if imb.mass.notna().sum() > 10:
            from scipy.stats import spearmanr
            rho, p = spearmanr(imb.mass, imb["coh_1.0"])
            results.append(dict(variant=variant, fset="SEVERITY_1X",
                                split="spearman", auc=rho, lo=p, hi=np.nan,
                                n_test=len(imb)))
    res = pd.DataFrame(results)
    res.to_parquet(OUTD / "results.parquet", index=False)
    summ = res[res.split.str.startswith("gkf")].groupby(
        ["variant", "fset"]).auc.mean().unstack()
    ext = res[res.split == "extreme_speed"].pivot_table(
        index="variant", columns="fset", values="auc")
    (OUTD / "summary.md").write_text(
        "# H1 summary\n\n## grouped 5-fold mean AUC\n" + summ.to_markdown() +
        "\n\n## extreme-speed hold-out AUC\n" + ext.to_markdown() +
        "\n\nPass: RATIO(tacho) gkf AUC > 0.90. Key question: how much does "
        "RATIO degrade tacho -> onex -> comb -> nominal?\n")
    json.dump({"pass_h1": bool(summ.loc["tacho", "RATIO"] > 0.90)},
              open(OUTD / "verdict.json", "w"))
    print(summ)


if __name__ == "__main__":
    main()
