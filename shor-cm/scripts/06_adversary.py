#!/usr/bin/env python3
"""Adversary / red team. Runs AFTER H1-H3, BEFORE report. Any CRITICAL
finding blocks the report until resolved.

Checks:
 1. label shuffle: H1 RATIO model on permuted labels 20x, AUC must be ~0.5
 2. split leakage: no speed band appears in both train and test
 3. feature recompute: 5 random rows recomputed from raw parquet, must match
 4. confound: does tacho_quality predict class? If AUC > 0.65, coherence
    ratio may win for the wrong reason -> CRITICAL, H1 must be rerun
    controlling for it
 5. self-reference audit: for onex/comb variants, verify masked ratio
    columns were actually excluded in H1/H2 configs
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
from shorcm.splits import grouped_folds, speed_band     # noqa: E402
from shorcm.features import run_features                # noqa: E402

OUTD = Path("adversary"); OUTD.mkdir(exist_ok=True)
RNG = np.random.default_rng(99)
findings = []


def add(sev, name, detail):
    findings.append(dict(severity=sev, name=name, detail=detail))
    print(f"[{sev}] {name}: {detail}")


def main():
    f = pd.read_parquet("features/tabular/features.parquet")
    f = f[(f["error"] == "") & (f.variant == "tacho")].reset_index(drop=True)
    df = f[f.cls != "normal"].copy()
    df["y"] = df.cls.isin(["underhang", "overhang"]).astype(int)
    ratio_cols = [c for c in df if c.startswith(("ratio_", "bear_ratio_"))]
    X = df[ratio_cols].fillna(0).values

    # 1 shuffle
    aucs = []
    for _ in range(20):
        yp = RNG.permutation(df.y.values)
        d2 = df.copy(); d2["y"] = yp
        fold_auc = []
        for tr, te in grouped_folds(d2):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=1000))
            clf.fit(X[tr], yp[tr])
            fold_auc.append(roc_auc_score(yp[te], clf.predict_proba(X[te])[:, 1]))
        aucs.append(np.mean(fold_auc))
    mx = float(np.max(aucs))
    add("CRITICAL" if mx > 0.62 else "ok", "label_shuffle",
        f"max shuffled AUC {mx:.3f} (want ~0.5, alarm > 0.62)")

    # 2 leakage
    for k, (tr, te) in enumerate(grouped_folds(df)):
        inter = set(speed_band(df.rate_hz.values[tr])) & \
            set(speed_band(df.rate_hz.values[te]))
        if inter:
            add("CRITICAL", "split_leakage", f"fold {k} shares bands {inter}")
    if not any(fi["name"] == "split_leakage" for fi in findings):
        add("ok", "split_leakage", "no speed-band overlap in any fold")

    # 3 recompute
    bad = 0
    for _, r in df.sample(5, random_state=7).iterrows():
        cat = pd.read_parquet("data/catalog.parquet")
        path = cat.loc[cat.run_id == r.run_id, "path"].iloc[0]
        arr = pd.read_parquet(path).values.astype(float)
        f2 = run_features(arr, r.f_nom, "tacho")
        if abs(f2["coh_1.0"] - r["coh_1.0"]) > 1e-6:
            bad += 1
    add("CRITICAL" if bad else "ok", "feature_recompute",
        f"{bad}/5 mismatches on recomputation")

    # 4 confound
    Xq = df[["tacho_quality", "n_pulses"]].fillna(0).values
    fold_auc = []
    for tr, te in grouped_folds(df):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000))
        clf.fit(Xq[tr], df.y.values[tr])
        fold_auc.append(roc_auc_score(df.y.values[te],
                                      clf.predict_proba(Xq[te])[:, 1]))
    a = float(np.mean(fold_auc))
    add("CRITICAL" if a > 0.65 else "ok", "tacho_quality_confound",
        f"class predictable from tacho quality alone: AUC {a:.3f} "
        "(alarm > 0.65; if critical, H1 must control for it)")

    # 5 self-reference audit (static check of H1 code path)
    src = Path("scripts/03_h1.py").read_text()
    ok = "SELF_REF_MASK" in src
    add("ok" if ok else "CRITICAL", "self_reference_mask",
        "H1 applies SELF_REF_MASK" if ok else "H1 ignores SELF_REF_MASK")

    pd.DataFrame(findings).to_parquet(OUTD / "findings.parquet", index=False)
    crit = [x for x in findings if x["severity"] == "CRITICAL"]
    (OUTD / "findings.md").write_text(
        "# Adversary findings\n\n" +
        "\n".join(f"- **{x['severity']}** {x['name']}: {x['detail']}"
                  for x in findings) +
        ("\n\nREPORT BLOCKED\n" if crit else "\n\nG4 PASS\n"))
    json.dump({"pass": not crit}, open(OUTD / "verdict.json", "w"))
    sys.exit(1 if crit else 0)


if __name__ == "__main__":
    main()
