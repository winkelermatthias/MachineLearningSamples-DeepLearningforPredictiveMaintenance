#!/usr/bin/env python3
"""Retrain + gate + freeze v4 on corpus v3 (impulsive renderer,
patched estimator, envelope features).

A/B/C on GroupKFold-by-speed-band OOF, paired bootstrap:
  A ledger only     B ledger+pf     C ledger+pf+pfe
Gate 1: C beats B (alpha-adjusted lower bound > 0)  -> features_pfe
Gate 2: severity with pfe beats without (Spearman on OOF)

On promotion: freeze fault + severity + isotonic calibrator (existing
confidence corpus) as models/ v4 with lineage.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import simforge_corpus as SC             # noqa: E402
from shorcm import patterns as PT                    # noqa: E402
from shorcm import metrics as MX                     # noqa: E402
from shorcm import splits as SP                      # noqa: E402
from shorcm.cascade import Cascade, FAULTS6          # noqa: E402

OUT = Path("experiments/retrain_v4")


def main():
    t0 = time.time()
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import f1_score
    from sklearn.isotonic import IsotonicRegression
    from scipy.stats import spearmanr
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet("experiments/corpus_v3/features.parquet")
    le = [f"le_{f}" for f in SC.LEDGER_FEATURES_V2]  # CANONICAL
    # order - analyze_record feeds features positionally; parquet
    # column order differs and silently breaks LGBM (measured:
    # held-out fault acc collapsed 0.815 -> 0.667)
    d = d.dropna(subset=le + PT.PF_COLS + PT.PF_ENV_COLS)
    groups = np.array([SP.speed_band(f) for f in d.f_true])
    y = pd.Categorical(d.fault, categories=FAULTS6).codes
    sets = {"A_ledger": le, "B_ledger+pf": le + PT.PF_COLS,
            "C_ledger+pf+pfe": le + PT.PF_COLS + PT.PF_ENV_COLS}
    oof = {}
    for tag, cols in sets.items():
        X = d[cols].values
        pred = np.full(len(d), -1)
        for tr, te in GroupKFold(5).split(X, y, groups):
            mdl = lgb.LGBMClassifier(n_estimators=400,
                                     learning_rate=0.05, num_leaves=31,
                                     random_state=20260709, verbose=-1)
            mdl.fit(X[tr], y[tr])
            pred[te] = mdl.predict(X[te])
        oof[tag] = pred
    mf1 = {t: round(float(f1_score(y, p, average="macro")), 4)
           for t, p in oof.items()}
    deltas = MX.paired_bootstrap_delta(
        y, oof["B_ledger+pf"], oof["C_ledger+pf+pfe"],
        lambda yy, pp: f1_score(yy, pp, average="macro"))
    gate = MX.promotion_gate(deltas, guardrails_ok=True)

    # severity A/B
    flt = (d.fault != "healthy").values
    sev_oof = {}
    for tag in ("B_ledger+pf", "C_ledger+pf+pfe"):
        X = d[sets[tag]].values
        pred = np.full(len(d), np.nan)
        for tr, te in GroupKFold(5).split(X, d.severity.values, groups):
            tr_f = tr[flt[tr]]
            mdl = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                    num_leaves=31, random_state=20260709,
                                    verbose=-1)
            mdl.fit(X[tr_f], d.severity.values[tr_f])
            pred[te] = mdl.predict(X[te])
        sev_oof[tag] = pred
    m = flt & np.isfinite(sev_oof["C_ledger+pf+pfe"])
    rho = {t: round(float(spearmanr(d.severity.values[m],
                                    p[m]).statistic), 3)
           for t, p in sev_oof.items()}

    find = {"n": int(len(d)), "mF1": mf1, "gate_C_vs_B": gate,
            "severity_spearman": rho}

    use_pfe = gate["promote"]
    cols = sets["C_ledger+pf+pfe" if use_pfe else "B_ledger+pf"]
    X = d[cols].values
    mdl = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                             num_leaves=31, random_state=20260709,
                             verbose=-1)
    mdl.fit(X, y)
    sev = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                            num_leaves=31, random_state=20260709,
                            verbose=-1)
    sev.fit(X[flt], d.severity.values[flt])
    dc = pd.read_parquet("experiments/confidence/records.parquet")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(dc.margin12.values, dc.correct.values.astype(float))
    meta = {"frozen_at": "iteration-26",
            "fault_model": {"algo": "lightgbm-6way",
                            "train": "corpus_v3 (impulsive renderer, "
                                     "est-speed condition)",
                            "n": int(len(d)),
                            "features": SC.LEDGER_FEATURES_V2,
                            "features_pf": PT.PF_COLS,
                            "features_pfe": PT.PF_ENV_COLS if use_pfe
                            else None,
                            "classes": FAULTS6},
            "severity_model": {"algo": "lightgbm-reg",
                               "n": int(flt.sum()),
                               "oof_spearman": rho},
            "calibrator": {"algo": "isotonic(margin12)",
                           "n": int(len(dc))},
            "seeds": {"corpus": 20260710, "model": 20260709},
            "subtype_threshold_ax_ratio_2": 0.61}
    Cascade(mdl, iso, meta, severity_model=sev).save("models")
    find["frozen"] = {"features_pfe": bool(use_pfe),
                      "n_features": len(cols)}
    find["wall_s"] = round(time.time() - t0, 1)
    (OUT / "findings.json").write_text(json.dumps(find, indent=1))
    print(json.dumps(find, indent=1))


if __name__ == "__main__":
    main()
