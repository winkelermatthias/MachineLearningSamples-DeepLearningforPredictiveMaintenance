#!/usr/bin/env python3
"""Freeze the deployment artifacts from the synthetic program:
 - fault_lgbm.joblib     6-way LightGBM on LEDGER_FEATURES_V2, trained
                         on the v2 corpus's est-speed (deployment
                         condition) ledgers — ALL 2000 runs
 - conf_isotonic.joblib  isotonic margin->P(speed correct) from the
                         confidence corpus — ALL 1199 records
 - meta.json             seeds, data lineage, feature list, versions

These artifacts are what gets pointed at MAFAULDA / Relos data. The
held-out certification (scripts/79) uses FRESH seeds no model has seen.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import simforge_corpus as SC             # noqa: E402
from shorcm.cascade import Cascade, FAULTS6          # noqa: E402

OUT = Path("models")


def main():
    t0 = time.time()
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from shorcm import patterns as PT
    df = pd.read_parquet("experiments/v2/results.parquet")
    pf = pd.read_parquet(
        "experiments/pattern_features/pattern_features.parquet")
    df = df.merge(pf, on="run_id", how="inner")
    feats = [f"le_{f}" for f in SC.LEDGER_FEATURES_V2] + PT.PF_COLS
    d = df.dropna(subset=feats)
    X = d[feats].values
    y = pd.Categorical(d.fault, categories=FAULTS6).codes
    mdl = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                             num_leaves=31, random_state=20260709,
                             verbose=-1)
    mdl.fit(X, y)

    dc = pd.read_parquet("experiments/confidence/records.parquet")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(dc.margin12.values, dc.correct.values.astype(float))

    meta = {"frozen_at": "iteration-19",
            "fault_model": {"algo": "lightgbm-6way",
                            "train": "experiments/v2 est-speed ledgers "
                                     "+ pattern-layer features",
                            "n": int(len(d)),
                            "features": SC.LEDGER_FEATURES_V2,
                            "features_pf": PT.PF_COLS,
                            "classes": FAULTS6},
            "calibrator": {"algo": "isotonic(margin12)",
                           "train": "experiments/confidence",
                           "n": int(len(dc))},
            "seeds": {"corpus_v2": 20260710, "confidence": 20260710,
                      "model": 20260709},
            "subtype_threshold_ax_ratio_2": 0.61}
    Cascade(mdl, iso, meta).save(OUT)
    print(json.dumps({"frozen": True, "n_fault": len(d),
                      "n_cal": len(dc),
                      "wall_s": round(time.time() - t0, 1)}, indent=1))


if __name__ == "__main__":
    main()
