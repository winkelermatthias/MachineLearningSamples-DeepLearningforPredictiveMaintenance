#!/usr/bin/env python3
"""Retrain fault + severity heads on the SimForge v6 corpus and freeze
as models_v6/ (PHASE3 A8: the artifact IS what gets evaluated).

Feature set = ledger + pattern + envelope (v5's winning set C), same
LGBM hyperparameters as v5 for a clean generator-only A/B. The speed
confidence calibrator is COPIED from v5 (speed head unchanged this
iteration — documented in meta.json lineage).

Internal (synthetic) OOF numbers are reported for the record but are
NOT claims: claims come from scripts/105_blind_gates.py on real data.
"""
import json, shutil, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import simforge_corpus as SC              # noqa: E402
from shorcm import patterns as PT                     # noqa: E402
from shorcm import splits as SP                       # noqa: E402
from shorcm.cascade import FAULTS6                    # noqa: E402

OUT = Path("models_v6")
EXP = Path("experiments/retrain_v6")


def main():
    t0 = time.time()
    import joblib
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import f1_score, accuracy_score
    from scipy.stats import spearmanr
    EXP.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet("experiments/corpus_v6/features.parquet")
    le = [f"le_{f}" for f in SC.LEDGER_FEATURES_V2]   # CANONICAL order
    feats = le + PT.PF_COLS + PT.PF_ENV_COLS
    n_total = len(d)
    d = d.dropna(subset=feats).reset_index(drop=True)
    groups = np.array([SP.speed_band(f) for f in d.f_true])
    y = pd.Categorical(d.fault, categories=FAULTS6).codes
    X = d[feats].values

    oof = np.full(len(d), -1)
    for tr, te in GroupKFold(5).split(X, y, groups):
        mdl = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                 num_leaves=31, random_state=20260709,
                                 verbose=-1)
        mdl.fit(X[tr], y[tr])
        oof[te] = mdl.predict(X[te])
    rep = {"n_corpus": n_total, "n_usable": len(d),
           "speed_ok_rate": round(float(d.speed_ok.mean()), 3),
           "oof_macro_f1": round(float(
               f1_score(y, oof, average="macro")), 4),
           "oof_acc": round(float(accuracy_score(y, oof)), 4),
           "note": "synthetic-internal; claims come from blind real gates"}

    # freeze: full fit
    fault = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                               num_leaves=31, random_state=20260709,
                               verbose=-1).fit(X, y)
    sv = d.severity.values
    sev = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                            num_leaves=31, random_state=20260709,
                            verbose=-1).fit(X, sv)
    sev_oof = np.full(len(d), np.nan)
    for tr, te in GroupKFold(5).split(X, sv, groups):
        m2 = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                               num_leaves=31, random_state=20260709,
                               verbose=-1)
        m2.fit(X[tr], sv[tr])
        sev_oof[te] = m2.predict(X[te])
    rep["oof_severity_spearman"] = round(float(
        spearmanr(sv, sev_oof).statistic), 4)

    OUT.mkdir(exist_ok=True)
    joblib.dump(fault, OUT / "fault_lgbm.joblib")
    joblib.dump(sev, OUT / "severity_lgbm.joblib")
    shutil.copy("models/conf_isotonic.joblib",
                OUT / "conf_isotonic.joblib")
    old_meta = json.loads(Path("models/meta.json").read_text())
    meta = {"version": "v6-iter33",
            "corpus": "experiments/corpus_v6 (SimForge v6, BASE_SEED6 "
                      "20260718, N=%d)" % n_total,
            "fault_model": {"features_pf": True, "features_pfe": True,
                            "classes": FAULTS6},
            "conf_isotonic": "copied from v5 (speed head unchanged): "
                             + str(old_meta.get("conf_isotonic", "")),
            "trained_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                         time.gmtime()),
            "lineage_prev": old_meta.get("version", "v5")}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    rep["wall_s"] = round(time.time() - t0, 1)
    (EXP / "findings.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
