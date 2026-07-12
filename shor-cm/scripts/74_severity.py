#!/usr/bin/env python3
"""O4 ordinal severity head.

LightGBM regressor: severity in [0, 1] for FAULTED machines from
ledger + pattern-layer features at est speed. GroupKFold by speed
band. PRE-REGISTERED gate: OOF Spearman rho >= 0.6 overall AND >= 0.45
per fault class (guardrail: no class rides the average).

Also reports healthy-separation: severity_score on healthy machines
must sit below the faulted median (AUC healthy-vs-severe>=0.5).
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import simforge_corpus as SC             # noqa: E402
from shorcm import patterns as PT                    # noqa: E402
from shorcm import splits as SP                      # noqa: E402

OUT = Path("experiments/severity")


def main():
    t0 = time.time()
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet("experiments/v2/results.parquet")
    pf = pd.read_parquet(
        "experiments/pattern_features/pattern_features.parquet")
    d = df.merge(pf, on="run_id", how="inner")
    feats = [f"le_{f}" for f in SC.LEDGER_FEATURES_V2] + PT.PF_COLS
    d = d.dropna(subset=feats)
    groups = np.array([SP.speed_band(f) for f in d.f_true])
    flt = (d.fault != "healthy").values
    X, ysev = d[feats].values, d.severity.values

    oof = np.full(len(d), np.nan)
    for tr, te in GroupKFold(5).split(X, ysev, groups):
        tr_f = tr[flt[tr]]                 # train on faulted only
        mdl = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                num_leaves=31, random_state=20260709,
                                verbose=-1)
        mdl.fit(X[tr_f], ysev[tr_f])
        oof[te] = mdl.predict(X[te])

    m = flt & np.isfinite(oof)
    rho_all = float(spearmanr(ysev[m], oof[m]).statistic)
    per_class = {}
    for f in sorted(d.fault[m].unique()):
        mm = m & (d.fault == f).values
        per_class[f] = round(float(
            spearmanr(ysev[mm], oof[mm]).statistic), 3)
    auc = float(roc_auc_score(
        flt[np.isfinite(oof)] & (ysev[np.isfinite(oof)] > 0.5),
        oof[np.isfinite(oof)]))
    gate = rho_all >= 0.6 and all(v >= 0.45 for v in per_class.values())

    pd.DataFrame({"run_id": d.run_id, "fault": d.fault,
                  "severity": ysev, "oof": oof}).to_parquet(
        OUT / "severity_oof.parquet", index=False)
    find = {"n_faulted": int(m.sum()), "spearman_all": round(rho_all, 3),
            "spearman_per_class": per_class,
            "auc_healthy_vs_severe": round(auc, 3),
            "gate_rho_0.6_all_0.45_per_class": bool(gate),
            "wall_s": round(time.time() - t0, 1)}
    (OUT / "findings.json").write_text(json.dumps(find, indent=1))
    print(json.dumps(find, indent=1))


if __name__ == "__main__":
    main()
