#!/usr/bin/env python3
"""Pattern-layer features -> blind fault ML.

Stage A: for every run in experiments/v2/results.parquet, decompose at
the EST speed (sheet_f_hat — deployment condition) with the kinematic
sheet and emit pf_* features from the isolating pattern layer.

Stage B: GroupKFold by speed band — 6-way LightGBM on ledger features
alone vs ledger + pattern features; paired bootstrap on per-fold OOF
predictions; promotion_gate decides.

ENV: WORKERS (4), STAGE (both|a|b).
"""
import json, os, sys, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import simforge_v2 as V2                 # noqa: E402
from shorcm import patterns as PT                    # noqa: E402
from shorcm import metrics as MX                     # noqa: E402
from shorcm import splits as SP                      # noqa: E402

WORKERS = int(os.environ.get("WORKERS", 4))
STAGE = os.environ.get("STAGE", "both")
OUT = Path("experiments/pattern_features")
FAULTS6 = ["healthy", "imbalance", "misalignment", "looseness",
           "bearing", "gear"]

PF_COLS = PT.PF_COLS


def pf_one(args):
    run_id, f_hat = args
    m, x = V2.sample_run(int(run_id))
    try:
        pats, _ = PT.decompose(x, V2.FS, float(f_hat),
                               sheet=V2.kinematic_sheet(m))
    except Exception:
        return None
    d = PT.pattern_features(pats)      # SHARED with the deployment path
    d["run_id"] = int(run_id)
    return d


def stage_a(df):
    args = list(zip(df.run_id.values, df.sheet_f_hat.values))
    with Pool(WORKERS, maxtasksperchild=40) as pool:
        rows = [r for r in pool.imap_unordered(pf_one, args, chunksize=4)
                if r is not None]
    pf = pd.DataFrame(rows)
    pf.to_parquet(OUT / "pattern_features.parquet", index=False)
    return pf


def stage_b(df, pf):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import f1_score
    d = df.merge(pf, on="run_id", how="inner")
    le = [c for c in d.columns if c.startswith("le_")]
    d = d.dropna(subset=le)
    groups = [SP.speed_band(f) for f in d.f_true]
    y = pd.Categorical(d.fault, categories=FAULTS6).codes
    oof = {}
    for tag, cols in (("ledger", le), ("ledger+pattern", le + PF_COLS)):
        X = d[cols].values
        pred = np.full(len(d), -1)
        for tr, te in GroupKFold(5).split(X, y, groups):
            mdl = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                     num_leaves=31, random_state=20260709,
                                     verbose=-1)
            mdl.fit(X[tr], y[tr])
            pred[te] = mdl.predict(X[te])
        oof[tag] = pred
    mf1 = {t: float(f1_score(y, p, average="macro"))
           for t, p in oof.items()}
    deltas = MX.paired_bootstrap_delta(          # metric(b) - metric(a)
        y, oof["ledger"], oof["ledger+pattern"],
        lambda yy, pp: f1_score(yy, pp, average="macro"))
    gate = MX.promotion_gate(deltas, guardrails_ok=True)
    find = {"n": int(len(d)), "mF1": mf1, "gate": gate,
            "features_added": PF_COLS}
    (OUT / "findings.json").write_text(json.dumps(find, indent=1))
    print(json.dumps(find, indent=1))


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet("experiments/v2/results.parquet")
    if STAGE in ("both", "a"):
        pf = stage_a(df)
    else:
        pf = pd.read_parquet(OUT / "pattern_features.parquet")
    if STAGE in ("both", "b"):
        stage_b(df, pf)
    print("wall_s", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
