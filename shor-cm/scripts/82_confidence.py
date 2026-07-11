#!/usr/bin/env python3
"""O1 confidence rebuild: margin features + trained calibrator.

The softmax over structure scores is uninformative (ECE 0.15-0.22
measured twice). Deployment contract (PHASE2 2.4): calibrated
P(top-1 within 1%), ECE <= 0.05, plus an abstain flag below 0.5.

Features per record (sheet arm, deployment condition): score margin
top1-top2 and top1-top3, raw top score, softmax conf, sheet template
match, s1x, snap fraction, penalty flags, octave-arbitration flag.
Calibrators, fitted on EVEN speed bands and evaluated on ODD bands:
  A  isotonic on margin12 alone
  B  LightGBM binary (correct within 1%?) with isotonic on its output
Metrics: ECE raw vs A vs B, Brier, risk-coverage at P>=0.5 abstain.

ENV: N_RUNS (1200), WORKERS (4).
"""
import json, os, sys, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import simforge_v2 as V2                 # noqa: E402
from shorcm import peakshor as PS                    # noqa: E402
from shorcm import splits as SP                      # noqa: E402

N_RUNS = int(os.environ.get("N_RUNS", 1200))
WORKERS = int(os.environ.get("WORKERS", 4))
OUT = Path("experiments/confidence")
FEATS = ["margin12", "margin13", "score1", "softmax", "sheet", "s1x_log",
         "snap_frac", "pens", "octave_arb"]


def one(i):
    m, x = V2.sample_run(i)
    est = PS.estimate_speed_sheet(x, V2.FS, V2.kinematic_sheet(m),
                                  meta={"component": "motor"})
    est = [c for c in est if np.isfinite(c["hz"])]
    if not est:
        return None
    s = [c.get("score", 0.0) for c in est] + [-99.0, -99.0]
    ev = est[0].get("ev", {})
    return {"run_id": i, "f_true": m["f_shaft"],
            "correct": abs(est[0]["hz"] / m["f_shaft"] - 1) <= 0.01,
            "margin12": s[0] - s[1], "margin13": s[0] - s[2],
            "score1": s[0], "softmax": est[0]["confidence"],
            "sheet": ev.get("sheet", 0.0),
            "s1x_log": float(np.log1p(ev.get("s1x", 0.0))),
            "snap_frac": ev.get("snap_frac", 0.0),
            "pens": (ev.get("alias_pen", 0) + ev.get("q2_pen", 0)
                     + ev.get("elec_pen", 0)),
            "octave_arb": ev.get("octave_arb", 0.0)}


def ece(p, y, bins=10):
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for k in range(bins):
        m = (p >= edges[k]) & (p < edges[k + 1] + (k == bins - 1))
        if m.sum():
            e += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return float(e)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with Pool(WORKERS, maxtasksperchild=100) as pool:
        for r in pool.imap_unordered(one, range(N_RUNS), chunksize=8):
            if r:
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "records.parquet", index=False)
    band = SP.speed_band(df.f_true.values)
    fit, hold = band % 2 == 0, band % 2 == 1
    y = df.correct.values.astype(float)

    from sklearn.isotonic import IsotonicRegression
    import lightgbm as lgb
    res = {"n": len(df), "n_fit": int(fit.sum()),
           "n_holdout": int(hold.sum()),
           "top1_rate": round(float(y.mean()), 3),
           "ece_softmax": round(ece(df.softmax.values[hold], y[hold]), 4)}
    isoA = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    isoA.fit(df.margin12.values[fit], y[fit])
    pA = isoA.predict(df.margin12.values[hold])
    res["ece_iso_margin"] = round(ece(pA, y[hold]), 4)
    mdl = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05,
                             num_leaves=15, min_child_samples=30,
                             random_state=20260709, verbose=-1)
    mdl.fit(df.loc[fit, FEATS], y[fit])
    raw = mdl.predict_proba(df.loc[fit, FEATS])[:, 1]
    isoB = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    isoB.fit(raw, y[fit])
    pB = isoB.predict(mdl.predict_proba(df.loc[hold, FEATS])[:, 1])
    res["ece_lgbm_iso"] = round(ece(pB, y[hold]), 4)
    res["brier_lgbm_iso"] = round(float(np.mean((pB - y[hold]) ** 2)), 4)
    for thr in (0.5, 0.7):
        sel = pB >= thr
        res[f"emit_p{int(thr*100)}"] = {
            "coverage": round(float(sel.mean()), 3),
            "accuracy": (round(float(y[hold][sel].mean()), 3)
                         if sel.sum() else None)}
    (OUT / "findings.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(f"wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
