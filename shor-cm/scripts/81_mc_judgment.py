#!/usr/bin/env python3
"""Multi-channel (v2.1) judgment: cross-channel peak fusion for speed,
axial-ratio evidence for misalignment subtype.

Arms on N mc machines:
  de     sheet estimator on the radial drive-end channel alone
  fused  sheet estimator on cross-channel-confirmed peaks (>=2 of 3)
Subtype: among misalignment machines, angular-vs-(parallel|coupling)
separability of ax_ratio_2; threshold fitted on even machines,
accuracy on odd.
ENV: N_RUNS (600), WORKERS (4).
"""
import json, os, sys, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import simforge_v2 as V2                 # noqa: E402
from shorcm import simforge_mc as MC                 # noqa: E402
from shorcm import peakshor as PS                    # noqa: E402

N_RUNS = int(os.environ.get("N_RUNS", 600))
WORKERS = int(os.environ.get("WORKERS", 4))
TOL = 0.01
OUT = Path("experiments/mc")


def one(i):
    m, X = MC.sample_run_mc(i)
    f0 = m["f_shaft"]
    sheet = V2.kinematic_sheet(m)
    out = {"run_id": i, "archetype": m["archetype"], "fault": m["fault"],
           "subtype": m["subtype"], "severity": m["severity"],
           "f_true": f0}
    for arm, kw in (("de", {}),
                    ("fused", {"peaks": MC.fused_peaks(X, MC.FS)})):
        est = PS.estimate_speed_sheet(X[:, 0], MC.FS, sheet,
                                      meta={"component": "motor"}, **kw)
        est = [c for c in est if np.isfinite(c["hz"])] or \
            [{"hz": 0.0, "confidence": 0}]
        errs = [abs(c["hz"] / f0 - 1) for c in est]
        out[f"{arm}_top1"] = errs[0] <= TOL
        out[f"{arm}_top3"] = min(errs) <= TOL
        out[f"{arm}_f_hat"] = est[0]["hz"]
    if m["fault"] == "misalignment":
        af = MC.axial_features(X, MC.FS,
                               out["fused_f_hat"] if out["fused_top1"]
                               else f0)
        out.update(af)
    return out


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with Pool(WORKERS, maxtasksperchild=100) as pool:
        for r in pool.imap_unordered(one, range(N_RUNS), chunksize=8):
            rows.append(r)
            if len(rows) % 100 == 0:
                print(f"{len(rows)}/{N_RUNS} {time.time()-t0:.0f}s",
                      flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "results.parquet", index=False)
    find = {"n": len(df)}
    for arm in ("de", "fused"):
        find[arm] = {"top1": round(float(df[f"{arm}_top1"].mean()), 4),
                     "top3": round(float(df[f"{arm}_top3"].mean()), 4)}
    mis = df[(df.fault == "misalignment")].dropna(subset=["ax_ratio_2"])
    if len(mis) > 30:
        from sklearn.metrics import roc_auc_score
        y = (mis.subtype == "angular").astype(int)
        find["subtype"] = {
            "n_misalign": len(mis),
            "auc_angular_vs_rest": round(float(
                roc_auc_score(y, mis.ax_ratio_2)), 3)}
        fit = mis.index % 2 == 0
        thr_grid = np.quantile(mis.ax_ratio_2[fit], np.linspace(.1, .9, 33))
        best_t = max(thr_grid, key=lambda t: (
            ((mis.ax_ratio_2[fit] > t).astype(int) == y[fit]).mean()))
        acc = (((mis.ax_ratio_2[~fit] > best_t).astype(int)
                == y[~fit]).mean())
        find["subtype"]["holdout_accuracy"] = round(float(acc), 3)
        find["subtype"]["threshold"] = round(float(best_t), 3)
    find["wall_s"] = round(time.time() - t0, 1)
    (OUT / "findings.json").write_text(json.dumps(find, indent=1))
    print(json.dumps(find, indent=1))


if __name__ == "__main__":
    main()
