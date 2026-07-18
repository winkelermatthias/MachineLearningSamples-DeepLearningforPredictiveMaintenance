#!/usr/bin/env python3
"""Kinematic-sheet-conditioned speed on the geared/belt/planetary subset
of the v2 corpus (bit-identical seeds). Compares:
  plain   estimate_speed_shor (no sheet)
  sheet   estimate_speed_sheet (ratio/mesh/passage/planetary known)
ENV: N_RUNS (2000 scan), WORKERS (4).
"""
import os, sys, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import simforge_v2 as V2                 # noqa: E402
from shorcm import peakshor as PS                    # noqa: E402

N_RUNS = int(os.environ.get("N_RUNS", 2000))
WORKERS = int(os.environ.get("WORKERS", 4))
TOL = 0.01
GEARED = {"fan_belt", "pump_gearbox1", "fan_gearbox2", "pump_planetary"}
OUT = Path("experiments/v2")


def sheet_of(m):
    s = {}
    if m["archetype"] == "fan_belt":
        s["ratio"] = m["belt_ratio"] * (1 - 0.012)   # nominal, mid slip
        s["passage"] = m.get("blades")
    elif m["archetype"] == "pump_gearbox1":
        s["ratio"] = m["z1"] / m["z2"]
        s["mesh"] = m["z1"]
        s["passage"] = m.get("vanes")
    elif m["archetype"] == "fan_gearbox2":
        s["ratio"] = m["z1"] / m["z2"]
        s["mesh"] = m["z1"]
        s["passage"] = m.get("blades")
    elif m["archetype"] == "pump_planetary":
        s["planetary"] = (m["Zs"], m["Zp"], m["Zr"], m["Np"])
        s["passage"] = m.get("vanes")
    return s


def one(i):
    m2, x = V2.sample_run(i)                 # bit-identical corpus run
    if m2["archetype"] not in GEARED:
        return None
    f0 = m2["f_shaft"]

    def judge(cands):
        cands = [c for c in cands if np.isfinite(c["hz"])] or \
            [{"hz": 0.0, "confidence": 0}]
        errs = [abs(c["hz"] / f0 - 1) for c in cands]
        return errs[0] <= TOL, min(errs) <= TOL

    p1, p3 = judge(PS.estimate_speed_shor(x, V2.FS))
    s1, s3 = judge(PS.estimate_speed_sheet(x, V2.FS, sheet_of(m2)))
    return {"run_id": i, "archetype": m2["archetype"], "fault": m2["fault"],
            "plain_top1": p1, "plain_top3": p3,
            "sheet_top1": s1, "sheet_top3": s3}


def main():
    t0 = time.time()
    rows = []
    with Pool(WORKERS, maxtasksperchild=100) as pool:
        for r in pool.imap_unordered(one, range(N_RUNS), chunksize=8):
            if r:
                rows.append(r)
                if len(rows) % 200 == 0:
                    print(f"{len(rows)} geared runs, "
                          f"{time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "sheet_speed.parquet", index=False)
    print(f"\nn={len(df)} geared/belt/planetary runs, "
          f"{time.time()-t0:.0f}s")
    print(df[["plain_top1", "plain_top3", "sheet_top1", "sheet_top3"]]
          .mean().round(3).to_string())
    print("\nby archetype:")
    print(df.groupby("archetype")[["plain_top1", "sheet_top1",
                                   "plain_top3", "sheet_top3"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
