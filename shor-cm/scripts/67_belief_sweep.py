#!/usr/bin/env python3
"""SpeedBelief offline parameter sweep (the CUSUM discipline: collect
per-record candidate series ONCE on dev seeds, sweep parameters
offline, select, then confirm once on the untouched eval protocol).

Stage A (STAGE=collect): 48 weak-1x dev fleets (56M seed block —
disjoint from the 55M eval block) x 16 records x 2 sensors -> pickle
of candidate lists + truths + nameplates.
Stage B (STAGE=sweep): grid over (prior_sigma, kin_scale, temper,
use_reliability, flip_ratio) evaluated on the stored series.
Selection: max overall map-hit subject to false corrections <=
1/3 of good corrections AND no-1x-phase hit >= per-record baseline.

ENV: STAGE (collect|sweep|both), WORKERS (4), N (48).
"""
import itertools, json, os, pickle, sys, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import simforge_v2 as V2                 # noqa: E402
from shorcm import simforge_mc as MC                 # noqa: E402
from shorcm import peakshor as PS                    # noqa: E402
from shorcm.speedbelief import SpeedBelief           # noqa: E402

STAGE = os.environ.get("STAGE", "both")
WORKERS = int(os.environ.get("WORKERS", 4))
N = int(os.environ.get("N", 48))
OUT = Path("experiments/speedbelief")
SERIES = OUT / "dev_series.pkl"


def collect_one(i):
    rng0 = V2.rng_for_run((56_000_000, i))
    m = V2.sample_machine(rng0)
    if m["fault"] == "healthy":
        m["fault"] = str(rng0.choice(
            ["imbalance", "misalignment", "looseness"]))
        m.setdefault("subtype", None)
        m["fault_shaft"] = "in"
    f_base = m["f_shaft"]
    f_nom = f_base * (1 + rng0.normal(0, 0.03))
    sheet = V2.kinematic_sheet(m)
    recs = []
    for t in range(16):
        rng = V2.rng_for_run((56_500_000 + i, t))
        mm = dict(m)
        if t < 6:
            mm["fault"], mm["severity"] = "healthy", 0.0
            om = 0.05
        else:
            mm["severity"] = float(np.clip(
                0.15 + 0.85 * (t - 6) / 9, 0.05, 1.0))
            om = 1.0
        X = MC.synth_run_mc(mm, rng, omega1x=om)
        sens = {}
        for j, name in ((0, "DE"), (2, "NDE")):
            est = PS.estimate_speed_sheet(X[:, j], MC.FS, sheet)
            sens[name] = [{"hz": float(c["hz"]),
                           "confidence": float(c["confidence"])}
                          for c in est if np.isfinite(c["hz"])]
        recs.append(sens)
    return {"machine": i, "f_base": f_base, "f_nom": f_nom,
            "records": recs}


def replay(series, cfg):
    hits = hits_pre = base = base_pre = n = n_pre = 0
    good = bad = 0
    for fl in series:
        sb = SpeedBelief(f_nom=fl["f_nom"], **cfg)
        for t, sens in enumerate(fl["records"]):
            out = sb.update(sens)
            hit = abs(out["map_hz"] / fl["f_base"] - 1) < 0.03
            b = sens["DE"][0]["hz"] if sens["DE"] else np.nan
            bh = bool(np.isfinite(b)
                      and abs(b / fl["f_base"] - 1) < 0.03)
            n += 1
            hits += hit
            base += bh
            if t < 6:
                n_pre += 1
                hits_pre += hit
                base_pre += bh
            if out["correction"]:
                if abs(out["correction"]["to_hz"] / fl["f_base"] - 1) \
                        < 0.04:
                    good += 1
                else:
                    bad += 1
    return {"hit": hits / n, "hit_pre": hits_pre / n_pre,
            "base": base / n, "base_pre": base_pre / n_pre,
            "corr_good": good, "corr_bad": bad}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if STAGE in ("collect", "both"):
        with Pool(WORKERS, maxtasksperchild=6) as pool:
            series = list(pool.imap_unordered(collect_one, range(N),
                                              chunksize=1))
        pickle.dump(series, open(SERIES, "wb"))
        print("collected", len(series), flush=True)
    if STAGE in ("sweep", "both"):
        series = pickle.load(open(SERIES, "rb"))
        grid = list(itertools.product(
            (0.25, 0.35, 0.5),           # prior_sigma_oct
            (0.5, 1.0, 1.5),             # kin_scale
            (0.3, 0.6, 1.2),             # temper
            (True, False),               # use_reliability
            (4.0, 8.0, 15.0)))           # flip_ratio
        rows = []
        for ps_, ks, tp, ur, fr in grid:
            cfg = {"prior_sigma_oct": ps_, "kin_scale": ks,
                   "temper": tp, "use_reliability": ur,
                   "flip_ratio": fr}
            r = replay(series, cfg)
            r.update(cfg)
            rows.append(r)
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_parquet(OUT / "sweep.parquet", index=False)
        ok = df[(df.corr_bad <= df.corr_good / 3)
                & (df.hit_pre >= df.base_pre)]
        best = (ok if len(ok) else df).sort_values(
            "hit", ascending=False).iloc[0]
        find = {"n_configs": len(df),
                "n_admissible": int(len(ok)),
                "baseline": {"hit": round(float(df.base.iloc[0]), 3),
                             "hit_pre": round(float(
                                 df.base_pre.iloc[0]), 3)},
                "best": {k: (round(float(v), 3)
                             if isinstance(v, (int, float, np.floating))
                             else bool(v))
                         for k, v in best.items()},
                "wall_s": round(time.time() - t0, 1)}
        (OUT / "sweep_findings.json").write_text(
            json.dumps(find, indent=1))
        print(json.dumps(find, indent=1))


if __name__ == "__main__":
    main()
