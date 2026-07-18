#!/usr/bin/env python3
"""SpeedBelief evaluation.

A. SYNTHETIC weak-1x fleets (the CWRU story rendered): 40 machines x
   16 records; records 0-5 healthy with 1x nearly absent
   (omega1x=0.05), fault grows from record 6; nameplate = truth +/- 3%
   noise; 3 sensors (MC channels). Compare per-record frame hit
   (within 3%, octave-strict) for: per-record estimator top-1,
   SpeedBelief MAP. Count corrections (fired / landed correct /
   false).

B. CWRU cross-sensor supplementary: all 64 records in RPM order as ONE
   asset history, DE + FE candidate lists per record, nameplate 29.5.
   Metrics: per-record MAP hit vs file RPM (blind C1 was 0.0).

ENV: WORKERS (4), N_A (40).
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
from shorcm.speedbelief import SpeedBelief           # noqa: E402

WORKERS = int(os.environ.get("WORKERS", 4))
N_A = int(os.environ.get("N_A", 40))
OUT = Path("experiments/speedbelief")


def one_fleet(i):
    rng0 = V2.rng_for_run((55_000_000, i))
    m = V2.sample_machine(rng0)
    if m["fault"] == "healthy":
        m["fault"] = str(rng0.choice(
            ["imbalance", "misalignment", "looseness"]))
        m.setdefault("subtype", None)
        m["fault_shaft"] = "in"
    f_base = m["f_shaft"]
    f_nom = f_base * (1 + rng0.normal(0, 0.03))
    sheet = V2.kinematic_sheet(m)
    sb = SpeedBelief(f_nom=f_nom)
    base_hits, map_hits = [], []
    corrections = []
    for t in range(16):
        rng = V2.rng_for_run((55_500_000 + i, t))
        mm = dict(m)
        if t < 6:
            mm["fault"], mm["severity"] = "healthy", 0.0
            om = 0.05                   # 1x nearly absent
        else:
            mm["severity"] = float(np.clip(
                0.15 + 0.85 * (t - 6) / 9, 0.05, 1.0))
            om = 1.0
        X = MC.synth_run_mc(mm, rng, omega1x=om)
        sens = {}
        for j, name in ((0, "DE"), (2, "NDE")):
            est = PS.estimate_speed_sheet(X[:, j], MC.FS, sheet)
            sens[name] = [c for c in est if np.isfinite(c["hz"])]
        out = sb.update(sens)
        b = sens["DE"][0]["hz"] if sens["DE"] else np.nan
        base_hits.append(bool(np.isfinite(b)
                              and abs(b / f_base - 1) < 0.03))
        map_hits.append(bool(abs(out["map_hz"] / f_base - 1) < 0.03))
        if out["correction"]:
            corrections.append(
                (t, bool(abs(out["correction"]["to_hz"] / f_base - 1)
                         < 0.04)))
    return {"machine": i, "fault": m["fault"],
            "base_hit": float(np.mean(base_hits)),
            "map_hit": float(np.mean(map_hits)),
            "base_hit_pre": float(np.mean(base_hits[:6])),
            "map_hit_pre": float(np.mean(map_hits[:6])),
            "n_corr": len(corrections),
            "corr_good": sum(1 for _, ok in corrections if ok),
            "corr_bad": sum(1 for _, ok in corrections if not ok)}


def cwru_belief():
    import importlib.util, scipy.io as sio
    spec = importlib.util.spec_from_file_location(
        "g", str(Path(__file__).parent / "70_cwru_gate.py"))
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    runs = g.inventory()
    rows = []
    for path, lab, sub, size, fs in runs:
        m = sio.loadmat(str(path))
        de = next((k for k in m if k.endswith("DE_time")), None)
        fe = next((k for k in m if k.endswith("FE_time")), None)
        rpmk = next((k for k in m if k.endswith("RPM")), None)
        rpm = float(np.asarray(m[rpmk]).ravel()[0]) if rpmk else np.nan
        rows.append((path.stem, rpm / 60.0 if np.isfinite(rpm)
                     else np.nan,
                     np.asarray(m[de], float).ravel(),
                     np.asarray(m[fe], float).ravel()
                     if fe else None))
    rows.sort(key=lambda r: (r[1] if np.isfinite(r[1]) else 29.5))
    sb = SpeedBelief(f_nom=29.5)
    hits = []
    for name, f_true, xde, xfe in rows:
        sens = {"DE": [c for c in PS.estimate_speed_sheet(
            xde, 12000.0, {}) if np.isfinite(c["hz"])]}
        if xfe is not None:
            sens["FE"] = [c for c in PS.estimate_speed_sheet(
                xfe, 12000.0, {}) if np.isfinite(c["hz"])]
        out = sb.update(sens)
        if np.isfinite(f_true):
            hits.append(bool(abs(out["map_hz"] / f_true - 1) < 0.03))
    return {"n": len(hits), "map_hit": round(float(np.mean(hits)), 3),
            "final_map_hz": round(out["map_hz"], 2),
            "final_p": out["p_map"],
            "reliability": out["reliability"]}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(WORKERS, maxtasksperchild=8) as pool:
        rows = list(pool.imap_unordered(one_fleet, range(N_A),
                                        chunksize=1))
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "synthetic.parquet", index=False)
    find = {"synthetic": {
        "base_hit": round(float(df.base_hit.mean()), 3),
        "belief_hit": round(float(df.map_hit.mean()), 3),
        "base_hit_no1x_phase": round(float(df.base_hit_pre.mean()), 3),
        "belief_hit_no1x_phase": round(float(df.map_hit_pre.mean()), 3),
        "corrections": {"fired": int(df.n_corr.sum()),
                        "landed_correct": int(df.corr_good.sum()),
                        "false": int(df.corr_bad.sum())}}}
    find["cwru_supplementary"] = cwru_belief()
    find["wall_s"] = round(time.time() - t0, 1)
    (OUT / "findings.json").write_text(json.dumps(find, indent=1))
    print(json.dumps(find, indent=1))


if __name__ == "__main__":
    main()
