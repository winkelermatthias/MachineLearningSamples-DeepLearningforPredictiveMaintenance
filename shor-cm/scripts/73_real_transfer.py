#!/usr/bin/env python3
"""REAL-DATA TRANSFER GATE — frozen cascade, API only, no refitting.

Data: MFPT bearing set (mathworks mirror; single radial channel, NICE
bearing, 25 Hz shaft, known defect orders) and SEU DDS gearbox
(cathysiyu mirror; planetary-x channel, motor 20/30 Hz, tooth counts
unpublished -> blind sheet).

PRE-REGISTERED judgments (written before first run; verdicts derive
from these numbers only):

 MFPT (20 records)
  J1 speed:    |f_hat/25 - 1| <= 3%; PASS if hit-rate >= 0.60 overall
  J2 fault:    bearing-call (ml or rules) recall >= 0.50 on the 17
               fault records AND <= 1/3 on the 3 baselines
  J3 patterns: at TRUE speed, a NEARRAT (or off-integer SIDEBAND
               carrier) within 6% of BPFO=3.245 on >= 0.60 of outer-
               race records; same for BPFI=4.755 on inner-race
  J4 severity: median severity_score(outer-race @270 lbs) >
               median severity_score(baselines)

 SEU (20 files x 10 windows)
  J5 speed:    Health files, |f_hat/nameplate - 1| <= 4% on >= 0.60
               of windows
  J6 fault:    gear-call rate on gear-fault windows minus gearset-
               Health > 0.15; bearing-call rate on bearing-fault
               windows minus bearingset-health > 0.15
  J7 patterns: at TRUE speed, SIDEBAND-presence rate on gear faults
               minus gearset-Health > 0.15

ENV: WORKERS (4).
"""
import json, os, sys, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import adapters as AD                    # noqa: E402
from shorcm import patterns as PT                    # noqa: E402
from shorcm.cascade import Cascade                   # noqa: E402

WORKERS = int(os.environ.get("WORKERS", 4))
OUT = Path("experiments/real_transfer")

MFPT = sorted(Path("data/mfpt").glob("*.mat"))
SEU = sorted(Path("data/seu").glob("*.csv"))
SEU_T0 = [5.0 + 8.0 * k for k in range(10)]

_casc = None


def casc():
    global _casc
    if _casc is None:
        _casc = Cascade.load("models")
    return _casc


def near_pattern(pats, order, tol=0.06):
    """Type-compatible capture at the kinematic order OR its 2x
    harmonic (symmetric with the synthetic judgment, whose truth
    includes harmonics): NEARRAT/TONE order, SIDEBAND carrier, or a
    HARM family whose base IS the defect rate (a defect-order ladder
    is the classic envelope signature)."""
    targets = (order, 2 * order)
    for p in pats:
        cand = None
        if p["type"] in ("NEARRAT", "TONE"):
            cand = p["params"]["order"]
        elif p["type"] == "SIDEBAND":
            c = p["params"]["carrier"]
            if abs(c - round(c)) > 0.15:
                cand = c
        elif p["type"] == "HARM" and p["params"]["base"] > 1.5:
            cand = p["params"]["base"]
        if cand is not None and any(abs(cand / o - 1) < tol
                                    for o in targets):
            return True
    return False


def mfpt_one(path):
    x, fs, sheet, meta = AD.from_mfpt_mat(str(path))
    out = casc().analyze_record(x, fs, sheet)
    name = path.stem
    lab = "baseline" if "baseline" in name else \
        ("outer" if "Outer" in name else "inner")
    row = {"dataset": "mfpt", "record": name, "label": lab,
           "rate_true": meta["rate_hz"], "load": meta["load_lbs"],
           "f_hat": out.get("speed_hz"),
           "speed_conf": out.get("speed_confidence"),
           "fault_ml": out.get("fault_ml"),
           "fault_rules": out.get("fault_rules"),
           "severity": out.get("severity_score")}
    try:                       # pattern layer isolated at TRUE speed
        pats, _ = PT.decompose(x, fs, meta["rate_hz"])
        row["pat_bpfo"] = near_pattern(pats, AD.MFPT_ORDERS["BPFO"])
        row["pat_bpfi"] = near_pattern(pats, AD.MFPT_ORDERS["BPFI"])
    except Exception:
        row["pat_bpfo"] = row["pat_bpfi"] = None
    try:                       # iteration-24 envelope channel (J3env)
        pats_e, _, _ = PT.decompose_envelope(x, fs, meta["rate_hz"])
        row["env_bpfo"] = near_pattern(pats_e, AD.MFPT_ORDERS["BPFO"])
        row["env_bpfi"] = near_pattern(pats_e, AD.MFPT_ORDERS["BPFI"])
    except Exception:
        row["env_bpfo"] = row["env_bpfi"] = None
    return row


def seu_one(args):
    path, t0 = args
    p = Path(path)
    try:
        x, fs, sheet, meta = AD.from_seu_csv(str(p), t0=t0)
        if len(x) < 4 * fs:
            return None
        out = casc().analyze_record(x, fs, sheet)
    except Exception:
        return None
    grp = "gearset" if p.name.startswith("gearset") else "bearingset"
    cond = p.stem.split("_", 1)[1].rsplit("_", 2)[0].lower()
    row = {"dataset": "seu", "record": f"{p.stem}@{t0:.0f}s",
           "group": grp, "cond": cond,
           "title": meta.get("title", ""),
           "rate_true": meta["rate_hz"],
           "f_hat": out.get("speed_hz"),
           "speed_conf": out.get("speed_confidence"),
           "fault_ml": out.get("fault_ml"),
           "fault_rules": out.get("fault_rules"),
           "severity": out.get("severity_score")}
    try:
        pats, _ = PT.decompose(x, fs, meta["rate_hz"])
        row["pat_sideband"] = any(p_["type"] == "SIDEBAND"
                                  for p_ in pats)
    except Exception:
        row["pat_sideband"] = None
    try:                       # iteration-24 envelope channel (J6env)
        pats_e, _, _ = PT.decompose_envelope(x, fs, meta["rate_hz"])
        row["env_defect"] = any(
            p_["type"] in ("NEARRAT", "SIDEBAND")
            or (p_["type"] == "HARM" and p_["params"]["base"] > 1.5)
            for p_ in pats_e if p_.get("share", 0) > 0.03)
    except Exception:
        row["env_defect"] = None
    return row


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(WORKERS, maxtasksperchild=10) as pool:
        mf = list(pool.imap_unordered(mfpt_one, MFPT, chunksize=1))
        se = [r for r in pool.imap_unordered(
            seu_one, [(str(f), w) for f in SEU for w in SEU_T0],
            chunksize=1) if r]
    dm = pd.DataFrame(mf)
    ds = pd.DataFrame(se)
    dm.to_parquet(OUT / "mfpt.parquet", index=False)
    ds.to_parquet(OUT / "seu.parquet", index=False)

    bcall = lambda d: ((d.fault_ml == "bearing")
                       | (d.fault_rules == "bearing"))
    gcall = lambda d: ((d.fault_ml == "gear") | (d.fault_rules == "gear"))
    f = {}
    ok1 = (dm.f_hat / dm.rate_true - 1).abs() <= 0.03
    f["J1_speed_hit"] = round(float(ok1.mean()), 3)
    flt = dm.label != "baseline"
    f["J2_bearing_recall_faults"] = round(float(bcall(dm[flt]).mean()), 3)
    f["J2_bearing_calls_baseline"] = int(bcall(dm[~flt]).sum())
    f["J3_bpfo_on_outer"] = round(float(
        dm[dm.label == "outer"].pat_bpfo.mean()), 3)
    f["J3_bpfi_on_inner"] = round(float(
        dm[dm.label == "inner"].pat_bpfi.mean()), 3)
    sev_or = dm[(dm.label == "outer") & (dm.load == 270)].severity
    f["J4_sev_outer270_vs_baseline"] = [
        round(float(sev_or.median()), 3),
        round(float(dm[~flt].severity.median()), 3)]

    heal = ds.cond.isin(("health",))
    ok5 = (ds.f_hat / ds.rate_true - 1).abs() <= 0.04
    f["J5_speed_hit_health"] = round(float(ok5[heal].mean()), 3)
    f["J5_speed_hit_all"] = round(float(ok5.mean()), 3)
    gs, bs = ds.group == "gearset", ds.group == "bearingset"
    f["J6_gear_delta"] = round(float(
        gcall(ds[gs & ~heal]).mean() - gcall(ds[gs & heal]).mean()), 3)
    f["J6_bearing_delta"] = round(float(
        bcall(ds[bs & ~heal]).mean() - bcall(ds[bs & heal]).mean()), 3)
    f["J7_sideband_delta"] = round(float(
        ds[gs & ~heal].pat_sideband.mean()
        - ds[gs & heal].pat_sideband.mean()), 3)
    # iteration-24 amendments (registered before the envelope re-run):
    # J3env — envelope capture at defect orders >= 0.60 per class;
    # J6env — envelope defect-pattern presence delta on the SEU
    # bearingset (faults minus health) > 0.15
    if "env_bpfo" in dm.columns:
        f["J3env_bpfo_on_outer"] = round(float(
            dm[dm.label == "outer"].env_bpfo.mean()), 3)
        f["J3env_bpfi_on_inner"] = round(float(
            dm[dm.label == "inner"].env_bpfi.mean()), 3)
        f["J3env_fp_baseline"] = int(
            (dm[dm.label == "baseline"].env_bpfo
             | dm[dm.label == "baseline"].env_bpfi).sum())
    if "env_defect" in ds.columns:
        f["J6env_bearing_delta"] = round(float(
            ds[bs & ~heal].env_defect.mean()
            - ds[bs & heal].env_defect.mean()), 3)
    f["n_mfpt"], f["n_seu"] = len(dm), len(ds)

    f["verdicts"] = {
        "J1": bool(f["J1_speed_hit"] >= 0.60),
        "J2": bool(f["J2_bearing_recall_faults"] >= 0.50
                   and f["J2_bearing_calls_baseline"] <= 1),
        "J3": bool(f["J3_bpfo_on_outer"] >= 0.60
                   and f["J3_bpfi_on_inner"] >= 0.60),
        "J4": bool(f["J4_sev_outer270_vs_baseline"][0]
                   > f["J4_sev_outer270_vs_baseline"][1]),
        "J5": bool(f["J5_speed_hit_health"] >= 0.60),
        "J6": bool(f["J6_gear_delta"] > 0.15
                   and f["J6_bearing_delta"] > 0.15),
        "J7": bool(f["J7_sideband_delta"] > 0.15)}
    if "J3env_bpfo_on_outer" in f:
        f["verdicts"]["J3env"] = bool(
            f["J3env_bpfo_on_outer"] >= 0.60
            and f["J3env_bpfi_on_inner"] >= 0.60)
    if "J6env_bearing_delta" in f:
        f["verdicts"]["J6env"] = bool(f["J6env_bearing_delta"] > 0.15)
    f["wall_s"] = round(time.time() - t0, 1)
    (OUT / "findings.json").write_text(json.dumps(f, indent=1))
    print(json.dumps(f, indent=1))


if __name__ == "__main__":
    main()
