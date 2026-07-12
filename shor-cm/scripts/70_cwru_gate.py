#!/usr/bin/env python3
"""CWRU REAL-DATA GATE — frozen cascade, API only, no refitting.

Data: full canonical CWRU bearing benchmark (git-cloned mirror
s-whynot/CWRU-dataset). Drive-end accelerometer channel, nameplate RPM
in each file. Drive-end bearing SKF 6205-2RS published defect orders:
BPFO 3.5848, BPFI 5.4152, BSF 2.3567 (x shaft).

PRE-REGISTERED (before first run):
 C1 speed: blind f_hat within 3% of file RPM/60; hit >= 0.60 overall
 C2 detect: bearing-call (ml or rules) recall >= 0.50 on fault files
    AND 0 bearing calls on the 4 Normal files
 C3 patterns: envelope capture (near_pattern incl. HARM base and 2x)
    at the subtype's true defect order on >= 0.60 of 12k DE fault
    files per subtype (IR / OR / B), 0 captures at BOTH probe orders
    on Normal
 C4 severity ordinality: frozen severity_score increases with fault
    size — Spearman(size, score) > 0.3 within IR and within OR
    (12k DE, sizes 007/014/021)

ENV: WORKERS (4).
"""
import json, os, re, sys, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import patterns as PT                    # noqa: E402
from shorcm.cascade import Cascade                   # noqa: E402

WORKERS = int(os.environ.get("WORKERS", 4))
OUT = Path("experiments/cwru")
ROOT = Path("data/cwru")
ORDERS = {"IR": 5.4152, "OR": 3.5848, "B": 2.3567}

_c = None


def casc():
    global _c
    if _c is None:
        _c = Cascade.load("models")
    return _c


def near_pattern(pats, order, tol=0.06):
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


def inventory():
    runs = []
    for f in sorted(ROOT.rglob("*.mat")):
        rel = str(f.relative_to(ROOT))
        if rel.startswith("Normal"):
            runs.append((f, "normal", None, 0, 12000))
        elif "12k_Drive_End" in rel:
            mlab = re.match(r".*/(IR|OR|B)/(\d{3})/", rel + "/")
            parts = f.relative_to(ROOT).parts
            sub, size = parts[1], parts[2]
            if sub in ORDERS:
                runs.append((f, "fault", sub, int(size), 12000))
    return runs


def one(args):
    path, lab, sub, size, fs = args
    import scipy.io as sio
    m = sio.loadmat(str(path))
    de = next((k for k in m if k.endswith("DE_time")), None)
    rpmk = next((k for k in m if k.endswith("RPM")), None)
    if de is None:
        return None
    x = np.asarray(m[de], float).ravel()
    rpm = float(np.asarray(m[rpmk]).ravel()[0]) if rpmk else np.nan
    f_true = rpm / 60.0 if np.isfinite(rpm) else np.nan
    out = casc().analyze_record(x, float(fs), {})
    row = {"record": path.stem, "label": lab, "subtype": sub,
           "size": size, "f_true": f_true,
           "f_hat": out.get("speed_hz"),
           "fault_ml": out.get("fault_ml"),
           "fault_rules": out.get("fault_rules"),
           "severity": out.get("severity_score")}
    try:
        f_ref = f_true if np.isfinite(f_true) else out.get("speed_hz")
        pats_e, _, _ = PT.decompose_envelope(x, float(fs), f_ref)
        for k, o in ORDERS.items():
            row[f"env_{k}"] = near_pattern(pats_e, o)
    except Exception:
        for k in ORDERS:
            row[f"env_{k}"] = None
    return row


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    runs = inventory()
    with Pool(WORKERS, maxtasksperchild=8) as pool:
        rows = [r for r in pool.imap_unordered(one, runs, chunksize=1)
                if r]
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "records.parquet", index=False)

    f = {"n": len(df)}
    ok1 = (df.f_hat / df.f_true - 1).abs() <= 0.03
    f["C1_speed_hit"] = round(float(ok1.mean()), 3)
    flt = df.label == "fault"
    bcall = ((df.fault_ml == "bearing") | (df.fault_rules == "bearing"))
    f["C2_bearing_recall"] = round(float(bcall[flt].mean()), 3)
    f["C2_bearing_calls_normal"] = int(bcall[~flt].sum())
    f["C3_env_by_subtype"] = {}
    for sub, o in ORDERS.items():
        d = df[(df.subtype == sub)]
        if len(d):
            f["C3_env_by_subtype"][sub] = round(float(
                d[f"env_{sub}"].mean()), 3)
    f["C3_normal_fp"] = int((df[~flt].env_IR & df[~flt].env_OR).sum())
    from scipy.stats import spearmanr
    f["C4_severity_spearman"] = {}
    for sub in ("IR", "OR"):
        d = df[(df.subtype == sub) & df["size"].isin((7, 14, 21))]
        if len(d) >= 6:
            f["C4_severity_spearman"][sub] = round(float(
                spearmanr(d["size"], d.severity).statistic), 3)
    f["verdicts"] = {
        "C1": bool(f["C1_speed_hit"] >= 0.60),
        "C2": bool(f["C2_bearing_recall"] >= 0.50
                   and f["C2_bearing_calls_normal"] == 0),
        "C3": bool(all(v >= 0.60
                       for v in f["C3_env_by_subtype"].values())
                   and f["C3_normal_fp"] == 0),
        "C4": bool(all(v > 0.3
                       for v in f["C4_severity_spearman"].values()))}
    f["wall_s"] = round(time.time() - t0, 1)
    (OUT / "findings.json").write_text(json.dumps(f, indent=1))
    print(json.dumps(f, indent=1))


if __name__ == "__main__":
    main()
