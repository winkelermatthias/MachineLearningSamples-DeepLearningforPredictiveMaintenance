#!/usr/bin/env python3
"""Blind end-to-end real gates (PHASE3 A1): the composed cascade —
speed -> patterns -> fault — with NO oracle speed anywhere. Truth
(nameplate/tacho metadata) is used on the judgment side only.

ENV: MODELS=models|models_v6  TAG=v5|v6
Outputs experiments/blind_gates/<TAG>/findings.json with Wilson CIs
(A9) and matched false-positive rates (A11).

Blind envelope capture (the M1 fix): a defect is "captured" iff some
envelope pattern's absolute frequency (pattern order x ESTIMATED
speed) lands within 6% of the true defect frequency. No true speed
enters the pipeline.
"""
import json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm.cascade import Cascade                    # noqa: E402
from shorcm import adapters as AD                     # noqa: E402

MODELS = os.environ.get("MODELS", "models")
TAG = os.environ.get("TAG", "v5")
OUT = Path(f"experiments/blind_gates/{TAG}")
CWRU_ORDERS = {"BPFO": 3.585, "BPFI": 5.415, "BSF": 2.357}
CWRU_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
F2O = {"IR": "BPFI", "OR": "BPFO", "B": "BSF"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(p, 3), round(c - h, 3), round(c + h, 3))


def speed_hits(f_hat, f_true):
    if not (np.isfinite(f_hat) and f_hat > 0):
        return False, False
    hit5 = abs(f_hat / f_true - 1) < 0.05
    octv = any(abs(f_hat / (k * f_true) - 1) < 0.05
               for k in (0.25, 1 / 3, 0.5, 1, 2, 3, 4))
    return bool(hit5), bool(octv)


def env_capture(out, f_true, orders):
    """Blind capture: envelope pattern order x est speed vs truth Hz."""
    f_hat = out.get("speed_hz", np.nan)
    if not (np.isfinite(f_hat) and f_hat > 0):
        return {k: False for k in orders}
    got = {}
    pats = out.get("patterns_envelope", [])
    for name, o in orders.items():
        f_t = o * f_true
        hit = False
        for p in pats:
            po = p["params"].get("order", p["params"].get("base"))
            if po is None:
                continue
            f_p = float(po) * f_hat
            if abs(f_p / f_t - 1) < 0.06 or abs(f_p / (2 * f_t) - 1) < 0.06:
                hit = True
                break
        got[name] = hit
    return got


def run_record(casc, x, fs, f_true, orders):
    t0 = time.time()
    out = casc.analyze_record(np.asarray(x, float), fs)
    hit5, octv = speed_hits(out.get("speed_hz", np.nan), f_true)
    r = {"speed_hz": out.get("speed_hz"),
         "speed_conf": out.get("speed_confidence"),
         "abstain_speed": out.get("abstain_speed"),
         "fault_ml": out.get("fault_ml", "unknown"),
         "fault_rules": out.get("fault_rules", "unknown"),
         "abstain_fault": out.get("abstain_fault"),
         "severity": out.get("severity_score"),
         "speed_hit5": hit5, "speed_octave": octv,
         "wall_s": round(time.time() - t0, 2)}
    if orders:
        r["env_capture"] = env_capture(out, f_true, orders)
    return r


def cwru(casc):
    import scipy.io as sio
    rows = []
    root = Path("data/cwru/12k_Drive_End_Bearing_Fault_Data")
    files = [(p, p.parts[-3]) for p in sorted(root.rglob("*.mat"))]
    files += [(p, "Normal") for p in
              sorted(Path("data/cwru/Normal").glob("*.mat"))]
    for p, cls in files:
        m = sio.loadmat(p, squeeze_me=True)
        key = [k for k in m if k.endswith("DE_time")]
        if not key:
            continue
        x = np.asarray(m[key[0]], float).ravel()
        fs = 48000.0 if cls == "Normal" else 12000.0
        load = int(p.stem.split("_")[-1]) if "_" in p.stem else 0
        f_true = CWRU_RPM.get(load, 1750) / 60.0
        r = run_record(casc, x, fs, f_true,
                       CWRU_ORDERS if cls != "Normal" else None)
        r.update({"file": p.stem, "class": cls})
        rows.append(r)
    fault = [r for r in rows if r["class"] != "Normal"]
    norm = [r for r in rows if r["class"] == "Normal"]
    n_b = sum(r["fault_ml"] == "bearing" for r in fault)
    cap = sum(r["env_capture"][F2O[r["class"]]] for r in fault)
    return {"n_fault": len(fault), "n_normal": len(norm),
            "speed_hit5": wilson(sum(r["speed_hit5"] for r in rows),
                                 len(rows)),
            "speed_octave": wilson(sum(r["speed_octave"] for r in rows),
                                   len(rows)),
            "bearing_recall_ml": wilson(n_b, len(fault)),
            "blind_env_capture_truth_order": wilson(cap, len(fault)),
            "normal_bearing_fp":
                f"{sum(r['fault_ml'] == 'bearing' for r in norm)}"
                f"/{len(norm)} (ANECDOTE)",
            "fault_abstain_rate": wilson(
                sum(bool(r["abstain_fault"]) for r in fault),
                len(fault)),
            "rows": rows}


def mfpt(casc):
    rows = []
    for p in sorted(Path("data/mfpt").glob("*.mat")):
        x, fs, _, meta = AD.from_mfpt_mat(p)
        cls = ("baseline" if "baseline" in p.stem
               else "OR" if "Outer" in p.stem else "IR")
        r = run_record(casc, x, fs, meta["rate_hz"],
                       AD.MFPT_ORDERS if cls != "baseline" else None)
        r.update({"file": p.stem, "class": cls})
        rows.append(r)
    fault = [r for r in rows if r["class"] != "baseline"]
    base = [r for r in rows if r["class"] == "baseline"]
    exp = {"OR": "BPFO", "IR": "BPFI"}
    cap = sum(r["env_capture"][exp[r["class"]]] for r in fault)
    return {"n_fault": len(fault), "n_baseline": len(base),
            "speed_hit5": f"{wilson(sum(r['speed_hit5'] for r in rows), len(rows))} (ANECDOTE)",
            "bearing_recall_ml":
                f"{sum(r['fault_ml'] == 'bearing' for r in fault)}"
                f"/{len(fault)} (ANECDOTE)",
            "blind_env_capture_truth_order": f"{cap}/{len(fault)}",
            "baseline_bearing_fp":
                f"{sum(r['fault_ml'] == 'bearing' for r in base)}"
                f"/{len(base)}",
            "rows": rows}


def seu(casc):
    rows = []
    for p in sorted(Path("data/seu").glob("*.csv")):
        x, fs, _, meta = AD.from_seu_csv(p, dur=8.0)
        stem = p.stem
        truth = ("healthy" if "health" in stem.lower() else
                 "bearing" if stem.startswith("bearingset") else "gear")
        r = run_record(casc, x, fs, meta["rate_hz"], None)
        r.update({"file": stem, "truth": truth})
        rows.append(r)
    fault = [r for r in rows if r["truth"] != "healthy"]
    heal = [r for r in rows if r["truth"] == "healthy"]
    ok = sum(r["fault_ml"] == r["truth"] for r in fault)
    return {"n_fault": len(fault), "n_healthy": len(heal),
            "speed_hit5": f"{wilson(sum(r['speed_hit5'] for r in rows), len(rows))} (ANECDOTE)",
            "fault_class_correct": f"{ok}/{len(fault)} (ANECDOTE)",
            "healthy_fault_fp":
                f"{sum(r['fault_ml'] not in ('healthy', 'unknown') for r in heal)}"
                f"/{len(heal)}",
            "rows": rows}


def windturbine(casc):
    rows = []
    for i, p in enumerate(sorted(
            Path("data/windturbine").glob("data-*.mat"))):
        x, fs, _, meta = AD.from_windturbine_mat(p)
        f_true = meta["tach_pulse_hz"]                # octave class only
        r = run_record(casc, x, fs, f_true, None)
        r.update({"file": p.stem, "day": i + 1})
        rows.append(r)
    early = [r for r in rows if r["day"] <= 10]
    late = [r for r in rows if r["day"] > 40]
    return {"n_days": len(rows),
            "speed_octave_vs_tach": wilson(
                sum(r["speed_octave"] for r in rows), len(rows)),
            "bearing_calls_days_1_10":
                f"{sum(r['fault_ml'] == 'bearing' for r in early)}/10",
            "bearing_calls_days_41_50":
                f"{sum(r['fault_ml'] == 'bearing' for r in late)}/10",
            "note": "single run-to-failure sequence: ANECDOTE",
            "rows": rows}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    casc = Cascade.load(MODELS)
    res = {"models": MODELS, "tag": TAG}
    for name, fn in (("cwru", cwru), ("mfpt", mfpt), ("seu", seu),
                     ("windturbine", windturbine)):
        try:
            res[name] = fn(casc)
            print(f"{name} done {time.time()-t0:.0f}s", flush=True)
        except Exception as e:                        # keep going
            res[name] = {"error": repr(e)}
            print(f"{name} ERROR {e!r}", flush=True)
    res["wall_s"] = round(time.time() - t0, 1)
    (OUT / "findings.json").write_text(json.dumps(res, indent=2))
    slim = {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
            if isinstance(v, dict) else v for k, v in res.items()}
    print(json.dumps(slim, indent=2))


if __name__ == "__main__":
    main()
