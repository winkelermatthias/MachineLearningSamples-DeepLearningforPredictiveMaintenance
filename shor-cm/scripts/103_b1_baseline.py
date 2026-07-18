#!/usr/bin/env python3
"""B1 classical baseline (PHASE3 A3): fast-kurtogram + envelope
spectrum peak test at nominal defect orders, nameplate speed.

Datasets:
  CWRU 12k drive-end: 60 fault files (IR/OR/B x sizes x loads)
       + 4 normals. SKF 6205-2RS orders/rev: BPFO 3.585, BPFI 5.415,
       BSF 2.357. Nameplate speed by load class {0..3} ->
       {1797,1772,1750,1730} rpm (motor slip curve = nameplate-level
       knowledge; the cascade does NOT get this — documented asymmetry
       that favors B1).
  MFPT: 20 records (3 baseline / 10 outer / 7 inner), NICE bearing
       orders from shorcm.adapters.MFPT_ORDERS, nameplate rate from
       record metadata.
  SEU: EXCLUDED from B1 defect-order testing — bearing geometry is not
       published with the mirror.
  Wind turbine: classical trend baseline — daily envelope kurtosis +
       band SNR trend; alarm day = first of 3 consecutive days above
       (median + 3*MAD) of days 1-10.

Verdicts + Wilson CIs (A9) + matched FP rates (A11) ->
experiments/b1_baseline/findings.json. All N<30 subgroups are
ANECDOTE by constitution.
"""
import json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import baselines as BL                    # noqa: E402
from shorcm import adapters as AD                     # noqa: E402

OUT = Path("experiments/b1_baseline")
CWRU_ORDERS = {"BPFO": 3.585, "BPFI": 5.415, "BSF": 2.357}
CWRU_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
FAULT_TO_ORDER = {"IR": "BPFI", "OR": "BPFO", "B": "BSF"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(p, 3), round(c - h, 3), round(c + h, 3))


def cwru():
    import scipy.io as sio
    rows = []
    root = Path("data/cwru/12k_Drive_End_Bearing_Fault_Data")
    files = [(p, next(c for c in p.parts if c in ("IR", "OR", "B")))
             for p in sorted(root.rglob("*.mat"))]
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
        f_nom = CWRU_RPM.get(load, 1750) / 60.0
        v = BL.b1_envelope_verdict(x, fs, f_nom, CWRU_ORDERS)
        v.update({"file": p.stem, "class": cls,
                  "truth_order": FAULT_TO_ORDER.get(cls, "")})
        rows.append(v)
    fault = [r for r in rows if r["class"] != "Normal"]
    norm = [r for r in rows if r["class"] == "Normal"]
    det = sum(r["detected"] for r in fault)
    sub_ok = sum(r["detected"] and r["called"] == r["truth_order"]
                 for r in fault)
    fp = sum(r["detected"] for r in norm)
    return {"n_fault": len(fault), "n_normal": len(norm),
            "detection_recall": wilson(det, len(fault)),
            "subtype_correct_given_any": wilson(sub_ok, len(fault)),
            "normal_false_alarms": f"{fp}/{len(norm)} (ANECDOTE n<30)",
            "per_class_recall": {c: wilson(
                sum(r["detected"] for r in fault if r["class"] == c),
                sum(1 for r in fault if r["class"] == c))
                for c in ("IR", "OR", "B")},
            "rows": rows}


def mfpt():
    rows = []
    for p in sorted(Path("data/mfpt").glob("*.mat")):
        x, fs, _, meta = AD.from_mfpt_mat(p)
        cls = ("baseline" if "baseline" in p.stem
               else "OR" if "Outer" in p.stem else "IR")
        v = BL.b1_envelope_verdict(x, fs, meta["rate_hz"],
                                   AD.MFPT_ORDERS)
        v.update({"file": p.stem, "class": cls})
        rows.append(v)
    fault = [r for r in rows if r["class"] != "baseline"]
    base = [r for r in rows if r["class"] == "baseline"]
    det = sum(r["detected"] for r in fault)
    exp = {"OR": "BPFO", "IR": "BPFI"}
    sub = sum(r["detected"] and r["called"] == exp[r["class"]]
              for r in fault)
    return {"n_fault": len(fault), "n_baseline": len(base),
            "detection_recall":
                f"{wilson(det, len(fault))} (ANECDOTE n<30)",
            "subtype_correct": f"{sub}/{len(fault)}",
            "baseline_false_alarms":
                f"{sum(r['detected'] for r in base)}/{len(base)}",
            "rows": rows}


def windturbine():
    from scipy.signal import butter, sosfilt, hilbert
    from scipy.stats import kurtosis
    days = []
    for p in sorted(Path("data/windturbine").glob("data-*.mat")):
        x, fs, _, _ = AD.from_windturbine_mat(p)
        band, k = BL.fast_kurtogram_band(x, fs)
        sos = butter(4, [band[0] / (fs / 2),
                         min(band[1] / (fs / 2), 0.999)],
                     btype="band", output="sos")
        env = np.abs(hilbert(sosfilt(sos, x)))
        days.append({"file": p.stem, "band_kurt": round(float(k), 3),
                     "env_kurt": round(float(kurtosis(env)), 3),
                     "raw_kurt": round(float(kurtosis(x)), 3)})
    sig = np.array([d["band_kurt"] for d in days])
    ref = sig[:10]
    thr = np.median(ref) + 3 * 1.4826 * np.median(
        np.abs(ref - np.median(ref)))
    above = sig > thr
    alarm = next((i + 1 for i in range(len(above) - 2)
                  if above[i] and above[i + 1] and above[i + 2]), None)
    return {"n_days": len(days), "threshold": round(float(thr), 3),
            "alarm_day_first_of_3_consecutive": alarm,
            "note": "single sequence: ANECDOTE by constitution",
            "days": days}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"cwru": cwru(), "mfpt": mfpt(), "windturbine": windturbine(),
           "seu": "excluded: bearing geometry unpublished in mirror",
           "wall_s": round(time.time() - t0, 1)}
    (OUT / "findings.json").write_text(json.dumps(res, indent=2))
    slim = {k: {kk: vv for kk, vv in v.items()
                if kk not in ("rows", "days")}
            if isinstance(v, dict) else v for k, v in res.items()}
    print(json.dumps(slim, indent=2))


if __name__ == "__main__":
    main()
