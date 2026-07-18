#!/usr/bin/env python3
"""End-to-end certification of the frozen cascade on HELD-OUT synthetic
data (fresh seed blocks no model or threshold ever saw), consumed only
through the deployment API — exactly as MAFAULDA / Relos data will be.

 A. Single records: N_SINGLE multi-channel machines via
    Cascade.analyze_record. Judged: speed top-1/top-3, calibrated
    confidence ECE, abstain contract (coverage/accuracy of emitted),
    fault (frozen ML + rules) with abstention, misalignment subtype.
 B. Fleets: N_FLEET machines x RECORDS via MachineMonitor with growth /
    stationary / healthy scenarios. Judged: growth recall, false-alarm
    rate, detection delay.

ENV: N_SINGLE (1500), N_FLEET (60), RECORDS (16), WORKERS (4).
"""
import json, os, sys, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import simforge_v2 as V2                 # noqa: E402
from shorcm import simforge_mc as MC                 # noqa: E402
from shorcm import tracker as TK                     # noqa: E402
from shorcm.cascade import Cascade                   # noqa: E402

N_SINGLE = int(os.environ.get("N_SINGLE", 1500))
N_FLEET = int(os.environ.get("N_FLEET", 60))
RECORDS = int(os.environ.get("RECORDS", 16))
WORKERS = int(os.environ.get("WORKERS", 4))
TOL = 0.01
OUT = Path("experiments/certification")
CASC = None


def _casc():
    global CASC
    if CASC is None:
        CASC = Cascade.load("models")
    return CASC


def one_single(i):
    rng = V2.rng_for_run((88_000_000, i))            # held-out seeds
    m = V2.sample_machine(rng)
    X = MC.synth_run_mc(m, rng)
    X = X + rng.uniform(0.0, 0.2) * rng.standard_normal(X.shape)
    out = _casc().analyze_record(X, MC.FS, V2.kinematic_sheet(m),
                                 meta={"component": "motor"})
    errs = [abs(c["hz"] / m["f_shaft"] - 1)
            for c in out["speed_candidates"]
            if np.isfinite(c["hz"])] or [9.9]
    r = {"run_id": i, "archetype": m["archetype"], "fault": m["fault"],
         "subtype": m["subtype"], "severity": m["severity"],
         "top1": errs[0] <= TOL, "top3": min(errs) <= TOL,
         "p_speed": out["speed_confidence"],
         "abstain_speed": out["abstain_speed"],
         "rules": out.get("fault_rules", "unknown"),
         "ml": out.get("fault_ml", "unknown"),
         "abstain_fault": out.get("abstain_fault", True)}
    if "misalignment_subtype" in out:
        r["subtype_pred"] = out["misalignment_subtype"]
    return r


def one_fleet(i):
    rng0 = V2.rng_for_run((99_000_000, i))
    m = V2.sample_machine(rng0)
    u = rng0.random()
    if u < 0.4 or m["fault"] == "healthy":
        m["fault"], m["severity"], scn = "healthy", 0.0, "healthy"
    elif u < 0.65:
        scn = "stationary"
    else:
        scn = "growing"
    f_base = m["f_shaft"]
    mon = _casc().monitor(V2.kinematic_sheet(m), f_ref=f_base)
    fkey = TK.FAULT_KEY.get(m["fault"])
    r0 = 3
    alarmed_t = None
    for t in range(RECORDS):
        rng = V2.rng_for_run((99_500_000 + i, t))
        sc = float(np.clip(1 + 0.18 * np.sin(1.7 * t + i)
                           + rng.normal(0, 0.03), 0.75, 1.25)) \
            if m["population"] == "vfd" else float(1 + rng.normal(0, .004))
        mm = dict(m)
        if scn == "growing":
            uu = max(t - r0, 0) / max(RECORDS - 1 - r0, 1)
            mm["severity"] = float(np.clip(0.05 + 0.85 * uu, 0.03, 1.0))
        elif scn == "stationary":
            mm["severity"] = float(np.clip(
                m["severity"] * (1 + 0.1 * rng.standard_normal()),
                0.05, 1.0))
        mm["f_shaft"] = f_base * sc
        if m["population"] == "vfd":
            mm["f_e"] = mm["f_shaft"] * mm["pole_pairs"] / (1 - mm["slip"])
        for k in ("f2", "f3", "fc", "gmf", "gmf2"):
            if k in mm:
                mm[k] = mm[k] * sc
        X = MC.synth_run_mc(mm, rng, omega1x=sc ** 2)
        rec = mon.feed(t, X, MC.FS)
        if rec["alarms"] and alarmed_t is None:
            alarmed_t = t
    key_alarm = False
    if fkey:
        tds = mon.tracker.trends(adaptive=True)
        if fkey == "NEARRAT":
            key_alarm = any(d["alarm"] for k, d in tds.items()
                            if k.startswith("NEARRAT#"))
        else:
            key_alarm = tds.get(fkey, {}).get("alarm", False)
    return {"machine": i, "scenario": scn, "fault": m["fault"],
            "any_alarm": alarmed_t is not None,
            "first_alarm_t": alarmed_t if alarmed_t is not None else np.nan,
            "onset": r0 if scn == "growing" else np.nan,
            "fault_key_alarm": bool(key_alarm)}


def ece(p, y, bins=10):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    p, y = np.asarray(p), np.asarray(y, float)
    for k in range(bins):
        s = (p >= edges[k]) & (p < edges[k + 1] + (k == bins - 1))
        if s.sum():
            e += s.sum() / len(p) * abs(y[s].mean() - p[s].mean())
    return float(e)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(WORKERS, maxtasksperchild=50) as pool:
        singles = list(pool.imap_unordered(one_single, range(N_SINGLE),
                                           chunksize=4))
        fleets = list(pool.imap_unordered(one_fleet, range(N_FLEET),
                                          chunksize=1))
    ds = pd.DataFrame(singles)
    dfl = pd.DataFrame(fleets)
    ds.to_parquet(OUT / "singles.parquet", index=False)
    dfl.to_parquet(OUT / "fleet.parquet", index=False)

    emit = ~ds.abstain_speed
    cert = {"n_single": len(ds), "n_fleet": len(dfl),
            "speed": {
                "top1": round(float(ds.top1.mean()), 4),
                "top3": round(float(ds.top3.mean()), 4),
                "ece": round(ece(ds.p_speed, ds.top1), 4),
                "emit_coverage": round(float(emit.mean()), 3),
                "emit_top1": round(float(ds.top1[emit].mean()), 4)},
            "fault": {
                "ml_acc": round(float((ds.ml == ds.fault).mean()), 4),
                "rules_acc": round(float((ds.rules == ds.fault).mean()),
                                   4),
                "ml_acc_emitted": round(float(
                    (ds.ml == ds.fault)[~ds.abstain_fault].mean()), 4),
                "fault_coverage": round(float(
                    (~ds.abstain_fault).mean()), 3)}}
    mis = ds[(ds.fault == "misalignment")].dropna(subset=["subtype_pred"])
    if len(mis) > 20:
        y = mis.subtype == "angular"
        yh = mis.subtype_pred == "angular"
        cert["misalignment_subtype"] = {
            "n": len(mis), "accuracy": round(float((y == yh).mean()), 3)}
    grow = dfl[dfl.scenario == "growing"]
    other = dfl[dfl.scenario != "growing"]
    delays = (grow.first_alarm_t - grow.onset).dropna()
    cert["tracking"] = {
        "growth_recall_fault_key": round(float(
            grow.fault_key_alarm.mean()), 3),
        "growth_any_alarm": round(float(grow.any_alarm.mean()), 3),
        "false_alarm_machines": round(float(other.any_alarm.mean()), 3),
        "median_delay": (round(float(delays.median()), 1)
                         if len(delays) else None),
        "n_growing": len(grow)}
    cert["wall_s"] = round(time.time() - t0, 1)
    (OUT / "certification.json").write_text(json.dumps(cert, indent=1))
    print(json.dumps(cert, indent=1))


if __name__ == "__main__":
    main()
