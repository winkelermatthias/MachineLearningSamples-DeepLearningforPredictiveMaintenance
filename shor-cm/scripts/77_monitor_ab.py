#!/usr/bin/env python3
"""Held-out A/B: key-based vs general-pattern monitor alarms.

Same fleet protocol and HELD-OUT seed block (99M) as certification:
N_FLEET machines x RECORDS records, VFD speed swings, healthy /
stationary / growing scenarios, consumed ONLY through the deployment
API (Cascade.monitor). Compares, per alarm channel:

  key      rec["alarms"]          (PatternTracker invariant keys)
  pattern  rec["pattern_alarms"]  (GeneralTracker kinematic groups)

false-alarm machines (healthy+stationary), growth recall, first-alarm
delay after onset, and right-type growth attribution.

ENV: N_FLEET (60), RECORDS (16), WORKERS (4).
"""
import json, os, sys, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import simforge_v2 as V2                 # noqa: E402
from shorcm import simforge_mc as MC                 # noqa: E402
from shorcm.cascade import Cascade                   # noqa: E402

N_FLEET = int(os.environ.get("N_FLEET", 60))
RECORDS = int(os.environ.get("RECORDS", 16))
WORKERS = int(os.environ.get("WORKERS", 4))
OUT = Path("experiments/monitor_ab")

WANT_TYPES = {"imbalance": ("HARM",), "misalignment": ("HARM",),
              "looseness": ("HALFHARM",),
              "bearing": ("NEARRAT", "SIDEBAND", "TONE"),
              "gear": ("SIDEBAND", "HARM", "BAND", "TONE")}


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
    mon = Cascade().monitor(V2.kinematic_sheet(m), f_ref=f_base)
    r0 = 3
    first_key, first_pat = None, None
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
        if rec["alarms"] and first_key is None:
            first_key = t
        if rec["pattern_alarms"] and first_pat is None:
            first_pat = t
    want = WANT_TYPES.get(m["fault"], ())
    right_pat = any(set(g["types"]) & set(want)
                    for g in mon.gen.trends_grouped() if g["alarm"])
    return {"machine": i, "scenario": scn, "fault": m["fault"],
            "key_alarm": first_key is not None,
            "pat_alarm": first_pat is not None,
            "key_delay": (first_key - r0) if (first_key is not None
                                              and scn == "growing")
            else np.nan,
            "pat_delay": (first_pat - r0) if (first_pat is not None
                                              and scn == "growing")
            else np.nan,
            "pat_right_type": bool(right_pat)}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(WORKERS, maxtasksperchild=20) as pool:
        rows = list(pool.imap_unordered(one_fleet, range(N_FLEET),
                                        chunksize=1))
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "monitor_ab.parquet", index=False)
    grow = df[df.scenario == "growing"]
    quiet = df[df.scenario != "growing"]
    find = {"n_fleet": len(df), "n_growing": len(grow),
            "records": RECORDS, "seed_block": "99M (held-out)"}
    for ch in ("key", "pat"):
        find[ch] = {
            "false_alarm_machines": round(float(
                quiet[f"{ch}_alarm"].mean()), 3),
            "growth_recall": round(float(grow[f"{ch}_alarm"].mean()), 3),
            "median_delay": (round(float(
                grow[f"{ch}_delay"].median()), 1)
                if grow[f"{ch}_delay"].notna().any() else None)}
    find["pat"]["growth_right_type"] = round(float(
        grow.pat_right_type.mean()), 3)
    find["wall_s"] = round(time.time() - t0, 1)
    (OUT / "findings.json").write_text(json.dumps(find, indent=1))
    print(json.dumps(find, indent=1))


if __name__ == "__main__":
    main()
