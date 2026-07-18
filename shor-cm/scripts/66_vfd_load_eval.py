#!/usr/bin/env python3
"""F5 blind-spot judgment: load-varying VFD (f_e constant, shaft
moving with slip). PRE-REGISTERED:
 V1: on HEALTHY machines with strong electrical lines, the fraction
     of tracked NEARRAT/TONE instances (>= 6 pts) flagged
     fixed_source=True must be >= 0.6 (the flag catches the disguised
     electrical lines), and
 V2: on GROWING-BEARING machines under the same load variation, NO
     bearing instance within 6% of the true defect order is flagged
     fixed_source=True (the flag never suppresses a real bearing).
ENV: N (30), WORKERS (4).
"""
import json, os, sys, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import simforge_v2 as V2                 # noqa: E402
from shorcm import patterns as PT                    # noqa: E402

N = int(os.environ.get("N", 30))
WORKERS = int(os.environ.get("WORKERS", 4))
OUT = Path("experiments/vfd_load")


def run_fleet(i, bearing):
    rng0 = V2.rng_for_run((57_000_000 + (500 if bearing else 0), i))
    m = V2.sample_machine(rng0)
    m["population"] = "vfd"
    m["component"] = "motor"
    if bearing:
        m["fault"], m["subtype"], m["fault_shaft"] = \
            "bearing", "BPFO", "in"
    else:
        m["fault"], m["severity"] = "healthy", 0.0
    f_e = m["f_shaft"] * m["pole_pairs"] / (1 - m["slip"])
    tr = PT.GeneralTracker()
    bo_true = m["brg_in"]["BPFO"] if bearing else None
    for t in range(14):
        rng = V2.rng_for_run((57_500_000 + i + (900 if bearing else 0),
                              t))
        slip_t = 0.005 + 0.030 * (0.5 + 0.5 * np.sin(1.9 * t + i))
        mm = dict(m)
        mm["slip"] = slip_t
        mm["f_shaft"] = f_e * (1 - slip_t) / m["pole_pairs"]
        mm["f_e"] = f_e                       # drive setpoint CONSTANT
        if bearing:
            mm["severity"] = float(np.clip(0.2 + 0.7 * t / 13,
                                           0.05, 1.0))
        x = V2.synth_run(mm, rng)
        try:
            pats, _ = PT.decompose(x, V2.FS, mm["f_shaft"])
        except Exception:
            continue
        tr.update(t, pats, f_speed=mm["f_shaft"])
    out = []
    for r in tr.reg:
        if r["type"] not in ("NEARRAT", "TONE") or len(r["pts"]) < 6:
            continue
        o_med = float(np.median([o for o, _ in r.get("obs", [])])) \
            if r.get("obs") else np.nan
        is_true_bearing = bool(
            bearing and np.isfinite(o_med)
            and any(abs(o_med / (k * bo_true) - 1) < 0.06
                    for k in (1.0, 2.0)))
        out.append({"machine": i, "bearing_fleet": bearing,
                    "o_med": o_med,
                    "fixed_source": tr.fixed_source(r),
                    "is_true_bearing": is_true_bearing})
    return out


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    args = [(i, False) for i in range(N)] + \
           [(i, True) for i in range(N // 2)]
    with Pool(WORKERS, maxtasksperchild=6) as pool:
        rows = [r for rr in pool.starmap(run_fleet, args, chunksize=1)
                for r in rr]
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "instances.parquet", index=False)
    h = df[~df.bearing_fleet]
    flagged = h.fixed_source.apply(lambda v: v is True)
    b = df[df.bearing_fleet & df.is_true_bearing]
    bad_flags = int(b.fixed_source.apply(lambda v: v is True).sum())
    find = {"healthy_instances": int(len(h)),
            "V1_flagged_frac": round(float(flagged.mean()), 3)
            if len(h) else None,
            "bearing_instances": int(len(b)),
            "V2_true_bearings_wrongly_flagged": bad_flags,
            "verdicts": {"V1": bool(len(h) and flagged.mean() >= 0.6),
                         "V2": bool(bad_flags == 0 and len(b) > 0)},
            "wall_s": round(time.time() - t0, 1)}
    (OUT / "findings.json").write_text(json.dumps(find, indent=1))
    print(json.dumps(find, indent=1))


if __name__ == "__main__":
    main()
