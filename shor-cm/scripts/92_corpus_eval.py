#!/usr/bin/env python3
"""Massive SimForge corpus + full O1/O3 execution.

For each corpus run (seed-regeneratable, shorcm.simforge_corpus):
 - ground-truth manifest row (machine, population, fault, severity, ...)
 - O1 blind speed: full system + no_ladder + no_structure ablations,
   top-1/top-3 at 1% tolerance, octave-error taxonomy, confidence
 - O3 evidence ledger on the TOP-1 ESTIMATED speed (cascade-honest:
   a wrong speed poisons the ledger, exactly as in deployment)
 - O3 rules-arm prediction from the ledger

Outputs (experiments/massive/):
  manifest.parquet   ground truth per run
  o1_results.parquet one row per (run, ablation)
  ledger.parquet     one row per run: ledger features + rules prediction

ENV: N_RUNS (default 2000), WORKERS (default 4).
"""
import os, sys, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import simforge_lite as SF               # noqa: E402
from shorcm import simforge_corpus as SC             # noqa: E402
from shorcm import blindspeed as BS                  # noqa: E402

N_RUNS = int(os.environ.get("N_RUNS", 2000))
WORKERS = int(os.environ.get("WORKERS", 4))
TOL = 0.01
OUT = Path("experiments/massive")


def one_run(i):
    m, x = SC.sample_run(i)
    meta = {"component": m["component"]}
    man = {k: m[k] for k in ("run_id", "population", "pole_pairs", "slip",
                             "lf_grid", "f_e", "f_shaft", "component",
                             "fault", "severity", "vanes", "wander",
                             "carrier", "extra_noise", "extra_floor",
                             "extra_neighbor")}
    o1_rows = []
    f_top1 = None
    for tag, kw in (("full", {}),
                    ("no_ladder", {"use_ladder": False}),
                    ("no_structure", {"use_structure": False})):
        cands = BS.estimate_speed(x, SF.FS, meta=meta, **kw)
        errs = [abs(c["hz"] / m["f_shaft"] - 1) for c in cands]
        top1 = errs[0] <= TOL
        top3 = min(errs) <= TOL
        octave = (not top1) and any(
            abs(cands[0]["hz"] / (m["f_shaft"] * r) - 1) <= TOL
            for r in (2, 0.5, 1.5, 3, 1 / 3))
        o1_rows.append(dict(run_id=i, ablation=tag,
                            f_hat=cands[0]["hz"],
                            rel_err=float(errs[0]),
                            conf=cands[0]["confidence"],
                            top1=top1, top3=top3, octave=octave))
        if tag == "full":
            f_top1 = cands[0]["hz"]
    led = SC.ledger(x, SF.FS, f_top1)
    lrow = {"run_id": i, "f_hat": f_top1,
            "rules_pred": SC.rules_from_ledger(led)}
    if led is not None:
        lrow.update(led)
    return man, o1_rows, lrow


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    mans, o1s, leds = [], [], []
    with Pool(WORKERS, maxtasksperchild=200) as pool:
        for k, (man, o1r, lrow) in enumerate(
                pool.imap_unordered(one_run, range(N_RUNS), chunksize=8)):
            mans.append(man); o1s.extend(o1r); leds.append(lrow)
            if (k + 1) % 100 == 0:
                el = time.time() - t0
                print(f"{k+1}/{N_RUNS}  {el:.0f}s "
                      f"(eta {el / (k+1) * (N_RUNS-k-1):.0f}s)", flush=True)
    man = pd.DataFrame(mans).sort_values("run_id")
    o1 = pd.DataFrame(o1s).sort_values(["run_id", "ablation"])
    led = pd.DataFrame(leds).sort_values("run_id")
    man.to_parquet(OUT / "manifest.parquet", index=False)
    o1.to_parquet(OUT / "o1_results.parquet", index=False)
    led.to_parquet(OUT / "ledger.parquet", index=False)

    j = o1.merge(man, on="run_id")
    print(f"\n== corpus: {N_RUNS} runs, {time.time()-t0:.0f}s wall ==")
    print("\nO1 blind speed (tol 1%), by ablation:")
    print(j.groupby("ablation")[["top1", "top3", "octave"]]
          .mean().round(3).to_string())
    fu = j[j.ablation == "full"]
    print("\nfull system by population x component:")
    print(fu.groupby(["population", "component"])[["top1", "top3"]]
          .mean().round(3).to_string())
    print("\nfull system by fault:")
    print(fu.groupby("fault")[["top1", "top3", "octave"]]
          .mean().round(3).to_string())
    print("\nconfidence: mean when top1 correct "
          f"{fu[fu.top1].conf.mean():.2f} vs wrong "
          f"{fu[~fu.top1].conf.mean():.2f}")
    lj = led.merge(man, on="run_id")
    okspd = lj.run_id.isin(fu[fu.top1].run_id)
    acc_all = (lj.rules_pred == lj.fault).mean()
    acc_ok = (lj[okspd].rules_pred == lj[okspd].fault).mean()
    print(f"\nO3 rules arm: acc {acc_all:.3f} overall, "
          f"{acc_ok:.3f} when speed correct (n={int(okspd.sum())})")
    print(pd.crosstab(lj[okspd].fault, lj[okspd].rules_pred).to_string())


if __name__ == "__main__":
    main()
