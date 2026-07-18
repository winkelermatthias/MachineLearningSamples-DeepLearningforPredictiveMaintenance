#!/usr/bin/env python3
"""Corpus feature regeneration v3 — the renderer changed (impulsive
bearings, iteration 24), so EVERY training feature is recomputed
consistently: blind est-speed at deployment condition, raw ledger,
raw pattern features, envelope pattern features.

Resumable: skips run_ids already in the output parquet.
ENV: WORKERS (4), N (2000).
"""
import json, os, sys, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import simforge_v2 as V2                 # noqa: E402
from shorcm import simforge_corpus as SC             # noqa: E402
from shorcm import peakshor as PS                    # noqa: E402
from shorcm import patterns as PT                    # noqa: E402

WORKERS = int(os.environ.get("WORKERS", 4))
N = int(os.environ.get("N", 2000))
OUT = Path("experiments/corpus_v3")


def one(i):
    truth = []
    m, x = V2.sample_run(i, truth=truth)
    row = {"run_id": i, "fault": m["fault"], "subtype": m.get("subtype"),
           "severity": m["severity"], "archetype": m["archetype"],
           "f_true": m["f_shaft"], "population": m["population"]}
    # unit-RMS normalization: identical to the analyze_record entry
    # treatment (audit F1) so train and serve see the same scale
    x = x - x.mean()
    x = x / max(float(np.sqrt(np.mean(x ** 2))), 1e-12)
    sheet = V2.kinematic_sheet(m)
    est = PS.estimate_speed_sheet(x, V2.FS, sheet)
    f_hat = est[0]["hz"] if est and np.isfinite(est[0]["hz"]) else None
    if f_hat is None or f_hat <= 0:
        return row                       # unusable record: label only
    row["f_hat"] = f_hat
    row["speed_ok"] = bool(abs(f_hat / m["f_shaft"] - 1) < 0.05)
    led = SC.ledger(x, V2.FS, f_hat, spr=256, uns_hi=16.0)
    if led:
        for k, v in led.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                row[f"le_{k}"] = float(v)
    try:
        pats, _ = PT.decompose(x, V2.FS, f_hat, sheet=sheet)
        row.update(PT.pattern_features(pats))
    except Exception:
        pass
    try:
        pats_e, _, _ = PT.decompose_envelope(x, V2.FS, f_hat)
        row.update(PT.pattern_features_env(pats_e))
    except Exception:
        pass
    return row


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    fp = OUT / "features.parquet"
    old = None
    if fp.exists():
        old = pd.read_parquet(fp)
        done = set(old.run_id)
    todo = [i for i in range(N) if i not in done]
    rows = []
    with Pool(WORKERS, maxtasksperchild=25) as pool:
        for k, r in enumerate(pool.imap_unordered(one, todo,
                                                  chunksize=2)):
            rows.append(r)
            if (k + 1) % 200 == 0:
                df = pd.DataFrame(rows)
                if old is not None:
                    df = pd.concat([old, df], ignore_index=True)
                df.to_parquet(fp, index=False)
                print(f"{k+1}/{len(todo)} wall {time.time()-t0:.0f}s",
                      flush=True)
    df = pd.DataFrame(rows)
    if old is not None:
        df = pd.concat([old, df], ignore_index=True)
    df.to_parquet(fp, index=False)
    print(json.dumps({"n": len(df),
                      "speed_ok": round(float(
                          df.get("speed_ok", pd.Series()).mean()), 3),
                      "wall_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
