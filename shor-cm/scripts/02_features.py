#!/usr/bin/env python3
"""Extract features for every (run, phase-variant). Parallel over runs.
GATE: refuses to run unless `pytest -q tests/` passes (G2).

Output: features/tabular/features.parquet, long format, one row per
(run_id, variant). Resumable: skips (run, variant) pairs already present.
"""
import subprocess, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm.features import run_features           # noqa: E402
from shorcm.tacho import VARIANTS                   # noqa: E402

CATALOG = Path("data/catalog.parquet")
OUT = Path("features/tabular/features.parquet")


def one(row, variant):
    arr = pd.read_parquet(row["path"]).values.astype(float)
    try:
        f = run_features(arr, row["f_nom"], variant)
    except Exception as e:  # noqa: BLE001  (log, don't kill the pool)
        return dict(run_id=row["run_id"], variant=variant, error=str(e)[:200])
    f.update(run_id=row["run_id"], cls=row["cls"], sub=row["sub"],
             sev=row["sev"], f_nom=row["f_nom"], error="")
    return f


def main():
    if subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/"]).returncode != 0:
        sys.exit("G2 GATE FAILED: synthetic tests must pass before real data")
    cat = pd.read_parquet(CATALOG)
    done = set()
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        done = set(zip(prev.run_id, prev.variant))
    jobs = [(r, v) for _, r in cat.iterrows() for v in VARIANTS
            if (r["run_id"], v) not in done]
    print(f"{len(jobs)} jobs ({len(cat)} runs x {len(VARIANTS)} variants, "
          f"{len(done)} already done)")
    rows, buf = [], 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor() as ex:
        futs = [ex.submit(one, r, v) for r, v in jobs]
        for fut in as_completed(futs):
            rows.append(fut.result())
            buf += 1
            if buf % 200 == 0:
                _flush(rows)
                print(f"{buf}/{len(jobs)}")
    _flush(rows)
    df = pd.read_parquet(OUT)
    print("errors:", (df["error"] != "").sum(), "/", len(df))


def _flush(rows):
    if not rows:
        return
    new = pd.DataFrame(rows)
    if OUT.exists():
        new = pd.concat([pd.read_parquet(OUT), new], ignore_index=True)
        new = new.drop_duplicates(["run_id", "variant"], keep="last")
    tmp = OUT.with_suffix(".tmp.parquet")
    new.to_parquet(tmp, index=False)
    tmp.rename(OUT)  # atomic publish
    rows.clear()


if __name__ == "__main__":
    main()
