#!/usr/bin/env python3
"""Convert extracted CSVs to float32 parquet, build catalog.parquet.

Labels come from directory structure, e.g.
  extracted/imbalance/6g/13.1072.csv
  extracted/underhang/outer_race/6g/32.768.csv
Filename stem = nominal rotation speed in Hz (this is the "estimated machine
speed reading" used by the no-tacho V-NOMINAL variant).

Also measures true speed + tacho quality from ch0 for the audit.
CSVs are deleted after successful conversion (disk budget).
"""
import hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm.tacho import phase_from_tacho          # noqa: E402
from shorcm.features import FS                     # noqa: E402

EXTRACT = Path("data/extracted")
PQ = Path("data/parquet")
CATALOG = Path("data/catalog.parquet")

TOP_CLASSES = {"normal", "horizontal-misalignment", "vertical-misalignment",
               "imbalance", "underhang", "overhang"}


def parse_labels(p: Path):
    rel = p.relative_to(EXTRACT).parts
    cls = rel[0]
    sub = rel[1] if len(rel) > 2 else ""
    sev = rel[2] if len(rel) > 3 else (rel[1] if len(rel) == 3 and cls in
                                       ("imbalance", "horizontal-misalignment",
                                        "vertical-misalignment") else "")
    # underhang/overhang: rel = (class, fault_type, severity, file)
    if cls in ("underhang", "overhang") and len(rel) >= 3:
        sub, sev = rel[1], (rel[2] if len(rel) > 3 else "")
    return cls, sub, sev


def main():
    PQ.mkdir(parents=True, exist_ok=True)
    rows = []
    files = sorted(EXTRACT.rglob("*.csv"))
    if not files:
        sys.exit("no CSVs found under data/extracted")
    for i, p in enumerate(files):
        cls, sub, sev = parse_labels(p)
        if cls not in TOP_CLASSES:
            continue
        f_nom = float(p.stem)
        arr = pd.read_csv(p, header=None, dtype=np.float32).values
        assert arr.shape[1] == 8, f"{p}: {arr.shape[1]} channels, want 8"
        run_id = f"{cls}_{sub}_{sev}_{p.stem}".replace("/", "-").replace(" ", "")
        out = PQ / cls / f"{run_id}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(arr).to_parquet(out, index=False)
        sha1 = hashlib.sha1(arr.tobytes()).hexdigest()[:16]
        try:
            _, m = phase_from_tacho(arr[:, 0].astype(float), FS)
            rate, quality, npulse = m["rate_hz"], m["quality"], m["n_pulses"]
        except ValueError:
            rate, quality, npulse = np.nan, np.nan, 0
        rows.append(dict(run_id=run_id, path=str(out), cls=cls, sub=sub,
                         sev=sev, f_nom=f_nom, rate_hz=rate,
                         tacho_quality=quality, n_pulses=npulse,
                         n_samples=arr.shape[0], sha1=sha1))
        p.unlink()  # free disk
        if i % 100 == 0:
            print(f"{i}/{len(files)}")
    cat = pd.DataFrame(rows)
    cat.to_parquet(CATALOG, index=False)
    # G1 evidence
    print(cat.groupby("cls").size())
    bad = cat[np.abs(cat.rate_hz - cat.f_nom) / cat.f_nom > 0.15]
    print(f"runs where tacho speed deviates >15% from filename: {len(bad)}")
    print(f"median tacho_quality: {cat.tacho_quality.median():.4f}")


if __name__ == "__main__":
    main()
