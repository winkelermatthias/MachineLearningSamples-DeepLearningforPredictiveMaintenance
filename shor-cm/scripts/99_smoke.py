#!/usr/bin/env python3
"""End-to-end smoke: build a synthetic mini-MAFAULDA (parquet + catalog),
run feature extraction for all variants and a tiny H1, without network.
Passing this means the plumbing works; it says nothing about real data.
Run: python scripts/99_smoke.py   (writes under smoke/)"""
import shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm.features import run_features, FS      # noqa: E402
from shorcm.tacho import VARIANTS                 # noqa: E402

ROOT = Path("smoke"); RNG = np.random.default_rng(1)


def synth_run(f0, kind, seed):
    rng = np.random.default_rng(seed)
    n = int(5.0 * FS)
    t = np.arange(n) / FS
    f_inst = f0 * (1 + 0.005 * np.sin(2 * np.pi * 0.7 * t))
    phase = 2 * np.pi * np.cumsum(f_inst) / FS
    tach = ((phase % (2 * np.pi)) < 0.12).astype(np.float32)
    arr = 0.2 * rng.standard_normal((n, 8)).astype(np.float32)
    arr[:, 0] = tach
    def locked(o, a): return a * np.cos(o * phase + rng.uniform(0, 6.28))
    def slip(o, a):
        return a * np.cos(2 * np.pi * o * f0 * 1.015 * t + rng.uniform(0, 6.28))
    for c in (2, 3, 5, 6):
        arr[:, c] += locked(1.0, 0.5)
        if kind == "imbalance":
            arr[:, c] += locked(1.0, 1.5)
        elif kind == "horizontal-misalignment":
            arr[:, c] += locked(2.0, 1.0) + locked(3.0, 0.4)
        elif kind == "underhang":
            arr[:, c] += slip(2.998, 0.8) + slip(0.4, 0.3)
    return arr


def main():
    if ROOT.exists():
        shutil.rmtree(ROOT)
    (ROOT / "parquet").mkdir(parents=True)
    rows, feats = [], []
    speeds = [15.0, 25.0, 35.0, 45.0, 55.0]
    for kind in ("normal", "imbalance", "horizontal-misalignment", "underhang"):
        for i, f0 in enumerate(speeds):
            arr = synth_run(f0, kind, hash((kind, i)) % 2**32)
            rid = f"{kind}_{f0}"
            p = ROOT / "parquet" / f"{rid}.parquet"
            pd.DataFrame(arr).to_parquet(p, index=False)
            rows.append(dict(run_id=rid, path=str(p), cls=kind,
                             sub="ball_fault" if kind == "underhang" else "",
                             sev="", f_nom=f0, rate_hz=f0, tacho_quality=0.002))
            for v in VARIANTS:
                f = run_features(arr.astype(float), f0, v)
                f.update(run_id=rid, cls=kind, sev="", f_nom=f0, error="")
                feats.append(f)
    pd.DataFrame(rows).to_parquet(ROOT / "catalog.parquet", index=False)
    df = pd.DataFrame(feats)
    df.to_parquet(ROOT / "features.parquet", index=False)

    # tiny H1: does ratio at bearing bins separate underhang from locked?
    for v in VARIANTS:
        d = df[(df.variant == v) & (df.cls != "normal")]
        bear = d.cls == "underhang"
        r = d["ratio_3.0"].values   # 2.998x lands in the 3.0 bin
        sep = r[bear.values].mean() < r[~bear.values].mean() - 0.2
        print(f"{v:8s} ratio@3.0 bearing={r[bear.values].mean():.2f} "
              f"locked={r[~bear.values].mean():.2f} separated={sep}")
        assert v == "nominal" or sep, f"{v}: coherence ratio failed to separate"
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
