#!/usr/bin/env python3
"""Evaluate O1 blind speed on SimForge-lite, with ablations, plus a
minimal O3 rules arm running on the ESTIMATED speed (cascade check).
Usage: N_RUNS=150 python scripts/94_eval_blindspeed.py"""
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import simforge_lite as SF               # noqa: E402
from shorcm import blindspeed as BS                  # noqa: E402
from shorcm import tacho as T, spectra as S          # noqa: E402

N = int(os.environ.get("N_RUNS", 150))
rng = np.random.default_rng(20260709)
TOL = 0.01


def rules_fault(x, fs, f_hat):
    """Minimal O3 rules arm on estimated speed, with FIXEDHZ masking:
    hum and neighbor-machine families must not masquerade as bearings."""
    try:
        ph, _ = T.phase_from_comb(x, fs, f_nom=f_hat, prior_rel_sigma=0.008)
        xa = S.angular_resample(x, ph, 128)
        A, o = S.fine_order_spectrum(xa, 128)
    except Exception:
        return "unknown"
    # fixed-Hz bases: grid hum, non-shaft comb families, and any NARROW
    # low-band line (electrical: drive-stable, concentration high; bearing
    # tones are smeared by wander and phase walk, concentration low)
    fz, Az = BS._spec(x, fs)
    bases = [50.0, 60.0, 100.0, 120.0]
    for c in BS.comb_candidates(fz, Az, topn=4):
        if all(abs(c / (f_hat * r) - 1) > 0.03 for r in (0.5, 1, 2, 3)):
            bases.append(c)
    Af = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    ff = np.fft.rfftfreq(len(x), 1 / fs)
    lbm = (ff > 30) & (ff < 380)
    med = np.median(Af[lbm]) + 1e-15
    idx = np.flatnonzero(lbm)
    for i in idx[1:-1]:
        if Af[i] > 8 * med and Af[i] >= Af[i - 1] and Af[i] > Af[i + 1]:
            concentration = Af[i - 1:i + 2].sum() / (Af[i - 12:i + 13].sum() + 1e-12)
            if concentration > 0.6:
                bases += [float(ff[i]), float(ff[i]) / 2.0]
    def masked(order):
        hz = order * f_hat
        return any(abs(hz - k * b) < 1.5
                   for b in bases for k in range(1, 6))
    def aabs(oo):
        bi = int(np.argmin(np.abs(o - oo)))
        return float(A[max(bi - 1, 0):bi + 2].max())
    a1, a2 = aabs(1), aabs(2)
    half = sum(aabs(k) > 0.10 for k in (0.5, 1.5, 2.5, 3.5))
    pk = S.peak_orders(A, o, 9.0, 25, guard=5.0)
    tol = max(S.snap_tol(max(int(len(xa) / 128), 10)), 0.004)
    do = o[1] - o[0]
    def narrow(oo, a):
        i = int(round(oo / do))
        loc = np.median(A[max(i - 40, 0):i + 40]) + 1e-15
        return a > 4.0 * loc          # tone above LOCAL floor, not hump texture
    uns = sum(a for oo, a, _ in pk
              if S.cf_snap(oo, tol=tol, qmax=8) is None
              and 1.8 < oo < 9.0 and not masked(oo) and narrow(oo, a))
    tot = sum(a for _, a, _ in pk) + 1e-12
    if half >= 2:
        return "looseness"
    if uns / tot > 0.15:
        return "bearing"
    if a2 > 0.35 and a2 > 0.6 * a1:
        return "misalignment"
    if a1 > 0.5:
        return "imbalance"
    return "healthy"


def main():
    t0 = time.time()
    rows = []
    for i in range(N):
        m = SF.sample_machine(rng)
        x = SF.synth_run(m, rng)
        meta = {"component": m["component"]}
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
            rows.append(dict(i=i, ablation=tag, pop=m["population"],
                             comp=m["component"], fault=m["fault"],
                             f_true=m["f_shaft"], f_hat=cands[0]["hz"],
                             conf=cands[0]["confidence"],
                             top1=top1, top3=top3, octave=octave))
            if tag == "full":
                fhat = cands[0]["hz"] if top1 else \
                    (cands[int(np.argmin(errs))]["hz"] if top3 else cands[0]["hz"])
                rows[-1]["fault_pred"] = rules_fault(x, SF.FS, cands[0]["hz"])
        if (i + 1) % 25 == 0:
            print(f"{i+1}/{N}  {time.time()-t0:.0f}s")
    df = pd.DataFrame(rows)
    out = Path("experiments/o1"); out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "results.parquet", index=False)

    print("\n== O1 blind speed, tolerance 1% ==")
    piv = df.groupby("ablation")[["top1", "top3", "octave"]].mean()
    print(piv.round(3).to_string())
    print("\nfull system by population / component:")
    print(df[df.ablation == "full"].groupby(["pop", "comp"])
          [["top1", "top3"]].mean().round(3).to_string())
    fu = df[df.ablation == "full"]
    print("\nconfidence sanity: mean conf when top1 correct "
          f"{fu[fu.top1].conf.mean():.2f} vs wrong {fu[~fu.top1].conf.mean():.2f}")
    print("\n== O3 rules arm on estimated speed (top-1 only) ==")
    ok = fu[fu.top1]
    acc = (ok.fault_pred == ok.fault).mean()
    print(f"fault accuracy when speed correct: {acc:.3f}  (n={len(ok)})")
    print(pd.crosstab(ok.fault, ok.fault_pred).to_string())
    print(f"\nwall {time.time()-t0:.0f}s for {N} runs x 3 ablations")


if __name__ == "__main__":
    main()
