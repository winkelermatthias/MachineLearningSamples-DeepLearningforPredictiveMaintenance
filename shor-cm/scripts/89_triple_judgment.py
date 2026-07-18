#!/usr/bin/env python3
"""Triple judgment: SPEED, PATTERNS, FAULT — separately and together.

Arms on the same 2,000-run corpus (baseline numbers come from the pinned
experiments/massive/o1_results.parquet, regenerated waveforms are
bit-identical):
  peakshor   pure peak-pair CF rationalization + Hz-domain scoring
  union      peak-Shor + baseline generators (comb/cepstrum/envelope +
             REPAIRED 2LF ladder) as extra candidates
  baseline2  repaired-ladder classic estimator, subsample (run_id%5==0),
             isolates the ladder-ordering fix

Judgments:
  SPEED     top1/top3 @1%, octave taxonomy, confidence + isotonic
            calibration (fit on even speed bands, ECE on odd bands)
  PATTERNS  per-family precision/recall of the peak ledger vs SimForge
            composition truth, under TRUE speed (separate) and under
            estimated top-1 (together)
  FAULT     O3 rules + ML(LightGBM, grouped CV) on the evidence ledger
            under TRUE speed (separate) and estimated speed (together)
  JOINT     end-to-end: speed top1 AND fault correct; all-three rate

ENV: N_RUNS (2000), WORKERS (4).
"""
import json, os, sys, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import simforge_lite as SF               # noqa: E402
from shorcm import simforge_corpus as SC             # noqa: E402
from shorcm import blindspeed as BS                  # noqa: E402
from shorcm import peakshor as PS                    # noqa: E402

N_RUNS = int(os.environ.get("N_RUNS", 2000))
WORKERS = int(os.environ.get("WORKERS", 4))
TOL = 0.01
OUT = Path("experiments/peakshor")

FAM_MAP = {"SHAFT": "SHAFT", "VANE": "SHAFT", "HALF": "HALF",
           "BEARING": "NEARRAT", "HUM": "HUM", "NEIGHBOR": "NEIGHBOR"}


def judge_patterns(pf, pa, pc, f0, truth):
    """Recall per truth family (fraction of members found with the right
    label) and precision per claimed family (amp-weighted fraction of
    claimed peaks that match some truth component of the mapped kind)."""
    led = PS.pattern_ledger_peaks(pf, pa, pc, f0)
    claimed = {d["family"]: d for d in led}
    tol_hz = lambda f: max(0.012 * f, 0.6)           # noqa: E731
    rec = {}
    truth_freqs = {}                                  # mapped family -> freqs
    for tfam in truth:
        fam = tfam["family"]
        if fam in ("ELEC_HF",):
            continue
        want = FAM_MAP.get(fam)
        if want is None:                              # ELEC judged separately
            continue
        freqs = [f for f in tfam["freqs_hz"] if 2.0 < f < 1990.0]
        if not freqs:
            continue
        truth_freqs.setdefault(want, []).extend(freqs)
        got = claimed.get(want, {"freqs_hz": []})["freqs_hz"]
        n_hit = sum(any(abs(g - f) < tol_hz(f) for g in got) for f in freqs)
        key = fam.lower()
        rec.setdefault(key, []).append(n_hit / len(freqs))
    # ELEC: not-misattributed metric (VFD 2 f_e must not be SHAFT/HALF)
    for tfam in truth:
        if tfam["family"] != "ELEC":
            continue
        f2e = tfam["freqs_hz"][0]
        if not (2.0 < f2e < 1990.0):
            continue
        bad = any(any(abs(g - f2e) < tol_hz(f2e) for g in
                      claimed.get(k, {"freqs_hz": []})["freqs_hz"])
                  for k in ("SHAFT", "HALF"))
        rec.setdefault("elec_excluded", []).append(0.0 if bad else 1.0)
    prec = {}
    all_truth = [(f, FAM_MAP.get(t["family"]))
                 for t in truth for f in t["freqs_hz"]
                 if FAM_MAP.get(t["family"])]
    for fam in ("SHAFT", "HALF", "NEARRAT", "NEIGHBOR", "HUM"):
        d = claimed.get(fam)
        if not d or not d["freqs_hz"]:
            continue
        wsum = hit = 0.0
        for g, a in zip(d["freqs_hz"], d["amps"]):
            wsum += a
            if any(w == fam and abs(g - f) < tol_hz(f) for f, w in all_truth):
                hit += a
        prec[fam.lower()] = hit / (wsum + 1e-12)
    return {f"rec_{k}": float(np.mean(v)) for k, v in rec.items()} | \
           {f"prec_{k}": float(v) for k, v in prec.items()}


def one_run(i):
    truth = []
    m, x = SC.sample_run(i, truth=truth)
    meta = {"component": m["component"]}
    f_true = m["f_shaft"]
    pf, pa, pc = PS.spectral_peaks(x, SF.FS)
    rows, extra = {}, []

    def spd(cands):
        cands = [c for c in cands if np.isfinite(c["hz"])] or \
            [{"hz": 0.0, "confidence": 0.0}]
        errs = [abs(c["hz"] / f_true - 1) for c in cands]
        top1, top3 = errs[0] <= TOL, min(errs) <= TOL
        octave = (not top1) and any(
            abs(cands[0]["hz"] / (f_true * r) - 1) <= TOL
            for r in (2, 0.5, 1.5, 3, 1 / 3))
        return dict(f_hat=cands[0]["hz"], conf=cands[0]["confidence"],
                    top1=top1, top3=top3, octave=octave)

    est_ps = PS.estimate_speed_shor(x, SF.FS, meta=meta)
    rows["peakshor"] = spd(est_ps)
    f, A = BS._spec(x, SF.FS)
    extra = BS.twolf_ladder(x, SF.FS, meta.get("component") == "motor")
    extra += BS.comb_candidates(f, A)
    extra += BS.cepstrum_candidate(x, SF.FS)
    extra += BS.envelope_candidate(x, SF.FS)
    est_un = PS.estimate_speed_shor(x, SF.FS, meta=meta,
                                    extra_candidates=extra)
    rows["union"] = spd(est_un)
    if i % 5 == 0:
        rows["baseline2"] = spd(BS.estimate_speed(x, SF.FS, meta=meta))

    pat_true = judge_patterns(pf, pa, pc, f_true, truth)
    pat_est = judge_patterns(pf, pa, pc, rows["union"]["f_hat"], truth)

    led_true = SC.ledger(x, SF.FS, f_true)
    led_est = SC.ledger(x, SF.FS, rows["union"]["f_hat"])
    bear_o = next((t["freqs_hz"][0] / f_true for t in truth
                   if t["family"] == "BEARING"), np.nan)
    out = {"run_id": i, "f_true": f_true, "fault": m["fault"],
           "severity": m["severity"], "population": m["population"],
           "component": m["component"], "bear_order_true": bear_o,
           "rules_true": SC.rules_from_ledger(led_true),
           "rules_est": SC.rules_from_ledger(led_est)}
    for arm, r in rows.items():
        for k, v in r.items():
            out[f"{arm}_{k}"] = v
    for k, v in pat_true.items():
        out[f"pt_{k}"] = v
    for k, v in pat_est.items():
        out[f"pe_{k}"] = v
    for k, v in (led_true or {}).items():
        out[f"lt_{k}"] = v
    for k, v in (led_est or {}).items():
        out[f"le_{k}"] = v
    return out


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with Pool(WORKERS, maxtasksperchild=100) as pool:
        for k, r in enumerate(pool.imap_unordered(one_run, range(N_RUNS),
                                                  chunksize=8)):
            rows.append(r)
            if (k + 1) % 100 == 0:
                el = time.time() - t0
                print(f"{k+1}/{N_RUNS} {el:.0f}s "
                      f"(eta {el/(k+1)*(N_RUNS-k-1):.0f}s)", flush=True)
    df = pd.DataFrame(rows).sort_values("run_id")
    df.to_parquet(OUT / "results.parquet", index=False)
    print(f"\nwall {time.time()-t0:.0f}s")

    base = pd.read_parquet("experiments/massive/o1_results.parquet")
    base = base[base.ablation == "full"]
    print("\n== SPEED (tol 1%) ==")
    print(f"baseline(pinned) top1 {base.top1.mean():.3f} "
          f"top3 {base.top3.mean():.3f} octave {base.octave.mean():.3f}")
    for arm in ("peakshor", "union", "baseline2"):
        d = df.dropna(subset=[f"{arm}_top1"])
        print(f"{arm:16s} top1 {d[f'{arm}_top1'].mean():.3f} "
              f"top3 {d[f'{arm}_top3'].mean():.3f} "
              f"octave {d[f'{arm}_octave'].mean():.3f}  (n={len(d)})")
    print("\nunion by fault:")
    print(df.groupby("fault")[["union_top1", "union_top3", "union_octave"]]
          .mean().round(3).to_string())
    print("\n== PATTERNS (recall by family / precision, true vs est speed) ==")
    for pre, lab in (("pt", "true-speed"), ("pe", "est-speed")):
        cols = [c for c in df.columns if c.startswith(pre + "_")]
        mm = df[cols].mean().round(3)
        print(f"[{lab}] " + "  ".join(f"{c[3:]}={v}" for c, v in mm.items()))
    bd = df[df.fault == "bearing"]
    if len(bd):
        hit_t = (np.abs(bd.lt_uns_top_order / bd.bear_order_true - 1) < 0.03) \
            & (bd.lt_uns_abs > 0.1)
        hit_e = (np.abs(bd.le_uns_top_order / bd.bear_order_true - 1) < 0.03) \
            & (bd.le_uns_abs > 0.1)
        print(f"\nBEARING pattern, order-domain drifting-cluster recall: "
              f"true-speed {hit_t.mean():.3f}  est-speed {hit_e.mean():.3f} "
              f"(n={len(bd)})")
    print("\n== FAULT rules arm ==")
    print(f"true speed: {(df.rules_true == df.fault).mean():.3f}   "
          f"est speed: {(df.rules_est == df.fault).mean():.3f}")
    print("\n== JOINT (union arm) ==")
    j1 = (df.union_top1 & (df.rules_est == df.fault)).mean()
    print(f"speed top1 AND rules fault correct: {j1:.3f}")


if __name__ == "__main__":
    main()
