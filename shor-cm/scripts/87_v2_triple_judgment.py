#!/usr/bin/env python3
"""V2 triple judgment: the expanded machine population, judged on
PATTERNS, SPEED, and FAULT — separately and together, per archetype.

Corpus: SimForge v2 (gearboxes, planetary, belt drives, roots blowers,
typed pumps/fans, size-correct bearings, 6-way faults with subtypes).
Speed truth = INPUT (motor) shaft; hits on the driven shaft are counted
separately (they are the honest gearbox ambiguity, not a win).

Ledger: spr=256 (orders to 128 for mesh), uns_hi=16 (big bearings).
ML: LightGBM 6-class on LEDGER_FEATURES_V2, GroupKFold by speed band.

ENV: N_RUNS (2000), WORKERS (4).
"""
import json, os, sys, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import simforge_v2 as V2                 # noqa: E402
from shorcm import simforge_corpus as SC             # noqa: E402
from shorcm import blindspeed as BS                  # noqa: E402
from shorcm import peakshor as PS                    # noqa: E402

N_RUNS = int(os.environ.get("N_RUNS", 2000))
WORKERS = int(os.environ.get("WORKERS", 4))
TOL = 0.01
OUT = Path("experiments/v2")

# truth family -> acceptable claim labels (Hz-domain pattern ledger)
ACCEPT = {"SHAFT": {"SHAFT"}, "HALF": {"HALF"},
          "SHAFT2": {"NEIGHBOR", "NEARRAT", "RATIONAL"},
          "PASSAGE": {"SHAFT", "RATIONAL", "GEAR", "NEIGHBOR"},
          "GMF": {"GEAR", "SHAFT"}, "GMF2": {"GEAR", "SHAFT"},
          "BEARING": {"NEARRAT"}, "HUM": {"HUM"},
          "NEIGHBOR": {"NEIGHBOR"}}


def judge_patterns(pf, pa, pc, f0, truth):
    led = PS.pattern_ledger_peaks(pf, pa, pc, f0)
    lab_of = {}
    for d in led:
        for g in d["freqs_hz"]:
            lab_of[g] = d["family"]
    gs = np.array(sorted(lab_of))
    tol_hz = lambda f: max(0.012 * f, 0.6)           # noqa: E731

    def labels_near(f):
        if not len(gs):
            return set()
        i = np.searchsorted(gs, f)
        out = set()
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(gs) and abs(gs[j] - f) < tol_hz(f):
                out.add(lab_of[gs[j]])
        return out

    rec = {}
    for tfam in truth:
        fam = tfam["family"]
        if fam == "ELEC_HF":
            continue
        if fam == "ELEC":
            f2e = tfam["freqs_hz"][0]
            if 2.0 < f2e < 1990.0:
                bad = labels_near(f2e) & {"SHAFT", "HALF"}
                rec.setdefault("elec_excluded", []).append(
                    0.0 if bad else 1.0)
            continue
        acc = ACCEPT.get(fam)
        if acc is None:
            continue
        freqs = [f for f in tfam["freqs_hz"] if 2.0 < f < 1990.0]
        if not freqs:
            continue
        hit = sum(bool(labels_near(f) & acc) for f in freqs)
        rec.setdefault(fam.lower(), []).append(hit / len(freqs))
        if fam in ("GMF", "GMF2"):        # mesh line itself, ex-sidebands
            rec.setdefault("gmf_center", []).append(
                float(bool(labels_near(freqs[0]) & acc)))
    return {f"rec_{k}": float(np.mean(v)) for k, v in rec.items()}


def one_run(i):
    truth = []
    m, x = V2.sample_run(i, truth=truth)
    f0 = m["f_shaft"]
    meta = {"component": "motor" if m["component"] == "motor" else "pump"}
    pf, pa, pc = PS.spectral_peaks(x, V2.FS)

    f, A = BS._spec(x, V2.FS)
    extra = BS.twolf_ladder(x, V2.FS, m["component"] == "motor")
    extra += BS.comb_candidates(f, A)
    extra += BS.cepstrum_candidate(x, V2.FS)
    extra += BS.envelope_candidate(x, V2.FS)
    arms = {"pure": PS.estimate_speed_shor(x, V2.FS, meta=meta),
            "union": PS.estimate_speed_shor(x, V2.FS, meta=meta,
                                            extra_candidates=extra)}
    others = [v for v in (m.get("f2"), m.get("f3"), m.get("fc"))
              if v and abs(v / f0 - 1) > TOL]
    out = {"run_id": i, "archetype": m["archetype"], "f_true": f0,
           "fault": m["fault"], "subtype": m["subtype"],
           "severity": m["severity"], "population": m["population"],
           "component": m["component"], "low_speed": m["low_speed"],
           "resid_1x": m["resid_1x"]}
    for arm, cands in arms.items():
        cands = [c for c in cands if np.isfinite(c["hz"])] or \
            [{"hz": 0.0, "confidence": 0.0}]
        errs = [abs(c["hz"] / f0 - 1) for c in cands]
        top1 = errs[0] <= TOL
        out[f"{arm}_top1"] = top1
        out[f"{arm}_top3"] = min(errs) <= TOL
        out[f"{arm}_octave"] = (not top1) and any(
            abs(cands[0]["hz"] / (f0 * r) - 1) <= TOL
            for r in (2, 0.5, 1.5, 3, 1 / 3))
        out[f"{arm}_shaft2"] = (not top1) and any(
            abs(cands[0]["hz"] / v - 1) <= TOL for v in others)
        out[f"{arm}_f_hat"] = cands[0]["hz"]
        out[f"{arm}_conf"] = cands[0]["confidence"]

    for k, v in judge_patterns(pf, pa, pc, f0, truth).items():
        out[f"pt_{k}"] = v
    for k, v in judge_patterns(pf, pa, pc, out["union_f_hat"],
                               truth).items():
        out[f"pe_{k}"] = v

    led_t = SC.ledger(x, V2.FS, f0, spr=256, uns_hi=16.0)
    led_e = SC.ledger(x, V2.FS, out["union_f_hat"], spr=256, uns_hi=16.0)
    out["rules_true"] = SC.rules_from_ledger(led_t)
    out["rules_est"] = SC.rules_from_ledger(led_e)
    for k, v in (led_t or {}).items():
        out[f"lt_{k}"] = v
    for k, v in (led_e or {}).items():
        out[f"le_{k}"] = v
    # bearing order-domain check
    bt = next((t for t in truth if t["family"] == "BEARING"), None)
    out["bear_hz_true"] = bt["freqs_hz"][0] if bt else np.nan
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
    print(f"wall {time.time()-t0:.0f}s")

    print("\n== SPEED vs input shaft (tol 1%) ==")
    for arm in ("pure", "union"):
        print(f"{arm:6s} top1 {df[f'{arm}_top1'].mean():.3f} "
              f"top3 {df[f'{arm}_top3'].mean():.3f} "
              f"octave {df[f'{arm}_octave'].mean():.3f} "
              f"driven-shaft-capture {df[f'{arm}_shaft2'].mean():.3f}")
    print("\nunion by archetype:")
    print(df.groupby("archetype")[["union_top1", "union_top3",
                                   "union_shaft2"]].mean().round(3)
          .to_string())
    print("\nunion by speed class:")
    print(df.groupby("low_speed")[["union_top1", "union_top3"]]
          .mean().round(3).to_string())
    print("\n== PATTERNS ==")
    for pre, lab in (("pt", "true-speed"), ("pe", "est-speed")):
        cols = [c for c in df.columns if c.startswith(pre + "_")]
        mm = df[cols].mean().round(3)
        print(f"[{lab}] " + "  ".join(f"{c[3:]}={v}"
                                      for c, v in mm.items()))
    print("\n== FAULT rules (6-way) ==")
    print(f"true speed {(df.rules_true == df.fault).mean():.3f}   "
          f"est speed {(df.rules_est == df.fault).mean():.3f}")
    print(pd.crosstab(df.fault, df.rules_true).to_string())
    print("\n== JOINT ==")
    print(f"union top1 AND rules_est correct: "
          f"{(df.union_top1 & (df.rules_est == df.fault)).mean():.3f}")


if __name__ == "__main__":
    main()
