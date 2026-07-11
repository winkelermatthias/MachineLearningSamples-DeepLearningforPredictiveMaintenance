#!/usr/bin/env python3
"""Iteration 0, run in-sandbox: the wave-6 loop executed once on synthetic
physics (mirror down, no real data, no substitutes).

What this proves mechanically:
 - the 2x2 detection matrix (coherent vs PPA x locked vs drifting tone)
 - slip_scan recovering per-run slip on bearing-like tones
 - M0 (ratio features) vs M1 (+H4) through paired bootstrap, the
   promotion_gate, the ledger, and the vault filter, end to end

Synthetic set: 60 runs, 4 classes, speeds 12-58 Hz, 1.5% speed wander,
bearing tones with BOTH mean slip (0.8-1.8%) and phase random walk.
1 CPU friendly: spr=256, 4 s runs, tacho variant only.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import tacho as T, spectra as S, amplify as AM, metrics as M

FS, SPR, REVS = 25_000, 256, 5
RNG = np.random.default_rng(20260709)
CLASSES = ["normal", "imbalance", "misalignment", "bearing"]


def synth_run(cls, f0, seed):
    rng = np.random.default_rng(seed)
    n = int(4.0 * FS)
    t = np.arange(n) / FS
    f_inst = f0 * (1 + 0.015 * np.sin(2 * np.pi * 0.6 * t + rng.uniform(0, 6)))
    phase = 2 * np.pi * np.cumsum(f_inst) / FS
    tach = ((phase % (2 * np.pi)) < 0.12).astype(float)
    x = 0.35 * rng.standard_normal(n) + 0.4 * np.cos(phase + rng.uniform(0, 6))
    slip = np.nan
    if cls == "imbalance":
        x += rng.uniform(0.8, 1.5) * np.cos(phase + rng.uniform(0, 6))
    elif cls == "misalignment":
        x += rng.uniform(0.5, 1.0) * np.cos(2 * phase + rng.uniform(0, 6)) \
           + 0.3 * np.cos(3 * phase + rng.uniform(0, 6))
    elif cls == "bearing":
        slip = rng.uniform(0.008, 0.018)
        # mean slip + phase random walk (the part coherence can't see)
        rw = np.cumsum(rng.normal(0, 0.9 / np.sqrt(FS / f0), n))
        x += rng.uniform(0.4, 0.8) * np.cos(2.998 * (1 + slip) * phase + rw)
    return tach, x, slip


def feats(tach, x, want_h4):
    ph, meta = T.phase_from_tacho(tach, FS)
    xa = S.angular_resample(x, ph, SPR)
    Z, orders = S.block_spectra(xa, SPR, REVS)
    coh, inc, ratio = S.coherent_split(Z)
    k = np.arange(1, 26)                       # orders 0.2 .. 5.0
    f = {f"ratio_{(i)/REVS:.1f}": float(ratio[i]) for i in k}
    f.update({f"inc_{(i)/REVS:.1f}": float(inc[i]) for i in k})
    if want_h4:
        sc = AM.slip_scan(xa, SPR, 2.998, span=0.03, n=201)
        b = int(round(2.998 * REVS))
        zs, *_ = AM.ppa_zscore(Z[:, b], n_perm=80)
        f.update(h4_slip=sc["slip"], h4_sharp=sc["sharpness"],
                 h4_amp=sc["amp"], h4_ppa=zs)
    return f, ratio, Z, orders


def main():
    t0 = time.time()
    rows = []
    for i in range(int(__import__("os").environ.get("N_RUNS", 60))):
        cls = CLASSES[i % 4]
        f0 = float(RNG.uniform(12, 58))
        tach, x, slip = synth_run(cls, f0, 1000 + i)
        f, *_ = feats(tach, x, want_h4=True)
        f.update(run_id=f"synth_{i:03d}", cls=cls, f0=f0, true_slip=slip)
        rows.append(f)
    df = pd.DataFrame(rows)

    # vault filter live (even in rehearsal)
    dev = ~M.vault_mask(df.run_id.values)
    d = df[dev].reset_index(drop=True)
    print(f"runs: {len(df)} total, {dev.sum()} dev, {len(df)-dev.sum()} vault")

    y = pd.factorize(d.cls)[0]
    groups = (d.f0 // 8).astype(int)
    m0_cols = [c for c in d if c.startswith("ratio_")]
    m1_cols = m0_cols + ["h4_slip", "h4_sharp", "h4_amp", "h4_ppa"]
    preds = {}
    for tag, cols in (("M0", m0_cols), ("M1", m1_cols)):
        X = d[cols].replace([np.inf, -np.inf], 0).fillna(0).values
        p = np.full(len(d), -1)
        for tr, te in GroupKFold(4).split(X, y, groups):
            p[te] = LogisticRegression(max_iter=3000).fit(
                (X[tr] - X[tr].mean(0)) / (X[tr].std(0) + 1e-9), y[tr]
            ).predict((X[te] - X[tr].mean(0)) / (X[tr].std(0) + 1e-9))
        preds[tag] = p
    f1m = lambda yy, pp: f1_score(yy, pp, average="macro")  # noqa: E731
    deltas = M.paired_bootstrap_delta(y, preds["M0"], preds["M1"], f1m, n=800)
    gate = M.promotion_gate(deltas, guardrails_ok=True, attempts=1)
    rec = M.log_attempt("iter0_H4_synthetic",
                        {"n_runs": len(d), "note": "sandbox rehearsal"},
                        "promote" if gate["promote"] else "kill",
                        {"f1_M0": f1m(y, preds["M0"]),
                         "f1_M1": f1m(y, preds["M1"]), **gate})

    # slip recovery on bearing runs
    b = d[d.cls == "bearing"]
    err = (b.h4_slip - b.true_slip).abs()
    print(f"\nslip recovery: median |err| = {err.median()*100:.3f} pp "
          f"(true slips 0.8-1.8%)")
    print(f"F1 M0={rec['metrics']['f1_M0']:.3f}  M1={rec['metrics']['f1_M1']:.3f}  "
          f"delta_mean={gate['delta_mean']:+.3f}  "
          f"lo(alpha={gate['alpha']:.3f})={gate['delta_lo_alpha_adj']:+.3f}  "
          f"verdict={rec['verdict'].upper()}")
    print(f"PPA z: bearing median={d[d.cls=='bearing'].h4_ppa.median():.1f}, "
          f"others median={d[d.cls!='bearing'].h4_ppa.median():.1f}")
    print(f"ledger entries: {M.n_attempts()}, "
          f"wall time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
