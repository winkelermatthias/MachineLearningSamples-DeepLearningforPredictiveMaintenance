#!/usr/bin/env python3
"""H4: Grover-style amplification features (slip_scan + PPA z-scores at
bearing kinematic orders) vs the H1 incumbent (coherence ratios).

Design, dev-set only (vault sealed out):
- For each run x variant, compute per bearing order in
  {FTF 0.375, BSF 1.994, BPFO 2.998, BPFI 5.002} x best channel:
  slip, sharpness, amplitude (slip_scan) and PPA z-score at the nearest
  block bin.
- Model M0: H1 RATIO features. Model M1: RATIO + H4 features.
- Metric: 7-way macro-F1, grouped CV. Promotion via
  metrics.promotion_gate on paired bootstrap deltas; attempt logged to
  the ledger either way.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import amplify as AM                     # noqa: E402
from shorcm import spectra as S                      # noqa: E402
from shorcm import tacho as T                        # noqa: E402
from shorcm import metrics as M                      # noqa: E402
from shorcm.features import (FS, SPR, REVS, CH, ACCEL_CHANNELS,
                             BEARING_ORDERS, SELF_REF_MASK,
                             pick_ref_channel)       # noqa: E402
from shorcm.splits import grouped_folds              # noqa: E402

OUTD = Path("experiments/h4"); OUTD.mkdir(parents=True, exist_ok=True)

try:
    from lightgbm import LGBMClassifier
    def model(): return LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                       random_state=20260709, verbosity=-1)
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    def model(): return HistGradientBoostingClassifier(random_state=20260709)


def h4_features(run, f_nom, variant):
    ref = pick_ref_channel(run, f_nom)
    phase, meta = T.get_phase(variant, run, FS, f_nom, ref_channel=ref)
    out = {}
    for name in ACCEL_CHANNELS[:2]:  # two channels keep runtime sane
        x = run[:, CH[name]]
        xa = S.angular_resample(x, phase, SPR)
        Z, orders = S.block_spectra(xa, SPR, REVS)
        for nm, bo in BEARING_ORDERS.items():
            sc = AM.slip_scan(xa, SPR, bo, span=0.03, n=301)
            b = int(round(bo * REVS))
            zs, *_ = AM.ppa_zscore(Z[:, b], n_perm=100)
            for k, v in (("slip", sc["slip"]), ("sharp", sc["sharpness"]),
                         ("amp", sc["amp"]), ("ppa", zs)):
                key = f"h4_{nm}_{k}"
                out[key] = max(out.get(key, -np.inf), float(v)) \
                    if k != "slip" else out.get(key, float(v))
    return out


def main():
    feats = pd.read_parquet("features/tabular/features.parquet")
    feats = feats[feats["error"] == ""]
    cat = pd.read_parquet("data/catalog.parquet").set_index("run_id")
    dev = ~M.vault_mask(feats.run_id.values)          # SEAL THE VAULT
    feats = feats[dev].reset_index(drop=True)
    results = []
    for variant in ["tacho", "onex", "comb", "nominal"]:
        df = feats[feats.variant == variant].reset_index(drop=True)
        # compute H4 features (cached per run in h4_cache.parquet)
        cache = OUTD / f"h4_cache_{variant}.parquet"
        if cache.exists():
            h4 = pd.read_parquet(cache)
        else:
            rows = []
            for _, r in df.iterrows():
                arr = pd.read_parquet(cat.loc[r.run_id, "path"]) \
                    .values.astype(float)
                d = h4_features(arr, r.f_nom, variant)
                d["run_id"] = r.run_id
                rows.append(d)
            h4 = pd.DataFrame(rows)
            h4.to_parquet(cache, index=False)
        df = df.merge(h4, on="run_id")
        df["label"] = np.where(df.cls.isin(["underhang", "overhang"]),
                               df.cls + ":" + df["sub"], df.cls)
        df["y"] = pd.factorize(df.label)[0]
        drop = set(SELF_REF_MASK.get(variant, []))
        base_cols = [c for c in df if c.startswith(("ratio_", "bear_ratio_"))
                     and c not in drop]
        h4_cols = [c for c in df if c.startswith("h4_")]
        preds = {}
        for tag, cols in (("M0", base_cols), ("M1", base_cols + h4_cols)):
            p = np.full(len(df), -1)
            X = df[cols].replace([np.inf, -np.inf], 0).fillna(0).values
            for tr, te in grouped_folds(df):
                p[te] = model().fit(X[tr], df.y.values[tr]).predict(X[te])
            preds[tag] = p
        y = df.y.values
        f1 = lambda yy, pp: f1_score(yy, pp, average="macro")  # noqa: E731
        deltas = M.paired_bootstrap_delta(y, preds["M0"], preds["M1"], f1)
        gate = M.promotion_gate(deltas, guardrails_ok=True)
        rec = M.log_attempt(f"H4_amplify_{variant}",
                            {"variant": variant, "cols": len(h4_cols)},
                            "promote" if gate["promote"] else "kill",
                            {"f1_M0": f1(y, preds["M0"]),
                             "f1_M1": f1(y, preds["M1"]), **gate})
        results.append({"variant": variant, **rec["metrics"]})
    res = pd.DataFrame(results)
    res.to_parquet(OUTD / "results.parquet", index=False)
    (OUTD / "summary.md").write_text(
        "# H4 summary\n\n" + res.to_markdown(index=False) +
        "\n\nBonus physics readout: h4_*_slip columns give per-run bearing "
        "slip estimates. Check slip vs severity (mass) monotonicity: slip "
        "grows with load in real bearings.\n")
    print(res)


if __name__ == "__main__":
    main()
