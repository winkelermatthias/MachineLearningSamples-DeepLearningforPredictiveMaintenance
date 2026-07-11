#!/usr/bin/env python3
"""H2: continued-fraction rational fingerprints improve multiclass fault
typing over raw order features, most at low RPM. Per phase variant.

Feature sets (nested): A = inc order amps, B = A + coherence ratios,
C = B + CF fingerprint columns + unsnapped residuals.
Model: LightGBM if available else GradientBoosting. McNemar B vs C per fold,
Holm-corrected across variants.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm.splits import grouped_folds            # noqa: E402
from shorcm.features import SELF_REF_MASK          # noqa: E402

OUTD = Path("experiments/h2"); OUTD.mkdir(parents=True, exist_ok=True)

try:
    from lightgbm import LGBMClassifier
    def model(): return LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                       num_leaves=31, random_state=20260709,
                                       verbosity=-1)
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    def model(): return HistGradientBoostingClassifier(random_state=20260709)


def cols_for(df, fset, variant):
    drop = set(SELF_REF_MASK.get(variant, []))
    a = [c for c in df if c.startswith(("inc_", "bear_inc_"))]
    b = a + [c for c in df if c.startswith(("ratio_", "bear_ratio_"))
             and c not in drop]
    c = b + [c_ for c_ in df if c_.startswith(("cf_", "uns_"))] + \
        ["unsnapped_energy", "n_unsnapped"]
    return {"A": a, "B": b, "C": c}[fset]


def main():
    all_f = pd.read_parquet("features/tabular/features.parquet")
    all_f = all_f[all_f["error"] == ""]
    rows, mcn = [], []
    for variant in ["tacho", "onex", "comb", "nominal"]:
        df = all_f[all_f.variant == variant].reset_index(drop=True)
        # 7-way: split bearing classes by fault type
        df["label"] = np.where(df.cls.isin(["underhang", "overhang"]),
                               df.cls + ":" + df["sub"], df.cls)
        df["y"] = pd.factorize(df.label)[0]
        preds = {f: np.full(len(df), -1) for f in "ABC"}
        for tr, te in grouped_folds(df):
            for fset in "ABC":
                X = df[cols_for(df, fset, variant)].fillna(0).values
                m = model().fit(X[tr], df.y.values[tr])
                preds[fset][te] = m.predict(X[te])
        y = df.y.values
        lowrpm = df.rate_hz.values < 20.0
        for fset in "ABC":
            p = preds[fset]
            rows.append(dict(variant=variant, fset=fset,
                             acc=accuracy_score(y, p),
                             f1=f1_score(y, p, average="macro"),
                             acc_lowrpm=accuracy_score(y[lowrpm], p[lowrpm])
                             if lowrpm.sum() > 10 else np.nan,
                             n_lowrpm=int(lowrpm.sum())))
        # McNemar B vs C
        b_ok = preds["B"] == y
        c_ok = preds["C"] == y
        tab = [[np.sum(b_ok & c_ok), np.sum(b_ok & ~c_ok)],
               [np.sum(~b_ok & c_ok), np.sum(~b_ok & ~c_ok)]]
        p = mcnemar(tab, exact=False, correction=True).pvalue
        mcn.append(dict(variant=variant, p_raw=float(p),
                        b_only=int(tab[0][1]), c_only=int(tab[1][0])))
    # Holm correction across the 4 variant tests
    mc = pd.DataFrame(mcn).sort_values("p_raw").reset_index(drop=True)
    m = len(mc)
    mc["p_holm"] = [min(1.0, mc.p_raw[i] * (m - i)) for i in range(m)]
    res = pd.DataFrame(rows)
    res.to_parquet(OUTD / "results.parquet", index=False)
    mc.to_parquet(OUTD / "mcnemar.parquet", index=False)
    piv = res.pivot_table(index="variant", columns="fset", values="acc")
    low = res.pivot_table(index="variant", columns="fset", values="acc_lowrpm")
    (OUTD / "summary.md").write_text(
        "# H2 summary\n\n## accuracy (7-way, grouped CV)\n" + piv.to_markdown()
        + "\n\n## accuracy at rate < 20 Hz\n" + low.to_markdown()
        + "\n\n## McNemar B vs C (Holm)\n" + mc.to_markdown()
        + "\n\nPass: C > B with p_holm < 0.05 on tacho, and the C-B gap is "
          "larger in the low-RPM slice.\n")
    print(piv)


if __name__ == "__main__":
    main()
