#!/usr/bin/env python3
"""O3 ML arm + program metrics on the massive synthetic corpus.

Consumes experiments/massive/{manifest,o1_results,ledger}.parquet.

 1. O1 confidence calibration: reliability curve + ECE of top-1
    confidence (full system).
 2. O1 degradation vs one-sided hardening (extra noise / extra floor):
    the confuser degradation curve PHASE2 pre-registers.
 3. O3 ML arm: LightGBM on the physics ledger (never raw bins),
    GroupKFold by speed band (shorcm.splits discipline), compared
    against the transparent rules arm through the paired-bootstrap
    promotion gate (metrics.promotion_gate, attempts from ledger).
 4. Coverage-accuracy curve via margin abstention (rules arm = point).
 5. O4 severity: Spearman rho of physical drivers vs severity latent
    per fault + controlled paired severity sweeps (same machine, same
    noise realization, severity varied) with monotonicity violation
    counts. Guardrail: ZERO violations tolerated on paired sweeps.

Outputs: experiments/massive/ml_findings.json + figures + parquet.
ENV: WORKERS (default 4) for the paired severity sweeps.
"""
import json, os, sys, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import simforge_lite as SF               # noqa: E402
from shorcm import simforge_corpus as SC             # noqa: E402
from shorcm import splits as SP                      # noqa: E402
from shorcm import metrics as M                      # noqa: E402

OUT = Path("experiments/massive")
SEED = 20260709
WORKERS = int(os.environ.get("WORKERS", 4))
FAULTS = SF.FAULTS
DRIVERS = {"imbalance": "a1", "misalignment": "a2_over_a1",
           "looseness": "e_half", "bearing": "uns_frac"}


def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(conf)
    rows = []
    for i in range(bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1] + (i == bins - 1))
        if m.sum() == 0:
            continue
        acc, c = correct[m].mean(), conf[m].mean()
        e += m.sum() / n * abs(acc - c)
        rows.append((c, acc, int(m.sum())))
    return e, rows


def o1_analysis(o1, man):
    fu = o1[o1.ablation == "full"].merge(man, on="run_id")
    ec, rel = ece(fu.conf.values, fu.top1.values.astype(float))
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot([0, 1], [0, 1], "--", c="gray", lw=1)
    xs = [r[0] for r in rel]; ys = [r[1] for r in rel]
    ax.plot(xs, ys, "o-", c="#0FB5A6")
    ax.set_xlabel("stated confidence"); ax.set_ylabel("observed top-1 rate")
    ax.set_title(f"O1 reliability, ECE={ec:.3f} (uncalibrated softmax)",
                 fontsize=10)
    fig.savefig(OUT / "o1_reliability.png", dpi=140, bbox_inches="tight")

    # degradation vs one-sided hardening
    fu["noise_bin"] = pd.cut(fu.extra_noise, [0, .05, .10, .15, .20, .25],
                             include_lowest=True)
    deg_n = fu.groupby("noise_bin", observed=True)[["top1", "top3"]].mean()
    fu["floor_bin"] = pd.cut(fu.extra_floor, [0, .3, .6, .9, 1.2, 1.5],
                             include_lowest=True)
    deg_f = fu.groupby("floor_bin", observed=True)[["top1", "top3"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
    for ax, d, lab in ((axes[0], deg_n, "extra broadband noise"),
                       (axes[1], deg_f, "extra shaped floor gain")):
        x = np.arange(len(d))
        ax.plot(x, d.top1, "o-", label="top-1", c="#0FB5A6")
        ax.plot(x, d.top3, "s-", label="top-3", c="#666")
        ax.set_xticks(x, [str(i) for i in d.index], rotation=30, fontsize=7)
        ax.set_xlabel(lab); ax.legend(fontsize=8)
    axes[0].set_ylabel("rate (tol 1%)")
    fig.suptitle("O1 degradation under one-sided hardening", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "o1_degradation.png", dpi=140, bbox_inches="tight")

    by_abl = o1.groupby("ablation")[["top1", "top3", "octave"]].mean()
    return {"n_runs": int(fu.shape[0]), "ece": round(float(ec), 4),
            "by_ablation": {a: {k: round(float(v), 4) for k, v in row.items()}
                            for a, row in by_abl.iterrows()},
            "by_population_component": {
                f"{p}_{c}": round(float(v), 4)
                for (p, c), v in fu.groupby(["population", "component"])
                .top1.mean().items()},
            "by_fault_top1": {f: round(float(v), 4) for f, v in
                              fu.groupby("fault").top1.mean().items()},
            "conf_correct": round(float(fu[fu.top1].conf.mean()), 3),
            "conf_wrong": round(float(fu[~fu.top1].conf.mean()), 3)}


def ml_arm(led, man, o1):
    df = led.merge(man, on="run_id")
    fu = o1[o1.ablation == "full"][["run_id", "top1"]]
    df = df.merge(fu, on="run_id")
    feats = SC.LEDGER_FEATURES
    df = df.dropna(subset=feats)
    X = df[feats].values
    y = pd.Categorical(df.fault, categories=FAULTS).codes
    groups = SP.speed_band(df.f_shaft.values)

    import lightgbm as lgb
    oof = np.full(len(df), -1)
    oof_margin = np.zeros(len(df))
    gkf = GroupKFold(n_splits=5)
    imps = np.zeros(len(feats))
    for tr, te in gkf.split(X, y, groups):
        mdl = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                 num_leaves=31, min_child_samples=20,
                                 random_state=SEED, verbose=-1)
        mdl.fit(X[tr], y[tr])
        p = mdl.predict_proba(X[te])
        oof[te] = p.argmax(1)
        ps = np.sort(p, axis=1)
        oof_margin[te] = ps[:, -1] - ps[:, -2]
        imps += mdl.feature_importances_ / 5
    y_named = df.fault.values
    ml_named = np.array(FAULTS)[oof]
    rules_named = df.rules_pred.values

    def macro_f1(a, b):
        return f1_score(a, b, average="macro",
                        labels=[f for f in FAULTS], zero_division=0)

    res = {}
    for cond, mask in (("all", np.ones(len(df), bool)),
                       ("speed_correct", df.top1.values),
                       ("speed_wrong", ~df.top1.values)):
        res[cond] = {
            "n": int(mask.sum()),
            "rules_acc": round(float((rules_named[mask] == y_named[mask])
                                     .mean()), 4),
            "ml_acc": round(float((ml_named[mask] == y_named[mask]).mean()),
                            4),
            "rules_macro_f1": round(float(macro_f1(y_named[mask],
                                                   rules_named[mask])), 4),
            "ml_macro_f1": round(float(macro_f1(y_named[mask],
                                                ml_named[mask])), 4)}

    # promotion gate: ML arm vs rules arm on macro-F1, paired bootstrap
    deltas = M.paired_bootstrap_delta(y_named, rules_named, ml_named,
                                      macro_f1, n=2000)
    gate = M.promotion_gate(deltas, guardrails_ok=True, attempts=1)

    # coverage-accuracy by margin abstention
    order = np.argsort(-oof_margin)
    covs = np.linspace(0.2, 1.0, 33)
    curve = []
    for c in covs:
        k = max(int(c * len(df)), 1)
        sel = order[:k]
        curve.append((float(c),
                      float((ml_named[sel] == y_named[sel]).mean()),
                      float(macro_f1(y_named[sel], ml_named[sel]))))
    pd.DataFrame(curve, columns=["coverage", "acc", "macro_f1"]) \
        .to_parquet(OUT / "coverage_accuracy.parquet", index=False)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot([c for c, _, _ in curve], [a for _, a, _ in curve],
            c="#0FB5A6", label="ML acc")
    ax.plot([c for c, _, _ in curve], [f for _, _, f in curve],
            c="#0FB5A6", ls="--", label="ML macro-F1")
    ax.axhline(res["all"]["rules_acc"], c="#666", lw=1,
               label="rules acc (full coverage)")
    ax.set_xlabel("coverage"); ax.set_ylabel("metric")
    ax.set_title("O3 coverage-accuracy, margin abstention", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(OUT / "o3_coverage_accuracy.png", dpi=140,
                bbox_inches="tight")

    cm = confusion_matrix(y_named, ml_named, labels=FAULTS)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(FAULTS)), FAULTS, rotation=30, fontsize=8)
    ax.set_yticks(range(len(FAULTS)), FAULTS, fontsize=8)
    for i in range(len(FAULTS)):
        for j in range(len(FAULTS)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("ML predicted"); ax.set_ylabel("true")
    ax.set_title("O3 ML arm, grouped 5-fold OOF", fontsize=10)
    fig.colorbar(im, shrink=0.8)
    fig.savefig(OUT / "o3_confusion.png", dpi=140, bbox_inches="tight")

    fi = sorted(zip(SC.LEDGER_FEATURES, imps), key=lambda t: -t[1])[:8]
    return {"conditions": res,
            "promotion_gate_ml_over_rules": gate,
            "top_features": [(f, round(float(v), 1)) for f, v in fi],
            "confusion_labels": FAULTS,
            "confusion": cm.tolist()}


# ---------- O4 severity: observational rho + paired sweeps ----------

def _paired_sweep(args):
    i, sev_grid = args
    rng0 = SC.rng_for_run(10_000_000 + i)
    m = SF.sample_machine(rng0)
    if m["fault"] == "healthy":
        m["fault"] = ["imbalance", "misalignment", "looseness",
                      "bearing"][i % 4]
    vals = []
    for s in sev_grid:
        mm = dict(m); mm["severity"] = float(s)
        rng = SC.rng_for_run(20_000_000 + i)     # same noise realization
        x = SF.synth_run(mm, rng)
        led = SC.ledger(x, SF.FS, mm["f_shaft"])  # true speed: isolates O4
        vals.append(led[DRIVERS[mm["fault"]]] if led else np.nan)
    return m["fault"], vals


def severity(led, man, pool):
    df = led.merge(man, on="run_id")
    obs = {}
    for f, drv in DRIVERS.items():
        d = df[(df.fault == f)].dropna(subset=[drv])
        if len(d) > 10:
            rho = spearmanr(d.severity, d[drv]).statistic
            obs[f] = {"driver": drv, "n": int(len(d)),
                      "spearman_rho": round(float(rho), 3)}
    sev_grid = np.linspace(0.15, 1.0, 8)
    res = pool.map(_paired_sweep, [(i, sev_grid) for i in range(48)])
    viol, tot, curves = 0, 0, {}
    for f, vals in res:
        v = np.asarray(vals, float)
        if np.isnan(v).any():
            continue
        tot += 1
        viol += int(np.any(np.diff(v) < -0.02 * (np.abs(v[:-1]) + 1e-9)))
        curves.setdefault(f, []).append(v)
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8))
    for ax, (f, cs) in zip(axes, sorted(curves.items())):
        for c in cs:
            ax.plot(sev_grid, c / (c[-1] + 1e-12), c="#0FB5A6", alpha=0.35)
        ax.set_title(f"{f} -> {DRIVERS[f]}", fontsize=9)
        ax.set_xlabel("severity latent")
    axes[0].set_ylabel("driver (norm.)")
    fig.suptitle("O4 paired severity sweeps, same machine + noise",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "o4_monotonicity.png", dpi=140, bbox_inches="tight")
    return {"observational_rho": obs,
            "paired_sweeps": {"n_machines": tot,
                              "violations": viol,
                              "grid": [round(float(s), 3)
                                       for s in sev_grid]}}


def main():
    t0 = time.time()
    man = pd.read_parquet(OUT / "manifest.parquet")
    o1 = pd.read_parquet(OUT / "o1_results.parquet")
    led = pd.read_parquet(OUT / "ledger.parquet")
    with Pool(WORKERS, maxtasksperchild=50) as pool:
        find = {"o1": o1_analysis(o1, man),
                "o3": ml_arm(led, man, o1),
                "o4": severity(led, man, pool)}
    find["wall_s"] = round(time.time() - t0, 1)
    (OUT / "ml_findings.json").write_text(json.dumps(find, indent=1,
                                                     default=str))
    print(json.dumps(find, indent=1, default=str))


if __name__ == "__main__":
    main()
