#!/usr/bin/env python3
"""Stage 4 (PHASE3): sim-vs-real discriminator realism gate.

For each real dataset's HEALTHY records, train a GBM to distinguish
sim windows from real windows using unit/fs-invariant shape features
(shorcm.baselines.shape_features). Grouped CV by record. Report AUC
for SimForge v2(v5-models corpus) and SimForge v6 against the SAME
real windows, plus top discriminating features (the actionable gap
list). Contract P3-33-D1: AUC(v6) <= AUC(v5) - 0.05.

Realism target band (PHASE2): 0.5-0.8. AUC ~1.0 = trivially fake.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import simforge_v2 as V2                  # noqa: E402
from shorcm import simforge_v6 as V6                  # noqa: E402
from shorcm import baselines as BL                    # noqa: E402
from shorcm import adapters as AD                     # noqa: E402

OUT = Path("experiments/simgap")
N_SIM = 120                     # healthy sim machines per forge


def sim_rows(forge, tag):
    rows, found = [], 0
    i = 0
    while found < N_SIM and i < 20000:
        rng = forge.rng_for_run(i)
        m = V2.sample_machine(rng)
        if m["fault"] == "healthy":
            _, x = forge.sample_run(i)
            for d in BL.shape_features(x, V2.FS):
                d["group"] = f"{tag}_{i}"
                d["src"] = tag
                rows.append(d)
            found += 1
        i += 1
    return rows


def real_rows():
    rows = []

    def add(x, fs, group, ds):
        for d in BL.shape_features(x, fs):
            d["group"] = group
            d["src"] = "real"
            d["dataset"] = ds
            rows.append(d)

    # CWRU normals (97-100.mat under Normal)
    import scipy.io as sio
    for p in sorted(Path("data/cwru/Normal").rglob("*.mat")):
        m = sio.loadmat(p, squeeze_me=True)
        key = [k for k in m if k.endswith("DE_time")]
        if not key:
            continue
        x = np.asarray(m[key[0]], float).ravel()
        fs = 48000.0                     # CWRU normal baselines are 48 k
        add(x, fs, f"cwru_{p.stem}", "cwru")
    # MFPT baselines
    for p in sorted(Path("data/mfpt").glob("baseline*.mat")):
        x, fs, _, _ = AD.from_mfpt_mat(p)
        add(x, fs, f"mfpt_{p.stem}", "mfpt")
    # SEU healthy
    for p in sorted(Path("data/seu").glob("*health*.csv")):
        x, fs, _, _ = AD.from_seu_csv(p, dur=16.0)
        add(x, fs, f"seu_{p.stem}", "seu")
    # Wind turbine days 1-6 (pre-degradation)
    for p in sorted(Path("data/windturbine").glob("data-*.mat"))[:6]:
        x, fs, _, _ = AD.from_windturbine_mat(p)
        add(x, fs, f"wt_{p.stem}", "wt")
    return rows


def auc_grouped(df, sim_tag):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    d = df[df.src.isin([sim_tag, "real"])].reset_index(drop=True)
    feats = [c for c in d.columns if c not in
             ("src", "group", "dataset")]
    X = d[feats].values
    y = (d.src == "real").astype(int).values
    g = d.group.values
    oof = np.full(len(d), np.nan)
    imp = np.zeros(len(feats))
    for tr, te in GroupKFold(5).split(X, y, g):
        mdl = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                 num_leaves=31, random_state=20260709,
                                 verbose=-1)
        mdl.fit(X[tr], y[tr])
        oof[te] = mdl.predict_proba(X[te])[:, 1]
        imp += mdl.feature_importances_
    auc = float(roc_auc_score(y, oof))
    top = sorted(zip(feats, imp), key=lambda t: -t[1])[:10]
    return auc, [{"feature": f, "gain": float(v)} for f, v in top]


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    real = real_rows()
    v2r = sim_rows(V2, "v2")
    v6r = sim_rows(V6, "v6")
    df = pd.DataFrame(real + v2r + v6r).fillna(0.0)
    n_real = int((df.src == "real").sum())
    res = {"n_real_windows": n_real,
           "n_real_records": int(df[df.src == "real"].group.nunique()),
           "real_by_dataset": df[df.src == "real"].groupby(
               "dataset").size().to_dict(),
           "n_sim_windows_v2": int((df.src == "v2").sum()),
           "n_sim_windows_v6": int((df.src == "v6").sum())}
    auc2, top2 = auc_grouped(df, "v2")
    auc6, top6 = auc_grouped(df, "v6")
    res["auc_v2_vs_real"] = round(auc2, 4)
    res["auc_v6_vs_real"] = round(auc6, 4)
    res["contract_P3-33-D1"] = {
        "target": "auc_v6 <= auc_v2 - 0.05",
        "delta": round(auc2 - auc6, 4),
        "verdict": "MET" if auc6 <= auc2 - 0.05 else "MISSED"}
    res["realism_band_0.5_0.8"] = {
        "v2_in_band": bool(0.5 <= auc2 <= 0.8),
        "v6_in_band": bool(0.5 <= auc6 <= 0.8)}
    res["top_discriminating_v2"] = top2
    res["top_discriminating_v6"] = top6
    res["wall_s"] = round(time.time() - t0, 1)
    (OUT / "findings.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
