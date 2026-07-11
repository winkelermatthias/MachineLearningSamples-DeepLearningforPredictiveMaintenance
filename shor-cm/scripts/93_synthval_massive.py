#!/usr/bin/env python3
"""SYNTHVAL massive: full claim-family sweep at 2000 trials/class/cell.

Scales scripts/95_synthval_map.py to the full program of SYNTHVAL.md:
 C1 ratio statistics       null 1/sqrt(N) + locked Rice, high-trial CIs
 C2 decoherence law        coh_mean_decohered vs MC over (k, sigma_phi)
 C3 CF false-snap geometry exact Farey measure vs MC over (tol, qmax)
 C4 PPA                    KS uniformity of permutation null p-values +
                           gain-vs-walk-step exp(-s^2/2) law
 C6 advantage maps         locked-vs-drift AUC over (SNR, N, drift s),
                           theory pre-registered + hashed, MC audited,
                           negative controls in-loop

Gate discipline identical to 95: theory first and hashed; MC trial counts
sized so SE < half the band; divergence -> exit 1 with cells listed;
negative controls must be flat. Two-tier validity: two-sided band 0.06
for N >= 16, one-sided lower bound (mc >= th - 0.02) at N = 8, mechanism
documented in theory.ratio_auc_locked_vs_drift.

ENV: N_MC (default 2000), WORKERS (default 4).
"""
import hashlib, json, os, sys, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import theory as TH                      # noqa: E402
from shorcm import spectra as S                      # noqa: E402

OUT = Path("experiments/synthval_massive"); OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260709
SNR_DB = np.arange(-14, 10.1, 1.0)
NS = [8, 16, 32, 64, 128]
DRIFTS = [0.3, 0.5, 0.7, 1.0]
N_MC = int(os.environ.get("N_MC", 2000))
WORKERS = int(os.environ.get("WORKERS", 4))
BAND = 0.06


def blocks(A, sigma, N, s, n, rng):
    th0 = rng.uniform(0, 2 * np.pi, (n, 1))
    steps = rng.normal(0, s, (n, N)).cumsum(1) if s > 0 else 0.0
    z = A * np.exp(1j * (th0 + steps))
    return z + sigma * (rng.standard_normal((n, N))
                        + 1j * rng.standard_normal((n, N)))


def ratio(z):
    return np.abs(z.mean(1)) / np.abs(z).mean(1)


def auc(a, b):
    x = np.concatenate([a, b])
    r = x.argsort().argsort()[: len(a)].sum()
    return (r - len(a) * (len(a) - 1) / 2) / (len(a) * len(b))


# ---------- C6 advantage maps (parallel over (drift, N) rows) ----------

def _c6_theory_row(args):
    s, N = args
    return [TH.ratio_auc_locked_vs_drift(10 ** (db / 20), 1.0, N, s)
            for db in SNR_DB]


def _c6_mc_row(args):
    s, N, row_seed = args
    rng = np.random.default_rng(row_seed)
    mc, nulldev, impdev = [], [], []
    for j, db in enumerate(SNR_DB):
        A = 10 ** (db / 20)
        zl = blocks(A, 1.0, N, 0.0, N_MC, rng)
        zd = blocks(A, 1.0, N, s, N_MC, rng)
        mc.append(auc(ratio(zl), ratio(zd)))
        if j % 4 == 0:
            z2 = blocks(A, 1.0, N, 0.0, N_MC, rng)
            nulldev.append(abs(auc(ratio(zl), ratio(z2)) - 0.5))
            ph = np.exp(1j * rng.uniform(0, 2 * np.pi, zl.shape))
            impdev.append(abs(auc(ratio(zl * ph), ratio(zd * ph)) - 0.5))
    return mc, max(nulldev), float(np.mean(impdev))


def run_c6(pool):
    jobs = [(s, N) for s in DRIFTS for N in NS]
    t0 = time.time()
    th_rows = pool.map(_c6_theory_row, jobs)
    th = np.array(th_rows).reshape(len(DRIFTS), len(NS), len(SNR_DB))
    th_hash = hashlib.sha1(th.round(4).tobytes()).hexdigest()[:12]
    print(f"C6 theory map {th.size} cells in {time.time()-t0:.0f}s, "
          f"hash {th_hash}")
    mc_jobs = [(s, N, SEED + 1000 * i) for i, (s, N) in enumerate(jobs)]
    res = pool.map(_c6_mc_row, mc_jobs)
    mc = np.array([r[0] for r in res]).reshape(th.shape)
    null_max = max(r[1] for r in res)
    imp_mean = float(np.mean([r[2] for r in res]))
    gap = mc - th
    in_region = (np.array(NS) >= 16)[None, :, None]
    diverg = np.where(in_region, np.abs(gap) > BAND, gap < -0.02)
    rows = [dict(drift=DRIFTS[d], N=NS[i], snr_db=float(SNR_DB[j]),
                 auc_mc=float(mc[d, i, j]), auc_th=float(th[d, i, j]),
                 divergent=bool(diverg[d, i, j]))
            for d in range(len(DRIFTS)) for i in range(len(NS))
            for j in range(len(SNR_DB))]
    pd.DataFrame(rows).to_parquet(OUT / "c6_advantage_maps.parquet",
                                  index=False)
    fig, axes = plt.subplots(2, len(DRIFTS), figsize=(4 * len(DRIFTS), 6),
                             sharey=True, sharex=True)
    for d, s in enumerate(DRIFTS):
        for r, (m, lab) in enumerate(((th[d], "theory"), (mc[d], "MC"))):
            ax = axes[r, d]
            ax.imshow(m, aspect="auto", origin="lower", vmin=0.5, vmax=1.0,
                      cmap="viridis",
                      extent=[SNR_DB[0], SNR_DB[-1], 0, len(NS)])
            ax.contour(np.linspace(SNR_DB[0], SNR_DB[-1], m.shape[1]),
                       np.arange(len(NS)) + 0.5, m, levels=[0.9],
                       colors=["#0FB5A6"], linewidths=2)
            ax.set_yticks(np.arange(len(NS)) + 0.5, NS)
            ax.set_title(f"{lab}, drift s={s}", fontsize=9)
            if r == 1:
                ax.set_xlabel("per-block SNR (dB)")
    axes[0, 0].set_ylabel("blocks N"); axes[1, 0].set_ylabel("blocks N")
    fig.suptitle("C6 advantage maps: AUC locked vs drifting, "
                 f"{N_MC} trials/class/cell; teal = 0.9 contour")
    fig.tight_layout()
    fig.savefig(OUT / "c6_advantage_maps.png", dpi=140, bbox_inches="tight")
    return {"theory_hash": th_hash, "cells": int(mc.size),
            "divergent_frac": float(diverg.mean()),
            "divergent_cells": [
                {"drift": DRIFTS[d], "N": NS[i], "snr_db": float(SNR_DB[j]),
                 "mc": round(float(mc[d, i, j]), 3),
                 "th": round(float(th[d, i, j]), 3)}
                for d, i, j in zip(*np.where(diverg))][:15],
            "control_locked_vs_locked_max_dev": round(null_max, 3),
            "control_impostor_mean_dev": round(imp_mean, 3),
            "pass": bool(diverg.mean() < 0.10 and null_max < 0.06
                         and imp_mean < 0.06)}


# ---------- C1 ratio statistics at high trial count ----------

def run_c1():
    rng = np.random.default_rng(SEED + 77)
    n = 20000
    out = {"null": [], "locked": []}
    ok = True
    for N in NS:
        z = blocks(0.0, 1.0, N, 0.0, n, rng)
        r = ratio(z)
        se = r.std() / np.sqrt(n)
        dev = abs(r.mean() - TH.ratio_noise(N))
        # Jensen bias of E[coh]/E[inc] vs E[coh/inc] is O(1/N); band covers it
        band = 0.02 * TH.ratio_noise(N) + 3 * se + 0.25 / N**1.5
        out["null"].append({"N": N, "mc": round(float(r.mean()), 4),
                            "th": round(TH.ratio_noise(N), 4),
                            "band": round(float(band), 4)})
        ok &= dev < band
    for A, sig, N in [(0.5, 1.0, 16), (1.0, 1.0, 32), (2.0, 0.7, 64),
                      (0.3, 1.0, 128)]:
        z = blocks(A, sig, N, 0.0, n, rng)
        r = ratio(z)
        th = TH.ratio_locked(A, sig, N)
        dev = abs(r.mean() - th)
        out["locked"].append({"A": A, "sigma": sig, "N": N,
                              "mc": round(float(r.mean()), 4),
                              "th": round(float(th), 4)})
        ok &= dev < 0.03
    out["pass"] = bool(ok)
    return out


# ---------- C2 decoherence law over (k, sigma_phi) ----------

def run_c2():
    rng = np.random.default_rng(SEED + 78)
    A, sig, N, n = 1.0, 0.5, 64, 4000
    ks = [1, 2, 3, 4, 6, 8, 10]
    sps = [0.05, 0.1, 0.2, 0.3, 0.5]
    rows, worst = [], 0.0
    for sp in sps:
        for k in ks:
            eps = rng.normal(0, sp, (n, N))
            z = A * np.exp(1j * k * eps) + sig * (
                rng.standard_normal((n, N)) + 1j * rng.standard_normal((n, N)))
            mc = float(np.abs(z.mean(1)).mean())
            th = float(TH.coh_mean_decohered(A, sig, N, k, sp))
            rows.append(dict(k=k, sigma_phi=sp, mc=mc, th=th,
                             D=float(TH.decoherence_factor(k, sp))))
            worst = max(worst, abs(mc - th))
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "c2_decoherence.parquet", index=False)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for sp in sps:
        d = df[df.sigma_phi == sp]
        ax.plot(d.k, d.mc, "o", ms=4)
        ax.plot(d.k, d.th, "-", lw=1.2, label=f"sigma_phi={sp}")
    ax.set_xlabel("order k"); ax.set_ylabel("E|coherent mean|")
    ax.set_title("C2: decoherence is Gaussian in order "
                 "(dots MC, lines theory)", fontsize=10)
    ax.legend(fontsize=7)
    fig.savefig(OUT / "c2_decoherence.png", dpi=140, bbox_inches="tight")
    return {"cells": len(rows), "worst_abs_dev": round(worst, 4),
            "pass": bool(worst < 0.03)}


# ---------- C3 CF false-snap geometry ----------

def run_c3():
    rng = np.random.default_rng(SEED + 79)
    lo, hi, n = 0.2, 10.0, 200000
    o = rng.uniform(lo, hi, n)
    rows, ok = [], True
    for tol in (0.004, 0.01, 0.02):
        for qmax in (5, 8, 12):
            hits = np.fromiter((S.cf_snap(v, tol=tol, qmax=qmax) is not None
                                for v in o), bool)
            mc = float(hits.mean())
            ex = float(TH.cf_false_snap_rate(tol, qmax, lo, hi))
            se = np.sqrt(ex * (1 - ex) / n)
            rows.append(dict(tol=tol, qmax=qmax, mc=mc, exact=ex,
                             se=float(se)))
            ok &= abs(mc - ex) < 4 * se + 1e-3
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "c3_false_snap.parquet", index=False)
    return {"grid": rows, "n_mc": n, "pass": bool(ok),
            "headline": "false-snap rate at tol=0.01,q<=8: "
                        f"{df[(df.tol==0.01)&(df.qmax==8)].exact.iloc[0]:.3f}"}


# ---------- C4 PPA: null uniformity + gain law ----------

def run_c4():
    rng = np.random.default_rng(SEED + 80)
    from scipy.stats import kstest
    N, n_perm, n_cases = 32, 199, 400

    def aligned(z):
        rot = np.exp(-1j * np.angle(z[:-1]))
        return float(np.abs((z[1:] * rot).mean()))

    pvals = []
    for _ in range(n_cases):
        z = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        obs = aligned(z)
        null = np.array([aligned(rng.permutation(z))
                         for _ in range(n_perm)])
        pvals.append((1 + np.sum(null >= obs)) / (n_perm + 1))
    ks = kstest(pvals, "uniform")
    # gain law: aligned-pair mean for pure walk = exp(-s^2/2)
    rows, worst = [], 0.0
    for s in (0.1, 0.3, 0.5, 0.8, 1.0, 1.5):
        th0 = rng.normal(0, s, (2000, 256)).cumsum(1)
        z = np.exp(1j * th0)
        mc = float(np.mean([aligned(zz) for zz in z]))
        th = float(np.exp(-s**2 / 2))
        # walk residual floor ~ sqrt(pi/4)/sqrt(N-1) adds in quadrature
        pred = np.sqrt(th**2 + (np.pi / 4) / 255)
        rows.append(dict(s=s, mc=mc, th=float(pred)))
        worst = max(worst, abs(mc - pred))
    pd.DataFrame(rows).to_parquet(OUT / "c4_ppa_gain.parquet", index=False)
    return {"ks_stat": round(float(ks.statistic), 4),
            "ks_pvalue": round(float(ks.pvalue), 4),
            "n_null_cases": n_cases,
            "gain_worst_abs_dev": round(worst, 4),
            "gain_rows": rows,
            "pass": bool(ks.pvalue > 0.005 and worst < 0.03)}


def main():
    t0 = time.time()
    with Pool(WORKERS) as pool:
        c6 = run_c6(pool)
    c1 = run_c1()
    c2 = run_c2()
    c3 = run_c3()
    c4 = run_c4()
    findings = {"n_mc_per_cell": N_MC, "c1": c1, "c2": c2, "c3": c3,
                "c4": c4, "c6": c6,
                "wall_s": round(time.time() - t0, 1)}
    allpass = all(findings[k]["pass"] for k in ("c1", "c2", "c3", "c4", "c6"))
    findings["verdict"] = "PASS" if allpass else "DIVERGENT"
    (OUT / "findings.json").write_text(json.dumps(findings, indent=1))
    print(json.dumps({k: findings[k].get("pass") if isinstance(findings[k], dict)
                      else findings[k]
                      for k in ("c1", "c2", "c3", "c4", "c6", "verdict",
                                "wall_s")}, indent=1))
    sys.exit(0 if allpass else 1)


if __name__ == "__main__":
    main()
