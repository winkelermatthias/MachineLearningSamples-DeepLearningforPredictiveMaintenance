#!/usr/bin/env python3
"""Figures for the fleet tracking experiment."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("experiments/tracking")


def main():
    dr = pd.read_parquet(OUT / "records.parquet")
    dm = pd.read_parquet(OUT / "machines.parquet")

    # 1: example trajectories, tracked vs true, one per scenario
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
    for ax, scn in zip(axes, ("growing_fault", "stationary_fault",
                              "stable_healthy")):
        cand = dm[dm.scenario == scn]
        pick = None
        for _, r in cand.iterrows():
            g = dr[(dr.machine == r.machine)].dropna(
                subset=["e_tracked_true"]) if scn != "stable_healthy" \
                else dr[dr.machine == r.machine]
            if scn == "stable_healthy":
                ax.plot(g.t, 10 * np.log10(g.shaft1_norm + 1e-12),
                        "o-", c="#0FB5A6", label="SHAFT_1 (norm.)")
                pick = r
                break
            if len(g) >= 12 and g.e_true.max() > 0:
                ax.plot(g.t, 10 * np.log10(g.e_tracked_true + 1e-12),
                        "o-", c="#0FB5A6", label="tracked")
                ax.plot(g.t, 10 * np.log10(g.e_true + 1e-12),
                        "--", c="#333", label="true injected")
                pick = r
                break
        ttl = scn.replace("_", " ")
        if pick is not None and scn != "stable_healthy":
            ttl += f" ({pick.fault})"
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("record"); ax.legend(fontsize=7)
    axes[0].set_ylabel("pattern energy (dB)")
    fig.suptitle("Pattern-energy tracking on the order invariant "
                 "(variable speed underneath)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "trajectories.png", dpi=140, bbox_inches="tight")

    # 2: detection delay histogram (true-speed arm)
    grow = dm[dm.scenario == "growing_fault"]
    d = (grow.true_first_alarm_t - grow.onset).dropna()
    fig, ax = plt.subplots(figsize=(4.6, 3))
    ax.hist(d, bins=np.arange(-0.5, 14.5, 1), color="#0FB5A6",
            edgecolor="white")
    ax.set_xlabel("records from onset to alarm")
    ax.set_ylabel("machines")
    ax.set_title(f"Detection delay (median {d.median():.0f} records, "
                 f"alarmed {grow.true_fault_alarm.mean():.0%})",
                 fontsize=10)
    fig.savefig(OUT / "detection_delay.png", dpi=140, bbox_inches="tight")

    # 3: rise_db separation, growing vs stationary
    fig, ax = plt.subplots(figsize=(4.6, 3))
    for scn, c in (("growing_fault", "#0FB5A6"),
                   ("stationary_fault", "#999")):
        v = dm[dm.scenario == scn].true_fault_rise_db.dropna()
        ax.hist(v, bins=24, alpha=0.6, label=scn.replace("_", " "),
                color=c)
    ax.axvline(6.0, c="#d33", lw=1.2, label="6 dB alarm gate")
    ax.set_xlabel("rise over baseline (dB)"); ax.set_ylabel("machines")
    ax.legend(fontsize=7)
    ax.set_title("Growth separates from stationary presence", fontsize=10)
    fig.savefig(OUT / "rise_separation.png", dpi=140, bbox_inches="tight")
    print("figures written")


if __name__ == "__main__":
    main()
