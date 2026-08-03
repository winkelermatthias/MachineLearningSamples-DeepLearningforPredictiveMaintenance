"""Plots: corpus distributions, example spectra, base-vs-distilled comparison."""
from collections import Counter

import numpy as np

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed categorical order


def setup_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
        "xtick.color": "#898781", "ytick.color": "#898781", "text.color": "#0b0b0b",
        "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "sans-serif", "lines.linewidth": 2.0, "figure.dpi": 110,
    })
    return plt


def plot_distributions(dataset):
    plt = setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for ax, key, title in ((axes[0], "fault", "Fault distribution"),
                           (axes[1], "zone", "ISO 20816-3 zone distribution")):
        cnt = Counter(s[key] for s in dataset)
        ks = sorted(cnt, key=cnt.get, reverse=True)
        ax.barh(range(len(ks)), [cnt[k] for k in ks], color=PAL[0], height=0.62)
        ax.set_yticks(range(len(ks)), ks)
        ax.invert_yaxis()
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    plt.show()


def plot_example(demo):
    from .synth import BEARING_RATIOS, FAX, amp_spectrum

    plt = setup_style()
    Xv = amp_spectrum(demo["wave"]["v"])
    Xe = amp_spectrum(demo["wave"]["env"])
    fr = demo["rpm"] / 60
    bpfi = BEARING_RATIOS[demo["machine"]["bearing"]]["BPFI"] * fr

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    axes[0].plot(FAX[FAX <= 1200], Xv[FAX <= 1200], color=PAL[0])
    axes[0].set_title(f"Velocity spectrum — {demo['fault']} @ {demo['rpm']:.0f} RPM", loc="left")
    axes[0].set_xlabel("Hz"); axes[0].set_ylabel("mm/s peak")
    axes[1].plot(FAX[FAX <= 400], Xe[FAX <= 400], color=PAL[1])
    axes[1].axvline(bpfi, color="#898781", linestyle=":", linewidth=1.2)
    axes[1].annotate(f"BPFI = {bpfi:.1f} Hz", (bpfi, Xe[FAX <= 400].max() * 0.92),
                     textcoords="offset points", xytext=(6, 0), color="#52514e", fontsize=9)
    axes[1].set_title("Envelope spectrum (2-8 kHz demod)", loc="left")
    axes[1].set_xlabel("Hz"); axes[1].set_ylabel("g peak")
    fig.tight_layout()
    plt.show()
    print("Truth:", {k: demo[k] for k in ("fault", "zone", "cond")}, f"| 1x = {fr:.2f} Hz")


def plot_comparison(base_scores, tuned_scores, base_label="Base student"):
    import pandas as pd

    plt = setup_style()
    labels = list(base_scores)
    bx = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(bx - w / 2, [base_scores[k] for k in labels], w, label=base_label, color=PAL[0])
    ax.bar(bx + w / 2, [tuned_scores[k] for k in labels], w, label="Distilled student", color=PAL[1])
    ax.set_xticks(bx, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy on held-out ground truth")
    ax.set_title("What did the distillation buy?", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()
    print(pd.DataFrame({"base": base_scores, "distilled": tuned_scores}).T.round(3))
