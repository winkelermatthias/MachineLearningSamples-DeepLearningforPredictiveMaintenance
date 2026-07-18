#!/usr/bin/env python3
"""Build the final self-contained HTML report + decision memo.
Refuses to run unless adversary verdict passed (G4).
Every number is read from results files; nothing typed by hand."""
import base64, io, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTD = Path("report"); OUTD.mkdir(exist_ok=True)
TEAL = "#0FB5A6"


def fig64(fig):
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def main():
    if not json.load(open("adversary/verdict.json"))["pass"]:
        sys.exit("G4 not passed: report blocked")
    h1 = pd.read_parquet("experiments/h1/results.parquet")
    h2 = pd.read_parquet("experiments/h2/results.parquet")
    h2m = pd.read_parquet("experiments/h2/mcnemar.parquet")
    h3g = pd.read_parquet("experiments/h3/gain_exponents.parquet")
    h3 = pd.read_parquet("experiments/h3/results.parquet")
    adv = Path("adversary/findings.md").read_text()

    # Fig 1: H1 AUC by variant (RATIO featureset), gkf vs extreme
    g = h1[(h1.fset == "RATIO") & h1.split.str.startswith("gkf")] \
        .groupby("variant").auc.mean()
    e = h1[(h1.fset == "RATIO") & (h1.split == "extreme_speed")] \
        .set_index("variant").auc
    order = ["tacho", "onex", "comb", "nominal"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(len(order))
    ax.bar(x - 0.18, [g.get(v, np.nan) for v in order], 0.36,
           color=TEAL, label="grouped 5-fold")
    ax.bar(x + 0.18, [e.get(v, np.nan) for v in order], 0.36,
           color="#666", label="extreme speed")
    ax.axhline(0.90, ls="--", c="k", lw=0.8)
    ax.set_xticks(x, order); ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUC, coherence-ratio features"); ax.legend()
    f1 = fig64(fig)

    # Fig 2: H3 detection curves, tacho variant, locked
    fig, ax = plt.subplots(figsize=(6, 3.2))
    d = h3[(h3.variant == "tacho") & h3.locked]
    for N, gN in d.groupby("N"):
        p = gN.groupby("snr_db").det_coh.mean()
        ax.plot(p.index, p.values, marker="o", label=f"coherent N={N}")
    p = d[d.N == d.N.max()].groupby("snr_db").det_inc.mean()
    ax.plot(p.index, p.values, marker="s", c="#666", label="power avg")
    ax.set_xlabel("injected SNR (dB)"); ax.set_ylabel("P(detect 0.4x)")
    ax.legend(); f2 = fig64(fig)

    acc = h2.pivot_table(index="variant", columns="fset", values="acc")
    memo = [
        "# Decision memo, SHOR-CM",
        f"- H1 (tacho, RATIO, grouped CV): AUC {g.get('tacho', float('nan')):.3f} "
        f"(pass > 0.90); no-tacho best: "
        f"{max(g.get('onex', 0), g.get('comb', 0)):.3f}",
        f"- H1 speed invariance (extreme hold-out, tacho): "
        f"{e.get('tacho', float('nan')):.3f}",
        f"- H2 (tacho): acc A/B/C = "
        f"{acc.loc['tacho'].round(3).to_dict() if 'tacho' in acc.index else 'n/a'}, "
        f"McNemar p_holm = "
        f"{float(h2m.loc[h2m.variant == 'tacho', 'p_holm'].iloc[0]):.4f}",
        f"- H3 gain exponents: " +
        ", ".join(f"{r.variant}={r.gain_exponent:.2f}"
                  for r in h3g.itertuples()) + " (theory 0.5)",
        "- Adversary: PASS",
        "",
        "Recommendation: fill after reading, one line: SHIP / ITERATE / KILL",
    ]
    (OUTD / "decision_memo.md").write_text("\n".join(memo))

    html = f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>SHOR-CM report</title><style>
body{{font-family:Onest,Geist,system-ui,sans-serif;max-width:760px;margin:2rem auto;
padding:0 1rem;color:#1a1a1a;background:#fff;line-height:1.55}}
h1,h2{{color:{TEAL}}} table{{border-collapse:collapse;font-size:.9rem}}
td,th{{border:1px solid #ddd;padding:4px 8px}} code{{font-size:.8em;color:#555}}
img{{max-width:100%}}</style></head><body>
<h1>Shor-inspired coherent order analysis on MAFAULDA</h1>
<p>Period finding, phase-coherent interference, and continued-fraction
rational snapping applied to rotating machinery. Four phase-reference
variants: real tacho, 1x Hilbert, harmonic-comb, nominal filename speed.</p>
<h2>H1: coherence ratio</h2><img src="data:image/png;base64,{f1}">
<pre>{h1[h1.fset == 'RATIO'].groupby(['variant', 'split']).auc.mean().to_string()}</pre>
<code>experiments/h1/results.parquet</code>
<h2>H2: CF fingerprints</h2><pre>{acc.to_string()}</pre>
<pre>{h2m.to_string(index=False)}</pre>
<code>experiments/h2/results.parquet</code>
<h2>H3: gain law</h2><img src="data:image/png;base64,{f2}">
<pre>{h3g.to_string(index=False)}</pre>
<code>experiments/h3/results.parquet</code>
<h2>Adversary</h2><pre>{adv}</pre>
</body></html>"""
    (OUTD / "shor_cm_report.html").write_text(html)
    print("report written:", OUTD / "shor_cm_report.html")


if __name__ == "__main__":
    main()
