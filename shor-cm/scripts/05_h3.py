#!/usr/bin/env python3
"""H3: coherent 5-rev averaging obeys ~sqrt(N) gain on subsync locked tones.

Design: take NORMAL-class runs (clean tacho decile), inject a 0.4x tone at
SNR grid, detect in coherent vs power-averaged spectrum, sweep N blocks.
Negative control: unlocked injection (coherent must NOT gain).
Repeated for phase variants tacho / onex / comb / nominal: this answers
"how much of the gain survives without a tacho" directly.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import tacho as T                       # noqa: E402
from shorcm import spectra as S                     # noqa: E402
from shorcm.features import FS, SPR, REVS, CH       # noqa: E402

OUTD = Path("experiments/h3"); OUTD.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260709)
SNR_DB = np.arange(-35, -4.9, 2.5)
N_BLOCKS = [4, 16, 60]           # 5 s at ~40 Hz gives ~40 blocks; 60 needs
                                 # >= 25 Hz runs of full length; clipped below
N_SEEDS = 20
ORDER = 0.4
DETECT_SIGMA = 3.0


def detect(spec, orders, order, guard_bins=2):
    b = int(np.argmin(np.abs(orders - order)))
    excl = np.zeros(len(spec), bool)
    excl[max(b - guard_bins, 0): b + guard_bins + 1] = True
    excl[0] = True
    floor = spec[~excl][:200]
    return spec[b] > np.median(floor) + DETECT_SIGMA * floor.std()


def main():
    cat = pd.read_parquet("data/catalog.parquet")
    norm = cat[(cat.cls == "normal")].copy()
    norm = norm[norm.tacho_quality <= norm.tacho_quality.quantile(0.5)]
    norm = norm.sort_values("rate_hz", ascending=False).head(12)
    rows = []
    for _, r in norm.iterrows():
        arr = pd.read_parquet(r["path"]).values.astype(float)
        true_phase, _ = T.phase_from_tacho(arr[:, 0], FS)
        x0 = arr[:, CH["uh_rad"]]
        for variant in ("tacho", "onex", "comb", "nominal"):
            for locked in (True, False):
                for snr in SNR_DB:
                    amp = S.amp_for_snr(x0, snr)
                    for seed in range(N_SEEDS):
                        phi0 = RNG.uniform(0, 2 * np.pi)
                        if locked:  # injection ALWAYS uses true tacho phase
                            x = S.inject_locked(x0, true_phase, ORDER, amp, phi0)
                        else:
                            f = ORDER * r["rate_hz"] * 1.015
                            x = S.inject_unlocked(x0, FS, f, amp, phi0)
                        run = arr.copy(); run[:, CH["uh_rad"]] = x
                        try:
                            ph, _ = T.get_phase(variant, run, FS, r["f_nom"],
                                                ref_channel=CH["uh_rad"])
                            xa = S.angular_resample(x, ph, SPR)
                        except Exception:
                            continue
                        for N in N_BLOCKS:
                            L = N * REVS * SPR
                            if len(xa) < L:
                                continue
                            Z, orders = S.block_spectra(xa[:L], SPR, REVS)
                            coh, inc, _ = S.coherent_split(Z)
                            rows.append(dict(
                                run_id=r["run_id"], variant=variant,
                                locked=locked, snr_db=float(snr), N=N,
                                seed=seed,
                                det_coh=bool(detect(coh, orders, ORDER)),
                                det_inc=bool(detect(inc, orders, ORDER))))
    df = pd.DataFrame(rows)
    df.to_parquet(OUTD / "results.parquet", index=False)

    # detection threshold = lowest SNR with P(detect) >= 0.5, per cell
    def thr(g):
        p = g.groupby("snr_db")[["det_coh", "det_inc"]].mean()
        out = {}
        for c in ("det_coh", "det_inc"):
            ok = p.index[p[c] >= 0.5]
            out[c] = float(ok.min()) if len(ok) else np.nan
        return pd.Series(out)
    th = df[df.locked].groupby(["variant", "N"]).apply(thr).reset_index()
    # gain exponent from threshold shift: thr(N) ~ thr0 - 10*a*log10(N)
    fits = []
    for v, g in th.groupby("variant"):
        g = g.dropna(subset=["det_coh"])
        if len(g) >= 2:
            a = np.polyfit(10 * np.log10(g.N), -g.det_coh, 1)[0]
            fits.append(dict(variant=v, gain_exponent=float(a)))
    fits = pd.DataFrame(fits)
    fits.to_parquet(OUTD / "gain_exponents.parquet", index=False)
    neg = df[(~df.locked)].groupby("variant")[["det_coh", "det_inc"]].mean()
    (OUTD / "summary.md").write_text(
        "# H3 summary\n\n## detection threshold (dB SNR) per variant x N\n"
        + th.to_markdown(index=False)
        + "\n\n## gain exponent (theory 0.5)\n" + fits.to_markdown(index=False)
        + "\n\n## negative control, unlocked injection, mean detect rate\n"
        + neg.to_markdown()
        + "\n\nPass: tacho exponent in [0.35, 0.65]; unlocked det_coh stays "
          "near det_inc noise level (coherent must not 'detect' slip tones "
          "better). Interesting number: exponent for onex/comb vs nominal.\n")
    print(fits)


if __name__ == "__main__":
    main()
