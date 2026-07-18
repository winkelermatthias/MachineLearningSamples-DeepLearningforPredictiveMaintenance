"""Per-run feature extraction, repeated for each phase-reference variant.

Two spectra per (run, channel, variant):
 - blocked (revs=5): coherence ratios on the k/5 grid (resolution 0.2 order,
   coarse on purpose; the ratio, not the frequency, is the discriminator)
 - fine (all revs in one FFT, resolution ~1/total_revs): precise peak orders
   feeding continued-fraction snapping
"""
import numpy as np
from . import tacho as T
from . import spectra as S

FS = 50_000
SPR = 1024          # samples per rev after angular resampling
REVS = 5            # metronome: resolves any k/5 subsynchronous component
K_GRID = 50         # coherent amps at orders 0.2 .. 10.0
MAX_PEAK_ORDER = 12.0

# MAFAULDA channel map (columns of each CSV)
CH = {"tacho": 0, "uh_ax": 1, "uh_rad": 2, "uh_tan": 3,
      "oh_ax": 4, "oh_rad": 5, "oh_tan": 6, "mic": 7}
ACCEL_CHANNELS = ["uh_rad", "uh_tan", "oh_rad", "oh_tan"]  # radial-ish first

# MAFAULDA rig bearing characteristic orders (per shaft rev).
# TODO(A1): verify against dataset documentation at download time.
BEARING_ORDERS = {"FTF": 0.375, "BSF": 1.994, "BPFO": 2.998, "BPFI": 5.002}

# rational vocabulary for CF fingerprint one-hot (q<=8, order<=10)
CF_VOCAB = sorted({p / q for q in range(1, 9) for p in range(1, 10 * q + 1)
                   if p / q <= 10.0})


def pick_ref_channel(run, f_nom):
    """Channel with strongest 1x line, used by signal-derived variants."""
    best, best_v = CH["uh_rad"], -1.0
    n = run.shape[0]
    freqs = np.fft.rfftfreq(n, 1 / FS)
    band = (freqs > f_nom * 0.9) & (freqs < f_nom * 1.1)
    for name in ACCEL_CHANNELS:
        c = CH[name]
        A = np.abs(np.fft.rfft(run[:, c] - run[:, c].mean()))
        v = A[band].max() / (np.median(A) + 1e-12)
        if v > best_v:
            best, best_v = c, v
    return best


def run_features(run, f_nom, variant):
    """run: (n,8) float array. Returns dict of scalar features + meta,
    aggregated over ACCEL_CHANNELS (max for amplitudes/ratios: a fault
    shows on its nearest sensor)."""
    ref = pick_ref_channel(run, f_nom)
    phase, meta = T.get_phase(variant, run, FS, f_nom, ref_channel=ref)

    coh_g, inc_g, ratio_g = [], [], []
    fine_peaks_all = []
    for name in ACCEL_CHANNELS:
        x = run[:, CH[name]]
        xa = S.angular_resample(x, phase, SPR)
        Z, orders = S.block_spectra(xa, SPR, REVS, window="rect")
        coh, inc, ratio = S.coherent_split(Z)
        k = np.arange(1, K_GRID + 1)          # bins at k/REVS = 0.2..10.0
        coh_g.append(coh[k]); inc_g.append(inc[k]); ratio_g.append(ratio[k])
        A, o = S.fine_order_spectrum(xa, SPR)
        fine_peaks_all += S.peak_orders(A, o, MAX_PEAK_ORDER)

    coh_g = np.max(coh_g, axis=0)
    inc_g = np.max(inc_g, axis=0)
    ratio_g = np.max(ratio_g, axis=0)

    total_revs = int(run.shape[0] / FS * meta["rate_hz"])
    tol = S.snap_tol(max(total_revs, 10))
    snapped, unsnapped = {}, []
    for o, a, prom in sorted(fine_peaks_all, key=lambda t: -t[1])[:30]:
        fr = S.cf_snap(o, tol=tol, qmax=8)
        if fr is not None:
            key = float(fr)
            snapped[key] = max(snapped.get(key, 0.0), a)
        else:
            unsnapped.append((o, a))

    feats = {"variant": variant, "rate_hz": meta["rate_hz"],
             "tacho_quality": meta.get("quality", np.nan),
             "ref_channel": ref,
             "n_blocks": total_revs // REVS,
             "unsnapped_energy": float(sum(a for _, a in unsnapped)),
             "n_unsnapped": len(unsnapped)}
    for i in range(K_GRID):
        o = (i + 1) / REVS
        feats[f"coh_{o:.1f}"] = float(coh_g[i])
        feats[f"inc_{o:.1f}"] = float(inc_g[i])
        # self-referential bias: order 1.0 ratio is ~1 by construction for
        # signal-derived variants; keep column, mask in model config
        feats[f"ratio_{o:.1f}"] = float(ratio_g[i])
    for v in CF_VOCAB:
        feats[f"cf_{v:.4f}"] = float(snapped.get(v, 0.0))
    for i, (o, a) in enumerate(sorted(unsnapped, key=lambda t: -t[1])[:5]):
        feats[f"uns_o{i}"] = o
        feats[f"uns_a{i}"] = a
    for i in range(len(unsnapped), 5):
        feats[f"uns_o{i}"] = 0.0
        feats[f"uns_a{i}"] = 0.0
    # ratio at bearing-order bins (nearest k/5 bin)
    for nm, bo in BEARING_ORDERS.items():
        kb = int(round(bo * REVS))
        if 1 <= kb <= K_GRID:
            feats[f"bear_ratio_{nm}"] = float(ratio_g[kb - 1])
            feats[f"bear_inc_{nm}"] = float(inc_g[kb - 1])
    return feats


SELF_REF_MASK = {  # ratio columns to drop per variant (bias by construction)
    "tacho": [],
    "onex": ["ratio_1.0"],
    "comb": [],           # comb phase is fitted to harmonics; treat 1x..n_harm
                          # ratios with suspicion in adversary checks
    "nominal": [],
}
