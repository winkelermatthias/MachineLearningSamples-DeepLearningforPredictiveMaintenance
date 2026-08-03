"""Physics-based synthetic vibration corpus: spectral patterns + ground-truth labels.

Two channels per sample, both 25.6 kHz / 65 536 points (df = 0.39 Hz):
- velocity (mm/s): 1x/harmonics, 2x misalignment, looseness combs, gear mesh
- acceleration (g): HF bearing impact bursts -> envelope spectrum (2-8 kHz demod)

Labels are MEASURED, not asserted: the ISO 20816-3 zone comes from the actual
velocity RMS of the synthesized signal against the machine's zone boundaries,
and the bearing-condition grade from the actual largest envelope peak — so
labels are self-consistent with what the model sees, by construction.
"""
import json
import math
import os

import numpy as np
from scipy import signal as sps

# ---- bearing catalogue (n rolling elements, ball dia d [mm], pitch dia D [mm], contact angle [deg])
BEARINGS = {
    "6205":   dict(n=9,  d=7.94,  D=39.04, phi=0.0),
    "6309":   dict(n=8,  d=17.46, D=72.50, phi=0.0),
    "6311":   dict(n=8,  d=20.64, D=85.50, phi=0.0),
    "7310B":  dict(n=12, d=19.05, D=81.50, phi=40.0),
    "NU216":  dict(n=16, d=15.00, D=120.0, phi=0.0),
    "22216E": dict(n=19, d=15.90, D=110.0, phi=9.0),
}


def _ratios(b):
    r = b["d"] / b["D"] * math.cos(math.radians(b["phi"]))
    return {
        "BPFO": b["n"] / 2 * (1 - r),
        "BPFI": b["n"] / 2 * (1 + r),
        "BSF":  b["D"] / (2 * b["d"]) * (1 - r**2),
        "FTF":  0.5 * (1 - r),
    }


BEARING_RATIOS = {k: _ratios(v) for k, v in BEARINGS.items()}

# ---- ISO 20816-3 zone boundaries (velocity mm/s RMS): (A/B, B/C, C/D)
ISO_BOUNDS = {
    (1, "rigid"): (2.3, 4.5, 7.1), (1, "flexible"): (3.5, 7.1, 11.0),   # group 1: > 300 kW
    (2, "rigid"): (1.4, 2.8, 4.5), (2, "flexible"): (2.3, 4.5, 7.1),    # group 2: 15–300 kW
}


def iso_bounds(power_kw, mount):
    return ISO_BOUNDS[(1 if power_kw > 300 else 2, mount)]


def iso_zone(v_rms, bounds):
    a, b, c = bounds
    return "A" if v_rms <= a else "B" if v_rms <= b else "C" if v_rms <= c else "D"


# ---- machine fleet templates
MACHINES = [
    dict(kind="end-suction centrifugal pump, direct coupled", tag="P", power=(15, 250),
         speeds=[(2940, 2985), (1465, 1492)], mount="rigid", bearings=["6309", "6311"], gear=None),
    dict(kind="centrifugal fan, belt driven", tag="FN", power=(11, 90),
         speeds=[(850, 1750)], mount="flexible", bearings=["22216E", "6309"], gear=None),
    dict(kind="oil-flooded screw compressor, direct coupled", tag="K", power=(75, 315),
         speeds=[(2950, 2978)], mount="rigid", bearings=["7310B", "NU216"], gear=None),
    dict(kind="cooling tower fan, gearbox driven (measurement at gearbox input)", tag="CT", power=(30, 132),
         speeds=[(1470, 1488)], mount="flexible", bearings=["22216E"], gear=dict(z1=17, z2=79)),
    dict(kind="process pump on VFD", tag="P", power=(22, 110),
         speeds=[(900, 2990)], mount="rigid", bearings=["6205", "6309"], gear=None),
    dict(kind="mill drive through gearbox (measurement at input pinion)", tag="M", power=(250, 900),
         speeds=[(984, 996)], mount="rigid", bearings=["NU216", "22216E"], gear=dict(z1=23, z2=104)),
]

FAULTS = ["healthy", "imbalance", "misalignment", "looseness",
          "bearing_outer_race", "bearing_inner_race", "bearing_rolling_element", "gear_mesh_wear"]

# bearing-condition grade from the largest envelope-spectrum peak (g):
COND_BOUNDS = (0.05, 0.30, 1.00)   # none < 0.05 <= early < 0.30 <= moderate < 1.00 <= severe


def cond_grade(env_pk):
    a, b, c = COND_BOUNDS
    return "none" if env_pk < a else "early" if env_pk < b else "moderate" if env_pk < c else "severe"


def sample_machine(rng):
    tpl = MACHINES[int(rng.integers(len(MACHINES)))]
    lo, hi = tpl["speeds"][int(rng.integers(len(tpl["speeds"])))]
    power = float(rng.uniform(*tpl["power"]))
    return dict(
        kind=tpl["kind"],
        asset=f"{tpl['tag']}-{int(rng.integers(100, 999))}",
        power_kW=power,
        mount=tpl["mount"],
        speed_lo=lo, speed_hi=hi,
        rpm=float(rng.uniform(lo, hi)),
        bearing=tpl["bearings"][int(rng.integers(len(tpl["bearings"])))],
        gear=tpl["gear"],
        bounds=iso_bounds(power, tpl["mount"]),
    )


# ---- waveform synthesis --------------------------------------------------
FS = 25_600
N = 1 << 16                      # 65 536 samples = 2.56 s  ->  df = 0.39 Hz
T = np.arange(N) / FS


def _tone(rng, f, amp_rms):
    return amp_rms * np.sqrt(2) * np.sin(2 * np.pi * f * T + rng.uniform(0, 2 * np.pi))


def _impact_train(rng, f_imp, amp, fc, zeta=0.04, mod_f=None, mod_depth=0.0):
    """Repetitive impacts at f_imp Hz exciting a structural resonance at fc Hz."""
    n_imp = int(f_imp * N / FS) + 2
    t_imp = (np.arange(n_imp) + rng.uniform(0.0, 1.0)) / f_imp
    t_imp = t_imp + rng.normal(0.0, 0.01 / f_imp, n_imp)          # ~1 % slip jitter
    t_imp = t_imp[(t_imp >= 0) & (t_imp < N / FS)]
    w = rng.uniform(0.7, 1.3, t_imp.size)
    if mod_f:                                                     # load-zone modulation
        w = w * (1.0 + mod_depth * np.sin(2 * np.pi * mod_f * t_imp))
    imp = np.zeros(N)
    np.add.at(imp, np.round(t_imp * FS).astype(int) % N, amp * np.abs(w))
    k = np.arange(int(FS * 0.004))                                # 4 ms ring-down kernel
    kern = np.exp(-2 * np.pi * fc * zeta * k / FS) * np.sin(2 * np.pi * fc * k / FS)
    return np.convolve(imp, kern)[:N]


_SOS_V = sps.butter(4, [10, 1000], btype="bandpass", fs=FS, output="sos")
_SOS_A = sps.butter(4, [2000, 8000], btype="bandpass", fs=FS, output="sos")
_WIN = np.hanning(N)
FAX = np.fft.rfftfreq(N, 1 / FS)


def amp_spectrum(x):
    return np.abs(np.fft.rfft(x * _WIN)) * 2 / _WIN.sum()         # peak-amplitude spectrum


def _peaks(f, X, fmin, fmax, top):
    m = (f >= fmin) & (f <= fmax)
    fi, Xi = f[m], X[m]
    pk, _ = sps.find_peaks(Xi, prominence=Xi.max() * 0.02)
    if pk.size == 0:
        return []
    sel = np.sort(pk[np.argsort(Xi[pk])[::-1][:top]])
    return [(float(fi[i]), float(Xi[i])) for i in sel]


def synth_sample(rng, blind_fraction, keep_wave=False):
    m = sample_machine(rng)
    fr = m["rpm"] / 60.0
    pool = [f for f in FAULTS if f != "gear_mesh_wear" or m["gear"]]
    fault = pool[int(rng.integers(len(pool)))]
    sev = 0.0 if fault == "healthy" else float(rng.uniform(0.4, 4.2))
    R = BEARING_RATIOS[m["bearing"]]
    kz = m["bounds"][2] / 4.5     # scale severity to THIS machine's zone widths

    # baseline residual imbalance/misalignment + broadband floors
    v = _tone(rng, fr, rng.uniform(0.25, 0.55)) + _tone(rng, 2 * fr, rng.uniform(0.06, 0.18))
    v = v + rng.normal(0, 0.05, N)                                # mm/s
    a = rng.normal(0, 0.02, N)                                    # g
    fc = rng.uniform(2500, 5500)                                  # bearing housing resonance

    if fault == "imbalance":
        v = v + _tone(rng, fr, 1.4 * sev * kz)
    elif fault == "misalignment":
        v = v + _tone(rng, 2 * fr, 1.1 * sev * kz) + _tone(rng, fr, 0.4 * sev * kz) + _tone(rng, 3 * fr, 0.35 * sev * kz)
    elif fault == "looseness":
        for k in range(1, 9):
            v = v + _tone(rng, k * fr, 0.6 * sev * kz / k**0.7)
        v = v + _tone(rng, 0.5 * fr, 0.2 * sev * kz)
    elif fault == "bearing_outer_race":
        f0 = R["BPFO"] * fr
        a = a + _impact_train(rng, f0, 2.0 * sev, fc)
        v = v + _tone(rng, fr, 0.2 * sev * kz)        # late-stage wear lifts 1x too
        for k in (1, 2, 3):
            v = v + _tone(rng, k * f0, 0.15 * sev * kz)
    elif fault == "bearing_inner_race":
        f0 = R["BPFI"] * fr
        a = a + _impact_train(rng, f0, 2.0 * sev, fc, mod_f=fr, mod_depth=0.8)
        v = v + _tone(rng, fr, 0.2 * sev * kz)
        for k in (1, 2):
            v = v + _tone(rng, k * f0, 0.12 * sev * kz) + _tone(rng, k * f0 + fr, 0.06 * sev * kz) + _tone(rng, k * f0 - fr, 0.06 * sev * kz)
    elif fault == "bearing_rolling_element":
        f0 = 2 * R["BSF"] * fr
        a = a + _impact_train(rng, f0, 1.8 * sev, fc, mod_f=R["FTF"] * fr, mod_depth=0.9)
        v = v + _tone(rng, fr, 0.2 * sev * kz) + _tone(rng, f0, 0.12 * sev * kz)
    elif fault == "gear_mesh_wear":
        gmf = m["gear"]["z1"] * fr
        v = v + _tone(rng, gmf, 0.7 * sev * kz) + _tone(rng, 2 * gmf, 0.2 * sev * kz)
        v = v + _tone(rng, gmf - fr, 0.25 * sev * kz) + _tone(rng, gmf + fr, 0.25 * sev * kz)

    # measured features -> labels are self-consistent by construction
    v_rms = float(np.std(sps.sosfiltfilt(_SOS_V, v)))
    a_hf = sps.sosfiltfilt(_SOS_A, a)
    crest = float(np.max(np.abs(a_hf)) / (np.std(a_hf) + 1e-12))
    env = np.abs(sps.hilbert(a_hf))
    env = env - env.mean()
    peaks_v = _peaks(FAX, amp_spectrum(v), 2.0, 1200.0, top=15)
    peaks_e = _peaks(FAX, amp_spectrum(env), 2.0, 600.0, top=10)
    env_pk = max((amp for _, amp in peaks_e), default=0.0)

    s = dict(
        machine=m, fault=fault, sev=sev, rpm=m["rpm"],
        v_rms=v_rms, crest=crest, env_pk=env_pk,
        zone=iso_zone(v_rms, m["bounds"]), cond=cond_grade(env_pk),
        peaks_v=peaks_v, peaks_e=peaks_e,
        blind=bool(rng.uniform() < blind_fraction),
    )
    if keep_wave:
        s["wave"] = dict(v=v, env=env)
    return s


def render_report(s):
    """The measurement report a technician would hand to an analyst.
    Contains everything needed to solve the case — and never the answer."""
    m = s["machine"]
    L = []
    L.append("=== VIBRATION MEASUREMENT REPORT ===")
    L.append(f"Asset {m['asset']}: {m['kind']}")
    L.append(f"Rated power: {m['power_kW']:.0f} kW | mounting: {m['mount']} | line frequency: 50 Hz")
    L.append(f"Nameplate speed range: {m['speed_lo']:.0f}-{m['speed_hi']:.0f} RPM "
             "(actual running speed NOT recorded — no tacho fitted)")
    if m["gear"]:
        L.append(f"Gearbox: pinion {m['gear']['z1']} teeth / wheel {m['gear']['z2']} teeth")
    if s["blind"]:
        L.append("Bearing designation: UNKNOWN. Candidate catalogue "
                 "(fault-frequency ratios in orders of shaft speed):")
        for name, r in BEARING_RATIOS.items():
            L.append(f"    {name:8s} BPFO={r['BPFO']:.3f}  BPFI={r['BPFI']:.3f}  "
                     f"BSF={r['BSF']:.3f}  FTF={r['FTF']:.3f}")
    else:
        r = BEARING_RATIOS[m["bearing"]]
        L.append(f"Bearing: {m['bearing']} (BPFO={r['BPFO']:.3f}, BPFI={r['BPFI']:.3f}, "
                 f"BSF={r['BSF']:.3f}, FTF={r['FTF']:.3f} x shaft speed)")
    a, b, c = m["bounds"]
    L.append(f"ISO 20816-3 zone boundaries for this machine (velocity mm/s RMS): "
             f"A <= {a} < B <= {b} < C <= {c} < D")
    L.append(f"Overall velocity 10-1000 Hz: {s['v_rms']:.2f} mm/s RMS | "
             f"HF acceleration crest factor (2-8 kHz): {s['crest']:.1f}")
    L.append("--- Velocity spectrum peaks (Hz | mm/s peak) ---")
    for f, amp in s["peaks_v"]:
        L.append(f"    {f:8.2f}  |  {amp:6.3f}")
    L.append("--- Envelope spectrum peaks, 2-8 kHz demod band (Hz | g peak) ---")
    if s["peaks_e"]:
        for f, amp in s["peaks_e"]:
            L.append(f"    {f:8.2f}  |  {amp:6.3f}")
    else:
        L.append("    (no significant peaks)")
    return "\n".join(L)


def build_corpus(cfg, rng):
    """Generate (or reload) the corpus. Cached to disk with stable sample ids so
    previously cached teacher answers stay valid. Delete the file to force
    regeneration (e.g. after changing n_samples or the simulator)."""
    from tqdm.auto import tqdm

    path = os.path.join(cfg.out_dir, "synthetic_corpus.jsonl")
    if os.path.exists(path):
        with open(path) as fh:
            dataset = [json.loads(l) for l in fh if l.strip()]
        if len(dataset) == cfg.n_samples:
            print(f"reloaded {len(dataset):,} cached samples from {path}")
            return dataset
        print(f"cache has {len(dataset):,} samples but n_samples={cfg.n_samples:,} — regenerating")

    dataset = []
    for i in tqdm(range(cfg.n_samples), desc="synthesizing"):
        s = synth_sample(rng, cfg.blind_fraction)
        s["id"] = i
        s["report"] = render_report(s)
        dataset.append(s)
    with open(path, "w") as fh:
        for s in dataset:
            fh.write(json.dumps(s) + "\n")
    return dataset


def demo_case(fault="bearing_inner_race", seed=7):
    """One example with waveforms kept, for plotting."""
    rng = np.random.default_rng(seed)
    s = None
    while s is None or s["fault"] != fault:
        s = synth_sample(rng, blind_fraction=0.0, keep_wave=True)
    return s
