"""SimForge-lite: enough physics to evaluate the blind cascade honestly.
Never fitted to MAFAULDA. Every run ships full ground truth.

Populations: mains-fed (fixed LF, shaft = LF/p * (1-slip)) and VFD-fed
(movable f_e, shaft = f_e/p * (1-slip), 2 f_e electrical, PWM carrier
with +/- 2 f_e sidebands). Electrical content only on motor points, at
-25 dB leakage on the coupled pump end. Confusers always on: plant hum
family at the grid 2LF, one neighbor machine, broadband noise shaped by
random resonances.
"""
import numpy as np

FS = 16_384
DUR = 4.0

BEARINGS = [  # (BPFO, BPFI, BSF, FTF) per rev, real-geometry spreads
    (3.585, 5.415, 2.322, 0.398),
    (2.998, 5.002, 1.994, 0.375),
    (4.940, 7.060, 3.220, 0.412),
    (3.052, 4.948, 2.660, 0.382),
]
FAULTS = ["healthy", "imbalance", "misalignment", "looseness", "bearing"]


def sample_machine(rng):
    pop = "vfd" if rng.random() < 0.5 else "mains"
    p = int(rng.choice([1, 2, 3]))
    slip = rng.uniform(0.005, 0.03)
    lf_grid = float(rng.choice([50.0, 60.0]))
    if pop == "mains":
        f_e = lf_grid
    else:
        f_shaft_target = rng.uniform(8.0, 60.0)
        f_e = f_shaft_target * p / (1 - slip)
    f_shaft = f_e / p * (1 - slip)
    comp = "motor" if rng.random() < 0.5 else "pump"
    fault = FAULTS[int(rng.integers(len(FAULTS)))]
    sev = float(rng.uniform(0.3, 1.0)) if fault != "healthy" else 0.0
    return {
        "population": pop, "pole_pairs": p, "slip": slip, "lf_grid": lf_grid,
        "f_e": f_e, "f_shaft": f_shaft, "component": comp,
        "equipment": f"{'Cooling Pump' if comp=='pump' else 'Main Motor'} "
                     f"P-{rng.integers(100,999)}",
        "fault": fault, "severity": sev,
        "bearing": BEARINGS[int(rng.integers(len(BEARINGS)))],
        "vanes": int(rng.choice([5, 6, 7])),
        "wander": rng.uniform(0.002, 0.015),
        "carrier": rng.uniform(2000, 6000),
    }


def synth_run(m, rng, truth=None):
    """truth: optional list; if given, every generated component family is
    appended as {"family", "freqs_hz", "amps", "drifting"} — exact
    composition ground truth for O2 judgment. Appending never touches the
    rng stream, so waveforms stay bit-identical with or without it."""
    def note(family, freqs, amps, drifting=False):
        if truth is not None:
            truth.append({"family": family,
                          "freqs_hz": [float(f) for f in freqs],
                          "amps": [float(a) for a in amps],
                          "drifting": bool(drifting)})
    n = int(DUR * FS)
    t = np.arange(n) / FS
    f0 = m["f_shaft"]
    f_inst = f0 * (1 + m["wander"] * np.sin(2 * np.pi * rng.uniform(.3, 1.2) * t
                                            + rng.uniform(0, 6.28)))
    phase = 2 * np.pi * np.cumsum(f_inst) / FS
    u = lambda: rng.uniform(0, 2 * np.pi)  # noqa: E731
    x = 0.30 * rng.standard_normal(n)

    # shaft family, always some residual imbalance
    a1 = 0.25 + (1.3 * m["severity"] if m["fault"] == "imbalance" else 0)
    x += a1 * np.cos(phase + u())
    x += 0.08 * np.cos(2 * phase + u())
    a2, a3 = 0.08, 0.0
    if m["fault"] == "misalignment":
        x += 0.9 * m["severity"] * np.cos(2 * phase + u()) \
           + 0.35 * m["severity"] * np.cos(3 * phase + u())
        a2 += 0.9 * m["severity"]
        a3 = 0.35 * m["severity"]
    note("SHAFT", [f0, 2 * f0] + ([3 * f0] if a3 else []),
         [a1, a2] + ([a3] if a3 else []))
    if m["fault"] == "looseness":
        for k in (0.5, 1.5, 2.5, 3.5):
            x += 0.45 * m["severity"] * np.cos(k * phase + u()) / (k + .5)
        note("HALF", [k * f0 for k in (0.5, 1.5, 2.5, 3.5)],
             [0.45 * m["severity"] / (k + .5) for k in (0.5, 1.5, 2.5, 3.5)])
    if m["fault"] == "bearing":
        bo = m["bearing"][int(rng.integers(3))]  # BPFO/BPFI/BSF
        bslip = rng.uniform(0.005, 0.02)
        rw = np.cumsum(rng.normal(0, 0.6 / np.sqrt(FS / f0), n))
        x += 0.8 * m["severity"] * np.cos(bo * (1 + bslip) * phase + rw)
        x += 0.35 * m["severity"] * np.cos(2 * bo * (1 + bslip) * phase + 2 * rw)
        note("BEARING", [bo * (1 + bslip) * f0, 2 * bo * (1 + bslip) * f0],
             [0.8 * m["severity"], 0.35 * m["severity"]], drifting=True)
    if m["component"] == "pump":
        x += 0.3 * np.cos(m["vanes"] * phase + u())
        note("VANE", [m["vanes"] * f0], [0.3])

    # electrical: full on motor, -25 dB on pump end
    eg = 1.0 if m["component"] == "motor" else 10 ** (-25 / 20)
    two_fe = 2 * m["f_e"]
    x += eg * 0.5 * np.cos(2 * np.pi * two_fe * t + u())
    x += eg * 0.12 * np.cos(2 * np.pi * m["f_e"] * t + u())
    note("ELEC", [two_fe, m["f_e"]], [eg * 0.5, eg * 0.12])
    fc = m["carrier"]
    for k in (-2, -1, 0, 1, 2):          # HF carrier +/- k*2LF sidebands
        amp = eg * (0.25 if k == 0 else 0.15 / abs(k))
        x += amp * np.cos(2 * np.pi * (fc + k * two_fe) * t + u())
    note("ELEC_HF", [fc + k * two_fe for k in (-2, -1, 0, 1, 2)],
         [eg * (0.25 if k == 0 else 0.15 / abs(k)) for k in (-2, -1, 0, 1, 2)])

    # plant environment: grid hum family + one neighbor machine
    hum = 2 * m["lf_grid"]
    for k in (1, 2):
        x += 0.10 / k * np.cos(2 * np.pi * k * hum * t + u())
    note("HUM", [hum, 2 * hum], [0.10, 0.05])
    fn = rng.uniform(5, 70)
    for k in (1, 2, 3):
        x += 0.12 / k * np.cos(2 * np.pi * k * fn * t + u())
    note("NEIGHBOR", [fn, 2 * fn, 3 * fn], [0.12, 0.06, 0.04])

    # one random resonance shaping the floor
    from scipy.signal import sosfilt, butter
    fr = rng.uniform(800, 6000)
    sos = butter(2, [max(fr * .9, 50) / (FS / 2), min(fr * 1.1, .48 * FS * 2 / 2) / (FS / 2)],
                 btype="band", output="sos")
    x += 3.5 * sosfilt(sos, rng.standard_normal(n))
    return x
