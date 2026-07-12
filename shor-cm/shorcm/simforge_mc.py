"""SimForge v2.1: multi-channel rendering of the v2 population.

Three channels — radial drive-end (DE), AXIAL, radial non-drive /
driven-end (NDE) — sharing each component's PHASE (one physical source)
with per-channel gains from the classic vibration-analysis casebook,
and INDEPENDENT noise/resonance paths per channel:

 - angular misalignment is AXIAL-dominant (axial 1x/2x ~ radial), while
   parallel misalignment stays radial: the subtype evidence a single
   radial channel cannot see;
 - driven-shaft families and output-shaft bearings are stronger at the
   NDE, input families weaker; electrical leaks a further -10 dB;
 - a component present in >= 2 channels is real — cross-channel peak
   confirmation kills single-channel noise peaks.

Machine sampling and kinematics are v2's (`sample_machine`,
`kinematic_sheet`) — this module only renders. Never fitted to MAFAULDA.
"""
import numpy as np
from scipy.signal import sosfilt, butter

from . import simforge_v2 as V2

FS = V2.FS
DUR = V2.DUR
CH = ("radial_de", "axial", "radial_nde")

# axial gain by (family, misalignment subtype where relevant)
AX_SHAFT1 = {"parallel": 0.2, "angular": 0.9, "coupling": 0.5, "": 0.15}
AX_SHAFT2 = {"parallel": 0.3, "angular": 0.85, "coupling": 0.6, "": 0.2}


def synth_run_mc(m, rng, truth=None, omega1x=1.0):
    """(n, 3) float array. Components share phases; channels get gains.
    Truth (optional list) matches simforge_v2 conventions, radial-DE
    amplitudes."""
    def note(family, freqs, amps, drifting=False):
        if truth is not None:
            truth.append({"family": family,
                          "freqs_hz": [float(f) for f in freqs],
                          "amps": [float(a) for a in amps],
                          "drifting": bool(drifting)})
    n = int(DUR * FS)
    t = np.arange(n) / FS
    f0 = m["f_shaft"]
    s = m["severity"]
    f_inst = f0 * (1 + m["wander"] * np.sin(
        2 * np.pi * rng.uniform(.3, 1.2) * t + rng.uniform(0, 6.28)))
    ph = 2 * np.pi * np.cumsum(f_inst) / FS
    f2 = m.get("f2", f0)
    ph2 = ph * (f2 / f0)
    if m["archetype"] == "fan_belt":
        ph2 = ph2 + np.cumsum(rng.normal(0, 0.4 / np.sqrt(FS / f0), n))
    f_out = m.get("f3", f2)
    ph_out = ph * (f_out / f0)
    u = lambda: rng.uniform(0, 2 * np.pi)         # noqa: E731

    comps = []                    # (waveform, gains(de, ax, nde))

    def add(w, g_de, g_ax, g_nde):
        comps.append((w, (g_de, g_ax, g_nde)))

    st = m["subtype"] if m["fault"] == "misalignment" else ""
    a1 = (m["resid_1x"] + (1.3 * s if m["fault"] == "imbalance" else 0)) \
        * omega1x
    a2 = rng.uniform(0.03, 0.10)
    a3 = 0.0
    if m["fault"] == "misalignment":
        if st == "parallel":
            a2 += 0.9 * s; a3 += 0.35 * s
        elif st == "angular":
            a1 += 0.20 * s; a2 += 0.5 * s; a3 += 0.15 * s
        else:
            a2 += 0.7 * s; a3 += 0.10 * s
    ax1, ax2 = AX_SHAFT1.get(st, 0.15), AX_SHAFT2.get(st, 0.2)
    add(a1 * np.cos(ph + u()), 1.0, ax1, 0.5)
    add(a2 * np.cos(2 * ph + u()), 1.0, ax2, 0.5)
    sh_f, sh_a = [f0, 2 * f0], [a1, a2]
    if a3:
        add(a3 * np.cos(3 * ph + u()), 1.0, 0.5 * ax2, 0.5)
        sh_f.append(3 * f0); sh_a.append(a3)
    if st == "coupling":
        add(0.25 * s * np.cos(4 * ph + u()), 1.0, 0.5, 0.5)
        sh_f.append(4 * f0); sh_a.append(0.25 * s)
    note("SHAFT", sh_f, sh_a)
    if m["fault"] == "looseness":
        ks = (0.5, 1.5, 2.5, 3.5)
        for k in ks:
            add(0.45 * s * np.cos(k * ph + u()) / (k + .5), 1.0, 0.2, 0.5)
        note("HALF", [k * f0 for k in ks], [0.45 * s / (k + .5) for k in ks])
    if f2 != f0:
        b1 = rng.uniform(0.10, 0.30)
        add(b1 * np.cos(ph2 + u()), 1.0, 0.2, 1.4)
        add(0.4 * b1 * np.cos(2 * ph2 + u()), 1.0, 0.2, 1.4)
        note("SHAFT2", [f2, 2 * f2], [b1, 0.4 * b1],
             drifting=(m["archetype"] == "fan_belt"))
        if f_out != f2:
            b2 = rng.uniform(0.10, 0.30)
            add(b2 * np.cos(ph_out + u()), 1.0, 0.2, 1.4)
            note("SHAFT2", [f_out], [b2])
    if "vanes" in m:
        av = rng.uniform(0.15, 0.45)
        add(av * np.cos(m["vanes"] * ph_out + u()), 1.0, 0.35, 1.3)
        note("PASSAGE", [m["vanes"] * f_out], [av])
    if "blades" in m:
        ab = rng.uniform(0.15, 0.45)
        add(ab * np.cos(m["blades"] * ph_out + u()), 1.0, 0.6, 1.3)
        note("PASSAGE", [m["blades"] * f_out], [ab])
    if "lobes" in m:
        al = rng.uniform(0.3, 0.6)
        add(al * np.cos(2 * m["lobes"] * ph + u()), 1.0, 0.3, 1.0)
        add(0.4 * al * np.cos(4 * m["lobes"] * ph + u()), 1.0, 0.3, 1.0)
        note("PASSAGE", [2 * m["lobes"] * f0, 4 * m["lobes"] * f0],
             [al, 0.4 * al])
    if m["archetype"] in ("pump_gearbox1", "fan_gearbox2",
                          "pump_planetary"):
        gax = rng.uniform(0.2, 0.7)       # helix angle -> axial mesh
        base = rng.uniform(0.2, 0.5)
        gmf_gain = 0.3 * s if m["fault"] == "gear" else 0.0
        if m["archetype"] == "pump_planetary":
            fc = m["fc"]
            sb_sp = {"sun": f0 - fc, "planet": 2 * fc * m["Zr"] / m["Zp"],
                     "ring": m["Np"] * fc}.get(m["subtype"], fc) \
                if m["fault"] == "gear" else fc
        else:
            f_flt = f0 if m.get("fault_gear") == "in" else f2
            sb_sp = f_flt if m["fault"] == "gear" else f0
        sb_amp = (0.04 if m["fault"] != "gear" else 0.04 + 0.45 * s)
        fr_l, am_l = [m["gmf"]], [base + gmf_gain]
        add((base + gmf_gain) * np.cos(m["gmf"] / f0 * ph + u()),
            1.0, gax, 1.1)
        for k in (1, 2, 3):
            a_sb = sb_amp / k
            if a_sb < 0.02:
                continue
            for sgn in (-1, 1):
                fq = m["gmf"] + sgn * k * sb_sp
                add(a_sb * np.cos(fq / f0 * ph + u()), 1.0, gax, 1.1)
                fr_l.append(fq); am_l.append(a_sb)
        note("GMF", fr_l, am_l)
    if m["fault"] == "bearing":
        brg = m["brg_in"] if m["fault_shaft"] == "in" else m["brg_out"]
        fsh = f0 if m["fault_shaft"] == "in" else f2
        g_nde = 0.5 if m["fault_shaft"] == "in" else 1.5
        bo = brg[m["subtype"]]
        bslip = rng.uniform(0.005, 0.02)
        phb = (bo * (1 + bslip)) * (fsh / f0) * ph
        rw = np.cumsum(rng.normal(0, 0.6 / np.sqrt(FS / max(f0, 3)), n))
        tone = 0.8 * s * np.cos(phb + rw)
        if m["subtype"] == "BPFI":
            tone = tone * (1 + 0.6 * np.cos(fsh / f0 * ph + u()))
        elif m["subtype"] == "BSF":
            tone = tone * (1 + 0.5 * np.cos(brg["FTF"] * fsh / f0 * ph
                                            + u()))
        add(tone, 1.0, 0.4, g_nde)
        add(0.35 * s * np.cos(2 * phb + 2 * rw), 1.0, 0.4, g_nde)
        bf = bo * (1 + bslip) * fsh
        note("BEARING", [bf, 2 * bf], [0.8 * s, 0.35 * s], drifting=True)

    eg = 1.0 if m["component"] == "motor" else 10 ** (-25 / 20)
    two_fe = 2 * m["f_e"]
    add(eg * 0.5 * np.cos(2 * np.pi * two_fe * t + u()), 1.0, 0.25, 0.32)
    add(eg * 0.12 * np.cos(2 * np.pi * m["f_e"] * t + u()), 1.0, 0.25, 0.32)
    note("ELEC", [two_fe, m["f_e"]], [eg * 0.5, eg * 0.12])
    fc_pwm = m["carrier"]
    for k in (-2, -1, 0, 1, 2):
        amp = eg * (0.25 if k == 0 else 0.15 / abs(k))
        add(amp * np.cos(2 * np.pi * (fc_pwm + k * two_fe) * t + u()),
            1.0, 0.25, 0.32)
    note("ELEC_HF", [fc_pwm + k * two_fe for k in (-2, -1, 0, 1, 2)],
         [eg * (0.25 if k == 0 else 0.15 / abs(k)) for k in
          (-2, -1, 0, 1, 2)])
    hum = 2 * m["lf_grid"]
    for k in (1, 2):
        add(0.10 / k * np.cos(2 * np.pi * k * hum * t + u()), 1.0, 0.8, 1.0)
    note("HUM", [hum, 2 * hum], [0.10, 0.05])
    fn = rng.uniform(5, 70)
    for k in (1, 2, 3):
        add(0.12 / k * np.cos(2 * np.pi * k * fn * t + u()), 1.0, 0.8, 1.0)
    note("NEIGHBOR", [fn, 2 * fn, 3 * fn], [0.12, 0.06, 0.04])

    X = np.zeros((n, 3))
    for w, g in comps:
        for c in range(3):
            X[:, c] += g[c] * w
    for c in range(3):                    # independent noise + resonance
        X[:, c] += m["noise"] * rng.standard_normal(n)
        fr_res = rng.uniform(800, 6000)
        lo = max(fr_res * .9, 50) / (FS / 2)
        hi = min(fr_res * 1.1, .48 * FS) / (FS / 2)
        sos = butter(2, [lo, hi], btype="band", output="sos")
        X[:, c] += rng.uniform(2.0, 4.0) * sosfilt(
            sos, rng.standard_normal(n))
    return X


def sample_run_mc(i, truth=None):
    rng = V2.rng_for_run((77_000_000, i))
    m = V2.sample_machine(rng)
    X = synth_run_mc(m, rng, truth=truth)
    X = X + rng.uniform(0.0, 0.2) * rng.standard_normal(X.shape)
    m["run_id"] = i
    return m, X


def fused_peaks(X, fs, tol_hz=0.6):
    """Cross-channel peak confirmation: keep peaks present in >= 2
    channels (within tol), amplitude = max across channels. Single-
    channel noise peaks die; real components survive."""
    from . import peakshor as PS
    lists = [PS.spectral_peaks(X[:, c], fs) for c in range(X.shape[1])]
    out = []
    pf0, pa0, pc0 = lists[0]
    for j, f in enumerate(pf0):
        support, amp, conc = 1, pa0[j], pc0[j]
        for pf, pa, pc in lists[1:]:
            k = np.searchsorted(pf, f)
            for kk in (k - 1, k):
                if 0 <= kk < len(pf) and abs(pf[kk] - f) < max(
                        tol_hz, 0.005 * f):
                    support += 1
                    amp = max(amp, pa[kk])
                    break
        if support >= 2:
            out.append((f, amp, conc))
    # channels 1..2 may hold confirmed peaks missing from channel 0
    pf1, pa1, pc1 = lists[1]
    pf2, pa2, pc2 = lists[2]
    for j, f in enumerate(pf1):
        if any(abs(f - g) < max(tol_hz, 0.005 * f) for g, _, _ in out):
            continue
        k = np.searchsorted(pf2, f)
        for kk in (k - 1, k):
            if 0 <= kk < len(pf2) and abs(pf2[kk] - f) < max(
                    tol_hz, 0.005 * f):
                out.append((f, max(pa1[j], pa2[kk]), pc1[j]))
                break
    if not out:
        return np.array([]), np.array([]), np.array([])
    out.sort()
    pf, pa, pc = map(np.array, zip(*out))
    return pf, pa, pc


def axial_features(X, fs, f_hat):
    """Axial/radial evidence for misalignment subtyping: amplitude
    ratios at orders 1 and 2 between the axial and radial-DE channels."""
    from . import peakshor as PS
    r = {}
    for name, c in (("de", 0), ("ax", 1)):
        pf, pa, _ = PS.spectral_peaks(X[:, c], fs, fmax=min(
            8 * f_hat, 1900.0))
        for k in (1, 2):
            r[f"{name}{k}"] = PS._amp_at_order(pf, pa, f_hat, float(k))
    return {"ax_ratio_1": r["ax1"] / (r["de1"] + 1e-9),
            "ax_ratio_2": r["ax2"] / (r["de2"] + 1e-9)}
