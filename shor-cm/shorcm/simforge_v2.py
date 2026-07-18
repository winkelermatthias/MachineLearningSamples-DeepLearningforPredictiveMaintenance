"""SimForge v2: archetype-based synthetic machinery population.

Physics- and standards-derived, never fitted to MAFAULDA. Expands v1 to
the machine types MachineDoctor actually meets (Matthias, 2026-07-11):

Archetypes (kinematic graphs, all motor-driven):
  pump_direct     centrifugal end-suction/double-suction (vane pass) or
                  gear pump (mesh at z_p teeth)
  fan_direct      blade pass, 3..12 blades
  blower_roots    2-3 lobe roots blower, lobe pass at 2*lobes*f0
  fan_belt        belt drive: ratio D1/D2 in [0.4, 2.2] with 0.5-2% creep
                  slip -> the second shaft is a NEAR-rational lattice
  pump_gearbox1   single-stage parallel gearbox, coprime-biased teeth
  fan_gearbox2    two-stage parallel gearbox
  pump_planetary  planetary: ring fixed, sun in, carrier out

Bearings from geometry, not a lookup table: element count Z and d/D
pitch ratio drive BPFO/BPFI/BSF/FTF. Size follows speed (Matthias):
low-speed machines carry BIG bearings (Z 12-24), high-speed carry small
(Z 7-12). Both shafts of geared/belt trains get their own bearing.

Faults (6-way): healthy, imbalance, misalignment (subtypes parallel /
angular / coupling — coupling issues per the field taxonomy), looseness,
bearing (outer/inner/ball, on either shaft, BPFI amplitude-modulated at
1x, ball at FTF), gear (parallel: wear / local tooth with sidebands
spaced by the faulty shaft; planetary: sun / planet / ring with
carrier-spaced modulation).

Residual imbalance is drawn 0.05-0.35 (v1 pinned it at 0.25, which made
the misalignment-needs-1x rule untestable — the masquerade population
now exists).

Every run emits complete composition ground truth (same contract as
simforge_lite.synth_run truth), plus the kinematic sheet in the machine
dict. Seed-regeneratable: (BASE_SEED2, run index).
"""
import numpy as np
from scipy.signal import sosfilt, butter

FS = 16_384
DUR = 4.0
BASE_SEED2 = 20260710

ARCHETYPES = ["pump_direct", "fan_direct", "blower_roots", "fan_belt",
              "pump_gearbox1", "fan_gearbox2", "pump_planetary"]
FAULTS6 = ["healthy", "imbalance", "misalignment", "looseness",
           "bearing", "gear"]

_COPRIME_Z = [(19, 43), (17, 53), (23, 47), (29, 61), (31, 64), (21, 52),
              (25, 57), (18, 41), (27, 59), (20, 49)]
_PLANET_SETS = [(9, 18, 45), (12, 21, 54), (15, 24, 63), (11, 20, 51),
                (13, 23, 59)]                     # (Zs, Zp, Zr), Zr=Zs+2Zp


def bearing_orders(rng, big):
    """Physical bearing kinematics. big=True: low-speed pillow-block
    class (many rollers); False: small high-speed deep-groove."""
    Z = int(rng.integers(12, 25)) if big else int(rng.integers(7, 13))
    g = rng.uniform(0.12, 0.22) if big else rng.uniform(0.18, 0.30)
    # g = (d/D) cos(phi)
    ftf = 0.5 * (1 - g)
    bpfo = Z * ftf
    bpfi = Z * 0.5 * (1 + g)
    bsf = (1.0 / (2 * g)) * (1 - g * g)          # (D/2d)(1-g^2), cos~1
    return {"Z": Z, "g": round(g, 4), "FTF": ftf, "BPFO": bpfo,
            "BPFI": bpfi, "BSF": bsf}


def sample_machine(rng):
    arch = ARCHETYPES[int(rng.integers(len(ARCHETYPES)))]
    # electrical population drives the input (motor) shaft
    pop = "vfd" if rng.random() < 0.5 else "mains"
    p = int(rng.choice([1, 2, 3]))
    slip = rng.uniform(0.005, 0.03)
    lf_grid = float(rng.choice([50.0, 60.0]))
    # speed class: geared/big machines run slow input or slow output
    low_speed = bool(rng.random() < (0.45 if arch in
                     ("fan_direct", "pump_planetary", "fan_gearbox2")
                     else 0.2))
    if pop == "mains":
        f_e = lf_grid
        f0 = f_e / p * (1 - slip)
        if low_speed and p == 1:
            p = int(rng.choice([2, 3])); f0 = f_e / p * (1 - slip)
    else:
        f0 = rng.uniform(3.5, 12.0) if low_speed else rng.uniform(15.0, 95.0)
        f_e = f0 * p / (1 - slip)
    m = {"archetype": arch, "population": pop, "pole_pairs": p,
         "slip": slip, "lf_grid": lf_grid, "f_e": f_e, "f_shaft": f0,
         "low_speed": low_speed,
         "component": "motor" if rng.random() < 0.5 else "driven",
         "wander": rng.uniform(0.002, 0.015),
         "carrier": rng.uniform(2000, 6000),
         "resid_1x": rng.uniform(0.05, 0.35),
         "noise": rng.uniform(0.25, 0.5)}
    # driven-side kinematics
    if arch == "pump_direct":
        ptype = str(rng.choice(["end_suction", "double_suction",
                                "gear_pump"], p=[0.5, 0.3, 0.2]))
        m["pump_type"] = ptype
        m["vanes"] = int(rng.choice([5, 6, 7] if ptype == "end_suction"
                                    else ([6, 7, 8] if ptype ==
                                          "double_suction"
                                          else [9, 11, 13])))
        m["f2"] = f0
    elif arch == "fan_direct":
        m["blades"] = int(rng.choice([3, 4, 5, 6, 7, 8, 9, 10, 12]))
        m["f2"] = f0
    elif arch == "blower_roots":
        m["lobes"] = int(rng.choice([2, 3]))
        m["f2"] = f0
    elif arch == "fan_belt":
        m["belt_ratio"] = rng.uniform(0.4, 2.2)
        m["belt_slip"] = rng.uniform(0.005, 0.02)
        m["f2"] = f0 * m["belt_ratio"] * (1 - m["belt_slip"])
        m["blades"] = int(rng.choice([4, 5, 6, 7, 8, 9]))
    elif arch == "pump_gearbox1":
        z1, z2 = _COPRIME_Z[int(rng.integers(len(_COPRIME_Z)))]
        if rng.random() < 0.3:
            z1, z2 = z2, z1                       # speed-increasing
        m["z1"], m["z2"] = z1, z2
        m["f2"] = f0 * z1 / z2
        m["gmf"] = f0 * z1
        m["vanes"] = int(rng.choice([5, 6, 7]))
    elif arch == "fan_gearbox2":
        z1, z2 = _COPRIME_Z[int(rng.integers(len(_COPRIME_Z)))]
        z3, z4 = _COPRIME_Z[int(rng.integers(len(_COPRIME_Z)))]
        m["z1"], m["z2"], m["z3"], m["z4"] = z1, z2, z3, z4
        m["f2"] = f0 * z1 / z2                    # intermediate
        m["f3"] = m["f2"] * z3 / z4               # output
        m["gmf"] = f0 * z1
        m["gmf2"] = m["f2"] * z3
        m["blades"] = int(rng.choice([4, 5, 6, 7, 8]))
    elif arch == "pump_planetary":
        Zs, Zp, Zr = _PLANET_SETS[int(rng.integers(len(_PLANET_SETS)))]
        Np = int(rng.choice([3, 4, 5]))
        m["Zs"], m["Zp"], m["Zr"], m["Np"] = Zs, Zp, Zr, Np
        m["fc"] = f0 * Zs / (Zs + Zr)             # carrier = output
        m["gmf"] = m["fc"] * Zr
        m["f2"] = m["fc"]
        m["vanes"] = int(rng.choice([5, 6, 7]))
    # bearings: input shaft speed class + output shaft speed class
    m["brg_in"] = bearing_orders(rng, big=low_speed)
    f2 = m.get("f2", f0)
    m["brg_out"] = bearing_orders(rng, big=bool(f2 < 12.0))
    # fault
    fault = FAULTS6[int(rng.integers(len(FAULTS6)))]
    if fault == "gear" and arch in ("pump_direct", "fan_direct",
                                    "blower_roots", "fan_belt"):
        fault = str(rng.choice(["bearing", "misalignment", "looseness",
                                "imbalance"]))
    m["fault"] = fault
    m["severity"] = float(rng.uniform(0.3, 1.0)) if fault != "healthy" \
        else 0.0
    if fault == "misalignment":
        m["subtype"] = str(rng.choice(["parallel", "angular", "coupling"]))
    elif fault == "bearing":
        m["subtype"] = str(rng.choice(["BPFO", "BPFI", "BSF"]))
        m["fault_shaft"] = "in" if (rng.random() < 0.6 or
                                    m.get("f2", f0) == f0) else "out"
    elif fault == "gear":
        if arch == "pump_planetary":
            m["subtype"] = str(rng.choice(["sun", "planet", "ring"]))
        else:
            m["subtype"] = str(rng.choice(["wear", "tooth"]))
            m["fault_gear"] = "in" if rng.random() < 0.5 else "out"
    else:
        m["subtype"] = ""
    m["equipment"] = f"{arch} P-{rng.integers(100, 999)}"
    return m


def synth_run(m, rng, truth=None, omega1x=1.0):
    """Waveform + optional composition truth. Truth appends never touch
    the rng stream (same contract as simforge_lite).
    omega1x: multiplies the 1x (mass-force) amplitude — imbalance force
    scales with omega^2, so monitoring sequences pass
    (f_op / f_base)^2 here. Pure multiplication: rng order unchanged."""
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
    ph = 2 * np.pi * np.cumsum(f_inst) / FS       # input shaft phase
    f2 = m.get("f2", f0)
    ph2 = ph * (f2 / f0)                          # kinematically locked
    if m["archetype"] == "fan_belt":              # belt creep: extra walk
        ph2 = ph2 + np.cumsum(rng.normal(0, 0.4 / np.sqrt(FS / f0), n))
    u = lambda: rng.uniform(0, 2 * np.pi)         # noqa: E731
    x = m["noise"] * rng.standard_normal(n)

    # input shaft family (residual imbalance now 0.05-0.35!)
    a1 = (m["resid_1x"] + (1.3 * s if m["fault"] == "imbalance" else 0)) \
        * omega1x
    a2 = rng.uniform(0.03, 0.10)
    a3 = 0.0
    if m["fault"] == "misalignment":
        st = m["subtype"]
        if st == "parallel":
            a2 += 0.9 * s; a3 += 0.35 * s
        elif st == "angular":
            a1 += 0.20 * s; a2 += 0.5 * s; a3 += 0.15 * s
        else:                                     # coupling wear/lock
            a2 += 0.7 * s; a3 += 0.10 * s
            x += 0.25 * s * np.cos(4 * ph + u())
    x += a1 * np.cos(ph + u()) + a2 * np.cos(2 * ph + u())
    if a3:
        x += a3 * np.cos(3 * ph + u())
    sh_f = [f0, 2 * f0] + ([3 * f0] if a3 else [])
    sh_a = [a1, a2] + ([a3] if a3 else [])
    if m["fault"] == "misalignment" and m["subtype"] == "coupling":
        sh_f.append(4 * f0); sh_a.append(0.25 * s)
    note("SHAFT", sh_f, sh_a)
    if m["fault"] == "looseness":
        ks = (0.5, 1.5, 2.5, 3.5)
        for k in ks:
            x += 0.45 * s * np.cos(k * ph + u()) / (k + .5)
        note("HALF", [k * f0 for k in ks], [0.45 * s / (k + .5) for k in ks])

    # driven-shaft family (second lattice) + passage tones. For the
    # two-stage box the fan rides the OUTPUT shaft f3, not the
    # intermediate f2.
    f_out = m.get("f3", f2)
    ph_out = ph * (f_out / f0)
    if f2 != f0:
        b1 = rng.uniform(0.10, 0.30)
        x += b1 * np.cos(ph2 + u()) + 0.4 * b1 * np.cos(2 * ph2 + u())
        note("SHAFT2", [f2, 2 * f2], [b1, 0.4 * b1],
             drifting=(m["archetype"] == "fan_belt"))
        if f_out != f2:
            b2 = rng.uniform(0.10, 0.30)
            x += b2 * np.cos(ph_out + u())
            note("SHAFT2", [f_out], [b2])
    if "vanes" in m:                              # gear pump: vane count
        av = rng.uniform(0.15, 0.45)              # IS the tooth count,
        x += av * np.cos(m["vanes"] * ph_out + u())   # one mesh tone
        note("PASSAGE", [m["vanes"] * f_out], [av])
    if "blades" in m:
        ab = rng.uniform(0.15, 0.45)
        x += ab * np.cos(m["blades"] * ph_out + u())
        note("PASSAGE", [m["blades"] * f_out], [ab])
    if "lobes" in m:
        al = rng.uniform(0.3, 0.6)                # roots lobe pass, strong
        x += al * np.cos(2 * m["lobes"] * ph + u())
        x += 0.4 * al * np.cos(4 * m["lobes"] * ph + u())
        note("PASSAGE", [2 * m["lobes"] * f0, 4 * m["lobes"] * f0],
             [al, 0.4 * al])

    # gear mesh families (healthy mesh always visible on geared trains)
    def mesh(gmf_hz, ph_carrier_hz, base_amp, sb_space_hz, sb_amp, tag):
        agm = base_amp
        x_l = agm * np.cos(gmf_hz / f0 * ph + u())
        fr, am = [gmf_hz], [agm]
        for k in (1, 2, 3):
            a_sb = sb_amp / k
            if a_sb < 0.02:
                continue
            for sgn in (-1, 1):
                fq = gmf_hz + sgn * k * sb_space_hz
                x_l_k = a_sb * np.cos(fq / f0 * ph + u())
                x_l = x_l + x_l_k
                fr.append(fq); am.append(a_sb)
        note(tag, fr, am)
        return x_l

    if m["archetype"] in ("pump_gearbox1", "fan_gearbox2"):
        base = rng.uniform(0.2, 0.5)
        sb0 = rng.uniform(0.04, 0.10)
        f_flt = f0 if m.get("fault_gear") == "in" else f2
        sb_amp = sb0 + (0.45 * s if m["fault"] == "gear" else 0.0)
        gmf_gain = 0.3 * s if m["fault"] == "gear" else 0.0
        x += mesh(m["gmf"], f0, base + gmf_gain,
                  f_flt if m["fault"] == "gear" else f0, sb_amp, "GMF")
        if m["archetype"] == "fan_gearbox2":
            x += mesh(m["gmf2"], f2, rng.uniform(0.15, 0.4),
                      f2, rng.uniform(0.04, 0.08), "GMF2")
    if m["archetype"] == "pump_planetary":
        base = rng.uniform(0.2, 0.5)
        fc = m["fc"]
        # planet spin (relative to carrier) fp = fc * Zr / Zp; a planet
        # defect strikes ring and sun once per spin each -> 2 fp fan
        sb_sp = {"sun": f0 - fc, "planet": 2 * fc * m["Zr"] / m["Zp"],
                 "ring": m["Np"] * fc}.get(m["subtype"], fc)
        sb_amp = 0.05 + (0.45 * s if m["fault"] == "gear" else 0.0)
        x += mesh(m["gmf"], fc, base + (0.3 * s if m["fault"] == "gear"
                                        else 0.0),
                  sb_sp if m["fault"] == "gear" else fc, sb_amp, "GMF")
        x += 0.1 * np.cos(m["Np"] * fc / f0 * ph + u())  # planet pass

    # bearing fault (drifting tone on its shaft, race-correct modulation)
    if m["fault"] == "bearing":
        brg = m["brg_in"] if m["fault_shaft"] == "in" else m["brg_out"]
        fsh = f0 if m["fault_shaft"] == "in" else f2
        bo = brg[m["subtype"]]
        bslip = rng.uniform(0.005, 0.02)
        phb = (bo * (1 + bslip)) * (fsh / f0) * ph
        rw = np.cumsum(rng.normal(0, 0.6 / np.sqrt(FS / max(f0, 3)), n))
        tone = 0.8 * s * np.cos(phb + rw)
        if m["subtype"] == "BPFI":                # 1x modulation sidebands
            tone = tone * (1 + 0.6 * np.cos(fsh / f0 * ph + u()))
        elif m["subtype"] == "BSF":               # cage modulation
            tone = tone * (1 + 0.5 * np.cos(brg["FTF"] * fsh / f0 * ph
                                            + u()))
        x += tone
        x += 0.35 * s * np.cos(2 * phb + 2 * rw)
        bf = bo * (1 + bslip) * fsh
        note("BEARING", [bf, 2 * bf], [0.8 * s, 0.35 * s], drifting=True)

    # electrical (motor point full, driven point -25 dB), hum, neighbor
    eg = 1.0 if m["component"] == "motor" else 10 ** (-25 / 20)
    two_fe = 2 * m["f_e"]
    x += eg * 0.5 * np.cos(2 * np.pi * two_fe * t + u())
    x += eg * 0.12 * np.cos(2 * np.pi * m["f_e"] * t + u())
    note("ELEC", [two_fe, m["f_e"]], [eg * 0.5, eg * 0.12])
    fc_pwm = m["carrier"]
    for k in (-2, -1, 0, 1, 2):
        amp = eg * (0.25 if k == 0 else 0.15 / abs(k))
        x += amp * np.cos(2 * np.pi * (fc_pwm + k * two_fe) * t + u())
    note("ELEC_HF", [fc_pwm + k * two_fe for k in (-2, -1, 0, 1, 2)],
         [eg * (0.25 if k == 0 else 0.15 / abs(k)) for k in
          (-2, -1, 0, 1, 2)])
    hum = 2 * m["lf_grid"]
    for k in (1, 2):
        x += 0.10 / k * np.cos(2 * np.pi * k * hum * t + u())
    note("HUM", [hum, 2 * hum], [0.10, 0.05])
    fn = rng.uniform(5, 70)
    for k in (1, 2, 3):
        x += 0.12 / k * np.cos(2 * np.pi * k * fn * t + u())
    note("NEIGHBOR", [fn, 2 * fn, 3 * fn], [0.12, 0.06, 0.04])

    # random resonance floor
    fr_res = rng.uniform(800, 6000)
    lo = max(fr_res * .9, 50) / (FS / 2)
    hi = min(fr_res * 1.1, .48 * FS) / (FS / 2)
    sos = butter(2, [lo, hi], btype="band", output="sos")
    x += rng.uniform(2.0, 4.0) * sosfilt(sos, rng.standard_normal(n))
    return x


def rng_for_run(i):
    return np.random.default_rng(np.random.SeedSequence((BASE_SEED2, i)))


def sample_run(i, truth=None):
    """Corpus entry point, v1.1-style one-sided hardening kept."""
    rng = rng_for_run(i)
    m = sample_machine(rng)
    x = synth_run(m, rng, truth=truth)
    extra_noise = float(rng.uniform(0.0, 0.2))
    x = x + extra_noise * rng.standard_normal(len(x))
    m["run_id"] = i
    m["extra_noise"] = extra_noise
    return m, x
