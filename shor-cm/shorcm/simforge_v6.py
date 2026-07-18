"""SimForge v6: realism upgrades over v2, Phase 3 iteration 33.

v2 (`simforge_v2.py`) is FROZEN — its seeded population underpins the
pinned test suite. v6 is a separate module with its own seed root that
reuses v2's machine sampler (kinematics are fine) and re-renders the
waveform with the realism upgrades the external audit ranked highest
(REVIEW_EXTERNAL_AUDIT.md O9, HANDOFF_PHASE3_LOCAL.md Stage 5):

  R1 physical units + sensor chain
     amplitudes leave in g; accelerometer mounting resonance (2nd-order
     peak 3-6.5 kHz), 24-bit ADC quantization, +-50 g clipping,
     sensor noise floor. (The cascade normalizes to unit RMS at entry,
     so R1 matters for realism/discriminator work and clipping physics,
     not for breaking the frozen feature contract.)
  R2 multi-resonance transmission path
     2-4 structural modes (random centre/damping/gain) replace v2's
     single band-pass floor shaper; the modal bank filters the SOURCE
     signal, so faults arrive through the structure, not beside it.
  R3 non-Gaussian, non-stationary noise
     sporadic non-fault impact bursts (Bernoulli, heavy-tailed
     amplitude, through the same modal bank — the classic kurtogram
     killer), slow noise-floor drift, occasional flow-turbulence burst.
  R4 per-impact bearing slip jitter
     each impact interval jitters independently (sigma 0.5-2%) around
     the drifting defect rate: real bearing lines smear because impacts
     are NOT phase-coherent. Extended spalls (severity > 0.7) add an
     entry/exit double impact. Incipient mode (severity < 0.45) drops
     impact energy near the noise floor — visible only after
     envelope/prewhitening, like real early faults.
  R5 in-record speed profiles
     30%: ramp (up to +-12% linear) or step-dwell profile instead of
     pure sinusoidal wander (audit F6).
  R6 load-varying VFD electrical content
     VFD f_e(t) tracks a slow load profile (+-2.5%); PWM carrier
     drifts; mains machines get pole-pass sidebands 2*s*f_e around
     2*f_e whose amplitude tracks the load profile (audit F5).

Contract notes:
 - sample_machine is v2's (identical kinematic population, new rng
   root BASE_SEED6) so kinematic_sheet() and all downstream metadata
   code work unchanged.
 - synth_run_v6 emits the same truth-ledger records as v2.synth_run.
 - No new rng draws are inserted into paths shared with v2 — v6 owns
   its whole rng stream, so ordering discipline is internal only.
"""
import numpy as np
from scipy.signal import sosfilt, butter, fftconvolve

from . import simforge_v2 as V2

FS = V2.FS
DUR = V2.DUR
BASE_SEED6 = 20260718

G_CLIP = 50.0          # accelerometer clip (+-g)
ADC_LSB = 2 * G_CLIP / 2 ** 24


def _speed_profile(m, rng, n):
    """Instantaneous shaft frequency over the record (R5)."""
    t = np.arange(n) / FS
    f0 = m["f_shaft"]
    mode = rng.random()
    if mode < 0.70:                                   # v2-style wander
        f_inst = f0 * (1 + m["wander"] * np.sin(
            2 * np.pi * rng.uniform(.3, 1.2) * t + rng.uniform(0, 6.28)))
        prof = "wander"
    elif mode < 0.90:                                 # linear ramp
        r = rng.uniform(-0.12, 0.12)
        f_inst = f0 * (1 + r * t / t[-1])
        f_inst = f_inst * (1 + 0.3 * m["wander"] * np.sin(
            2 * np.pi * rng.uniform(.3, 1.2) * t + rng.uniform(0, 6.28)))
        prof = "ramp"
    else:                                             # step-dwell
        r = rng.uniform(-0.08, 0.08)
        t_step = rng.uniform(0.3, 0.7) * t[-1]
        f_inst = f0 * np.where(t < t_step, 1.0, 1.0 + r)
        # soften the step over ~0.5 s (drive dynamics)
        k = max(int(0.5 * FS), 1)
        kern = np.ones(k) / k
        f_inst = np.convolve(f_inst, kern, mode="same")
        f_inst[:5] = f_inst[5]; f_inst[-5:] = f_inst[-6]
        prof = "step"
    return f_inst, prof


def _modal_bank(rng, n_modes=None):
    """R2: random structural modes; returns (sos_list, gains)."""
    if n_modes is None:
        n_modes = int(rng.integers(2, 5))
    bank = []
    for _ in range(n_modes):
        fc = rng.uniform(400, 6200)
        bw = fc * rng.uniform(0.05, 0.25)             # Q ~ 4..20
        lo = max(fc - bw, 40) / (FS / 2)
        hi = min(fc + bw, 0.49 * FS) / (FS / 2)
        sos = butter(2, [lo, hi], btype="band", output="sos")
        bank.append((sos, rng.uniform(0.4, 2.2)))
    return bank


def _through_path(sig, bank, direct=0.35):
    """Source signal through the structure: direct (stiffness) path +
    modal contributions."""
    out = direct * sig
    for sos, g in bank:
        out = out + g * sosfilt(sos, sig)
    return out


def _sensor_chain(x, rng):
    """R1: mounting resonance, gain to g, noise floor, ADC."""
    f_mnt = rng.uniform(3000, 6500)
    bw = f_mnt * rng.uniform(0.10, 0.20)
    lo = (f_mnt - bw) / (FS / 2)
    hi = min(f_mnt + bw, 0.495 * FS) / (FS / 2)
    sos = butter(2, [lo, hi], btype="band", output="sos")
    x = x + rng.uniform(0.5, 1.5) * sosfilt(sos, x)   # resonance lift
    g_scale = float(np.exp(rng.normal(np.log(0.08), 0.6)))  # units -> g
    x = x * g_scale
    x = x + (g_scale * 0.003) * rng.standard_normal(len(x))  # sensor floor
    x = np.clip(x, -G_CLIP, G_CLIP)
    x = np.round(x / ADC_LSB) * ADC_LSB               # 24-bit ADC
    return x, g_scale


def _bearing_impacts(m, rng, ph_shaft, f_inst_shaft, truth_note):
    """R4: jittered impact train through a resonance, replacing v2's
    phase-coherent train. Returns the additive source-domain signal."""
    n = len(ph_shaft)
    brg = m["brg_in"] if m["fault_shaft"] == "in" else m["brg_out"]
    f0 = m["f_shaft"]
    fsh_ratio = 1.0 if m["fault_shaft"] == "in" else \
        m.get("f2", f0) / f0
    bo = brg[m["subtype"]]
    s = m["severity"]
    bslip = rng.uniform(0.005, 0.02)                  # mean slip offset
    jit = rng.uniform(0.005, 0.02)                    # per-impact jitter
    # instantaneous defect rate (Hz) follows the speed profile
    f_def = bo * (1 + bslip) * fsh_ratio * f_inst_shaft
    # generate impact times by integrating the rate with per-interval
    # multiplicative jitter
    times = []
    t_k = float(rng.uniform(0, 1.0 / max(f_def[0], 1e-3)))
    while t_k < DUR:
        times.append(t_k)
        idx = min(int(t_k * FS), n - 1)
        T = 1.0 / max(f_def[idx], 1e-3)
        t_k = t_k + T * (1 + jit * rng.standard_normal())
    k_idx = np.minimum((np.array(times) * FS).astype(int), n - 1)
    train = np.zeros(n)
    if len(k_idx):
        amp_j = 1 + 0.35 * rng.standard_normal(len(k_idx))
        amp_j = np.abs(amp_j)
        if m["subtype"] == "BPFI":                    # load-zone
            amp_j *= 1 + 0.6 * np.cos(fsh_ratio * ph_shaft[k_idx] + 0.7)
        elif m["subtype"] == "BSF":
            amp_j *= 1 + 0.5 * np.cos(
                brg["FTF"] * fsh_ratio * ph_shaft[k_idx] + 0.7)
        train[k_idx] += amp_j
        if s > 0.7:                                   # extended spall
            d_exit = int(rng.uniform(0.2, 0.5) / bo / fsh_ratio *
                         FS / max(f0, 1.0))
            exit_idx = k_idx + max(d_exit, 2)
            ok = exit_idx < n
            train[exit_idx[ok]] += 0.6 * amp_j[ok]
    # defect-excited resonance kernel (own high-freq mode)
    f_res = rng.uniform(2500, 6800)
    tau = rng.uniform(0.0008, 0.0025)
    tk = np.arange(int(5 * tau * FS)) / FS
    kern = np.exp(-tk / tau) * np.sin(2 * np.pi * f_res * tk)
    # incipient scaling: below ~0.45 severity the impacts sit near the
    # broadband floor (envelope-only visibility)
    lvl = 2.2 * s if s >= 0.45 else 0.55 * s
    sig = lvl * fftconvolve(train, kern)[:n]
    # faint tonal residue at the defect order (late-stage only)
    if s >= 0.6:
        rw = np.cumsum(rng.normal(0, 0.6 / np.sqrt(FS / max(f0, 3)), n))
        phb = bo * (1 + bslip) * fsh_ratio * ph_shaft
        sig = sig + 0.12 * s * np.cos(phb + rw)
    bf = bo * (1 + bslip) * fsh_ratio * f0
    truth_note("BEARING", [bf, 2 * bf], [lvl * 0.1, lvl * 0.03],
               drifting=True, impulsive=True)
    return sig


def synth_run_v6(m, rng, truth=None):
    """v6 waveform for a v2-sampled machine. Same truth contract."""
    def note(family, freqs, amps, drifting=False, impulsive=False):
        if truth is not None:
            truth.append({"family": family,
                          "freqs_hz": [float(f) for f in freqs],
                          "amps": [float(a) for a in amps],
                          "drifting": bool(drifting),
                          "impulsive": bool(impulsive)})
    n = int(DUR * FS)
    t = np.arange(n) / FS
    f0 = m["f_shaft"]
    s = m["severity"]
    f_inst, prof = _speed_profile(m, rng, n)          # R5
    m["speed_profile"] = prof
    ph = 2 * np.pi * np.cumsum(f_inst) / FS
    f2 = m.get("f2", f0)
    ph2 = ph * (f2 / f0)
    if m["archetype"] == "fan_belt":
        ph2 = ph2 + np.cumsum(rng.normal(0, 0.4 / np.sqrt(FS / f0), n))
    u = lambda: rng.uniform(0, 2 * np.pi)             # noqa: E731

    src = np.zeros(n)                                 # source domain

    # ---- shaft family (identical amplitude logic to v2) ----
    a1 = m["resid_1x"] + (1.3 * s if m["fault"] == "imbalance" else 0)
    a2 = rng.uniform(0.03, 0.10)
    a3 = 0.0
    if m["fault"] == "misalignment":
        st = m["subtype"]
        if st == "parallel":
            a2 += 0.9 * s; a3 += 0.35 * s
        elif st == "angular":
            a1 += 0.20 * s; a2 += 0.5 * s; a3 += 0.15 * s
        else:
            a2 += 0.7 * s; a3 += 0.10 * s
            src += 0.25 * s * np.cos(4 * ph + u())
    src += a1 * np.cos(ph + u()) + a2 * np.cos(2 * ph + u())
    if a3:
        src += a3 * np.cos(3 * ph + u())
    sh_f = [f0, 2 * f0] + ([3 * f0] if a3 else [])
    sh_a = [a1, a2] + ([a3] if a3 else [])
    if m["fault"] == "misalignment" and m["subtype"] == "coupling":
        sh_f.append(4 * f0); sh_a.append(0.25 * s)
    note("SHAFT", sh_f, sh_a)
    if m["fault"] == "looseness":
        ks = (0.5, 1.5, 2.5, 3.5)
        # cycle-to-cycle variability: looseness raps are not clean tones
        for k in ks:
            am = 0.45 * s / (k + .5)
            wob = 1 + 0.3 * np.cumsum(
                rng.normal(0, 1.0 / np.sqrt(FS), n))
            src += am * np.clip(wob, 0.3, 1.7) * np.cos(k * ph + u())
        note("HALF", [k * f0 for k in ks],
             [0.45 * s / (k + .5) for k in ks])

    # ---- driven shaft + passage (v2 logic) ----
    f_out = m.get("f3", f2)
    ph_out = ph * (f_out / f0)
    if f2 != f0:
        b1 = rng.uniform(0.10, 0.30)
        src += b1 * np.cos(ph2 + u()) + 0.4 * b1 * np.cos(2 * ph2 + u())
        note("SHAFT2", [f2, 2 * f2], [b1, 0.4 * b1],
             drifting=(m["archetype"] == "fan_belt"))
        if f_out != f2:
            b2 = rng.uniform(0.10, 0.30)
            src += b2 * np.cos(ph_out + u())
            note("SHAFT2", [f_out], [b2])
    if "vanes" in m:
        av = rng.uniform(0.15, 0.45)
        src += av * np.cos(m["vanes"] * ph_out + u())
        note("PASSAGE", [m["vanes"] * f_out], [av])
    if "blades" in m:
        ab = rng.uniform(0.15, 0.45)
        src += ab * np.cos(m["blades"] * ph_out + u())
        note("PASSAGE", [m["blades"] * f_out], [ab])
    if "lobes" in m:
        al = rng.uniform(0.3, 0.6)
        src += al * np.cos(2 * m["lobes"] * ph + u())
        src += 0.4 * al * np.cos(4 * m["lobes"] * ph + u())
        note("PASSAGE", [2 * m["lobes"] * f0, 4 * m["lobes"] * f0],
             [al, 0.4 * al])

    # ---- gear mesh (v2 logic, + tooth-to-tooth transmission error) ----
    def mesh(gmf_hz, base_amp, sb_space_hz, sb_amp, tag):
        te = 1 + 0.08 * np.cumsum(rng.normal(0, 1.0 / np.sqrt(FS), n))
        te = np.clip(te, 0.6, 1.4)                    # slow TE variation
        x_l = base_amp * te * np.cos(gmf_hz / f0 * ph + u())
        fr, am = [gmf_hz], [base_amp]
        for k in (1, 2, 3):
            a_sb = sb_amp / k
            if a_sb < 0.02:
                continue
            for sgn in (-1, 1):
                fq = gmf_hz + sgn * k * sb_space_hz
                x_l = x_l + a_sb * np.cos(fq / f0 * ph + u())
                fr.append(fq); am.append(a_sb)
        note(tag, fr, am)
        return x_l

    if m["archetype"] in ("pump_gearbox1", "fan_gearbox2"):
        base = rng.uniform(0.2, 0.5)
        sb0 = rng.uniform(0.04, 0.10)
        f_flt = f0 if m.get("fault_gear") == "in" else f2
        sb_amp = sb0 + (0.45 * s if m["fault"] == "gear" else 0.0)
        gmf_gain = 0.3 * s if m["fault"] == "gear" else 0.0
        src += mesh(m["gmf"], base + gmf_gain,
                    f_flt if m["fault"] == "gear" else f0, sb_amp, "GMF")
        if m["archetype"] == "fan_gearbox2":
            src += mesh(m["gmf2"], rng.uniform(0.15, 0.4),
                        f2, rng.uniform(0.04, 0.08), "GMF2")
    if m["archetype"] == "pump_planetary":
        base = rng.uniform(0.2, 0.5)
        fc = m["fc"]
        sb_sp = {"sun": f0 - fc, "planet": 2 * fc * m["Zr"] / m["Zp"],
                 "ring": m["Np"] * fc}.get(m["subtype"], fc)
        sb_amp = 0.05 + (0.45 * s if m["fault"] == "gear" else 0.0)
        src += mesh(m["gmf"], base + (0.3 * s if m["fault"] == "gear"
                                      else 0.0),
                    sb_sp if m["fault"] == "gear" else fc, sb_amp, "GMF")
        src += 0.1 * np.cos(m["Np"] * fc / f0 * ph + u())

    # ---- bearing fault: jittered impact physics (R4) ----
    if m["fault"] == "bearing":
        src += _bearing_impacts(m, rng, ph, f_inst, note)

    # ---- transmission path: source through the structure (R2) ----
    bank = _modal_bank(rng)
    x = _through_path(src, bank)

    # ---- electrical, hum, neighbor (post-path: airborne/electrical) --
    eg = 1.0 if m["component"] == "motor" else 10 ** (-25 / 20)
    load = 1 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.05, 0.3) * t
                            + u())                    # slow load profile
    if m["population"] == "vfd":                      # R6: f_e tracks load
        fe_t = m["f_e"] * (1 + 0.025 * (load - 1))
        ph_e = 2 * np.pi * np.cumsum(fe_t) / FS
        x += eg * 0.5 * np.cos(2 * ph_e + u())
        x += eg * 0.12 * np.cos(ph_e + u())
        fc_pwm = m["carrier"] * (1 + 0.01 * np.sin(
            2 * np.pi * rng.uniform(0.02, 0.1) * t + u()))
        ph_c = 2 * np.pi * np.cumsum(fc_pwm) / FS
        for k in (-2, -1, 0, 1, 2):
            amp = eg * (0.25 if k == 0 else 0.15 / abs(k))
            x += amp * np.cos(ph_c + k * 2 * ph_e + u())
    else:                                             # mains + pole-pass
        two_fe = 2 * m["f_e"]
        x += eg * 0.5 * np.cos(2 * np.pi * two_fe * t + u())
        x += eg * 0.12 * np.cos(2 * np.pi * m["f_e"] * t + u())
        f_pp = 2 * m["slip"] * m["f_e"]               # pole-pass
        a_pp = eg * 0.15 * np.clip(load, 0.4, 1.6)
        x += a_pp * np.cos(2 * np.pi * (two_fe + f_pp) * t + u())
        x += a_pp * np.cos(2 * np.pi * (two_fe - f_pp) * t + u())
        fc_pwm = m["carrier"]
        for k in (-2, -1, 0, 1, 2):
            amp = eg * (0.25 if k == 0 else 0.15 / abs(k))
            x += amp * np.cos(2 * np.pi * (fc_pwm + k * two_fe) * t + u())
    note("ELEC", [2 * m["f_e"], m["f_e"]], [eg * 0.5, eg * 0.12])
    hum = 2 * m["lf_grid"]
    for k in (1, 2):
        x += 0.10 / k * np.cos(2 * np.pi * k * hum * t + u())
    note("HUM", [hum, 2 * hum], [0.10, 0.05])
    fn = rng.uniform(5, 70)
    for k in (1, 2, 3):
        x += 0.12 / k * np.cos(2 * np.pi * k * fn * t + u())
    note("NEIGHBOR", [fn, 2 * fn, 3 * fn], [0.12, 0.06, 0.04])

    # ---- noise: colored floor + non-Gaussian bursts (R3) ----
    drift = 1 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.03, 0.15) * t
                             + u())
    x += m["noise"] * drift * rng.standard_normal(n)
    x += _through_path(0.8 * m["noise"] * rng.standard_normal(n),
                      bank, direct=0.0)               # structure-colored
    # sporadic non-fault impacts (handling, cavitation clicks): the
    # kurtogram killer — heavy-tailed, NOT periodic
    n_imp = rng.poisson(rng.uniform(0.5, 5.0) * DUR)
    if n_imp:
        idx = rng.integers(0, n, size=n_imp)
        amp = rng.standard_cauchy(n_imp)
        amp = np.sign(amp) * np.minimum(np.abs(amp), 8.0)
        spikes = np.zeros(n)
        spikes[idx] = amp * m["noise"] * rng.uniform(2.0, 6.0)
        x += _through_path(spikes, bank, direct=0.1)
    if rng.random() < 0.25:                           # turbulence burst
        b0 = int(rng.uniform(0, 0.7) * n)
        b1 = b0 + int(rng.uniform(0.2, 1.0) * FS)
        burst = np.zeros(n)
        burst[b0:min(b1, n)] = rng.standard_normal(min(b1, n) - b0)
        x += _through_path(1.5 * m["noise"] * burst, bank, direct=0.0)

    # ---- sensor chain (R1) ----
    x, g_scale = _sensor_chain(x, rng)
    m["g_scale"] = g_scale
    return x


def rng_for_run(i):
    return np.random.default_rng(np.random.SeedSequence((BASE_SEED6, i)))


def sample_run(i, truth=None):
    """Corpus entry point, mirrors v2.sample_run."""
    rng = rng_for_run(i)
    m = V2.sample_machine(rng)
    x = synth_run_v6(m, rng, truth=truth)
    m["run_id"] = i
    m["forge"] = "v6"
    return m, x
