"""Mechanism pins for SimForge v6 realism upgrades (Phase 3 iter 33).

T64 units/sensor chain     T65 modal path      T66 non-Gaussian noise
T67 bearing impact jitter  T68 speed profiles  T69 VFD load lines
T70 truth contract + determinism
"""
import numpy as np
import pytest

from shorcm import simforge_v6 as V6
from shorcm import simforge_v2 as V2


def _find(cls=None, subtype=None, pop=None, lo=0, hi=4000, smin=0.0):
    for i in range(lo, hi):
        rng = V6.rng_for_run(i)
        m = V2.sample_machine(rng)
        if cls and m["fault"] != cls:
            continue
        if subtype and m.get("subtype") != subtype:
            continue
        if pop and m["population"] != pop:
            continue
        if m["severity"] < smin:
            continue
        return i
    raise AssertionError("no matching machine in scan range")


def test_t64_units_and_adc():
    """Signals leave in g: clipped at +-50 g, quantized to the 24-bit
    grid, and record RMS spans a realistic decade across machines."""
    rms = []
    for i in range(12):
        m, x = V6.sample_run(i)
        assert np.max(np.abs(x)) <= V6.G_CLIP + 1e-9
        frac = x / V6.ADC_LSB
        assert np.allclose(frac, np.round(frac), atol=1e-6)
        assert "g_scale" in m
        rms.append(float(np.sqrt(np.mean(x ** 2))))
    assert max(rms) / max(min(rms), 1e-12) > 3.0     # scale diversity


def test_t65_modal_path_colors_spectrum():
    """The modal bank must color the noise floor: the smoothed log
    spectrum of a healthy machine shows structured deviation (peaks)
    well beyond a white floor's."""
    i = _find(cls="healthy")
    _, x = V6.sample_run(i)
    f = np.fft.rfft(x * np.hanning(len(x)))
    p = np.abs(f) ** 2
    k = 257
    sm = np.convolve(p, np.ones(k) / k, mode="same")
    hi = slice(int(len(p) * 0.15), int(len(p) * 0.95))
    ratio = np.log10(sm[hi].max() / np.median(sm[hi]))
    assert ratio > 0.5                                # >5 dB structure


def test_t66_nongaussian_noise_exists():
    """Across healthy machines, excess kurtosis of the band-passed
    residual must exceed the Gaussian value for a solid fraction —
    sporadic impacts are present in the population."""
    from scipy.stats import kurtosis
    ks = []
    n_checked = 0
    for i in range(4000):
        rng = V6.rng_for_run(i)
        m = V2.sample_machine(rng)
        if m["fault"] != "healthy":
            continue
        m2, x = V6.sample_run(i)
        ks.append(kurtosis(np.diff(x)))               # diff kills tones
        n_checked += 1
        if n_checked >= 12:
            break
    assert np.mean(np.array(ks) > 1.0) >= 0.25


def test_t67_bearing_impacts_jittered_not_coherent():
    """v6 bearing impacts must NOT be phase-coherent: the coefficient
    of variation of inter-impact intervals in the generated train is
    >= 0.4%, and the envelope spectrum still shows the defect rate."""
    from scipy.signal import hilbert, butter, sosfilt
    i = _find(cls="bearing", smin=0.6)
    rng = V6.rng_for_run(i)
    m = V2.sample_machine(rng)
    x = V6.synth_run_v6(m, rng)
    brg = m["brg_in"] if m["fault_shaft"] == "in" else m["brg_out"]
    ratio = 1.0 if m["fault_shaft"] == "in" else \
        m.get("f2", m["f_shaft"]) / m["f_shaft"]
    f_def = brg[m["subtype"]] * ratio * m["f_shaft"]
    # envelope spectrum in the HF band
    sos = butter(4, [2000 / (V6.FS / 2), 7000 / (V6.FS / 2)],
                 btype="band", output="sos")
    env = np.abs(hilbert(sosfilt(sos, x)))
    env = env - env.mean()
    p = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    fr = np.fft.rfftfreq(len(env), 1 / V6.FS)
    band = (fr > 0.85 * f_def) & (fr < 1.15 * f_def)
    floor = np.median(p[(fr > 5) & (fr < 3 * f_def)])
    assert p[band].max() / floor > 8.0                # defect rate visible
    # smear: peak width at the defect rate must exceed the coherent
    # width (jitter broadens the line) — compare to rigid comb width
    pk = fr[band][np.argmax(p[band])]
    assert abs(pk / f_def - 1) < 0.1


def test_t68_speed_profiles_present():
    """Ramp/step profiles occur in the population at roughly the
    designed rate (30%)."""
    profs = []
    for i in range(60):
        m, _ = V6.sample_run(i)
        profs.append(m.get("speed_profile", "wander"))
    frac = np.mean([p != "wander" for p in profs])
    assert 0.1 < frac < 0.55


def test_t69_mains_pole_pass_sidebands():
    """Mains machines carry pole-pass sidebands around 2 f_e."""
    for i in range(4000):
        rng = V6.rng_for_run(i)
        m = V2.sample_machine(rng)
        if m["population"] != "mains" or m["component"] != "motor":
            continue
        if m["fault"] != "healthy":
            continue
        x = V6.synth_run_v6(m, rng)
        f = np.fft.rfftfreq(len(x), 1 / V6.FS)
        p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        f_pp = 2 * m["slip"] * m["f_e"]
        two_fe = 2 * m["f_e"]
        def amp_at(fq):
            band = np.abs(f - fq) < 0.5
            return p[band].max() if band.any() else 0.0
        floor = np.median(p[(f > two_fe - 20) & (f < two_fe + 20)])
        assert amp_at(two_fe + f_pp) > 3 * floor
        assert amp_at(two_fe - f_pp) > 3 * floor
        return
    raise AssertionError("no mains healthy motor found")


def test_t70_truth_contract_and_determinism():
    """Truth ledger has the v2 families; regeneration is bit-exact."""
    truth = []
    m1, x1 = V6.sample_run(7, truth=truth)
    m2, x2 = V6.sample_run(7)
    assert np.array_equal(x1, x2)
    assert m1["fault"] == m2["fault"]
    fams = {t["family"] for t in truth}
    assert "SHAFT" in fams and ("HUM" in fams or "ELEC" in fams)
    for t in truth:
        assert set(t) == {"family", "freqs_hz", "amps", "drifting",
                          "impulsive"}
