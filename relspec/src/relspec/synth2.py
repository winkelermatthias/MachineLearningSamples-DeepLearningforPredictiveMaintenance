"""Synthetic generator v2: wider physics, same exact ground truth.

Extends synth.py without touching it (v1 results stay reproducible bit for
bit). What is new, and why it earns a place in a compression evaluation:

  speed PROFILES     ramps and dwell inside one acquisition. Wander was v1's
                     only speed threat; a run-up is the hard case for a
                     Welch-based extractor and the showcase for order tracking.
  rotor bar damage   sidebands at +/- 2*slip*f_line in FREQUENCY around order
                     harmonics: content that is neither order-locked nor
                     fixed, the worst case for any single normalisation.
  VFD line family    tones locked to 2*f_line regardless of shaft speed -
                     the frequency-locked structure the anchor study warned
                     about, now present in the fleet on purpose.
  belt drives        a sub-1x comb at the belt pass ratio, plus flutter.
  bent shaft         1x/2x loading with axial coupling.
  resonance shift    a grown spall lowers the ringing mode and its Q: the
                     carrier moves while the modulation stays, so the codec's
                     band-gain layer gets exercised, not just peak deltas.
  sensor pathologies clipping and dropouts. Real fleets upload them daily;
                     a codec that has never seen a clipped frame will meet
                     one in production.

Every fault severity remains continuous in [0,1] with exact ground truth.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.signal import butter, filtfilt, lfilter
from .synth import (Bearing, CATALOG, Kinematics, FaultState, MachineSpec,
                    _impulse_train, _ring, _modulate, _pink, progression)

@dataclass
class FaultState2(FaultState):
    bent_shaft: float = 0.0
    rotor_bar: float = 0.0
    belt_wear: float = 0.0
    resonance_shift: float = 0.0     # spall growth moving the ringing mode
    sensor_clip: float = 0.0
    sensor_dropout: float = 0.0

@dataclass
class MachineSpec2(MachineSpec):
    belt_ratio: float = 0.0          # belt pass frequency as a shaft ratio
    vfd_line_hz: float = 0.0         # inverter output line frequency
    slip_nominal: float = 0.02

def _speed_profile(kind, fr_hz, n, fs, rng, wander_pct=0.3):
    """Instantaneous shaft frequency over one acquisition.
    'steady' = v1 behaviour; 'ramp' sweeps a fraction of nominal; 'dwell'
    ramps, holds, ramps back - a batch machine changing recipe.

    The wander is centred EXPLICITLY. v1 normalised the low-passed pink noise
    by its std alone, and a 0.08 Hz low-pass leaves an almost-constant signal
    whose mean dwarfs its std - so "0.3% wander" silently became a random
    CONSTANT speed offset of up to +/-15% and every synthetic acquisition ran
    at a speed other than the one requested. v1's blind 20-70 Hz estimator
    absorbed the offset, which is why nothing caught it; a nominal-guided
    estimator surfaced it within one fleet run."""
    t = np.arange(n)/fs
    w = _pink(n, rng)
    b, a = butter(2, 0.08/(fs/2)); w = filtfilt(b, a, w)
    w -= w.mean(); w /= (w.std()+1e-12)
    base = np.full(n, fr_hz)
    if kind == 'ramp':
        f0, f1 = fr_hz*(1-0.06), fr_hz*(1+0.06)
        base = f0+(f1-f0)*t/t[-1]
    elif kind == 'dwell':
        third = n//3
        base[:third] = np.linspace(fr_hz*0.94, fr_hz*1.04, third)
        base[third:2*third] = fr_hz*1.04
        base[2*third:] = np.linspace(fr_hz*1.04, fr_hz*0.97, n-2*third)
    return base*(1.0+wander_pct/100.0*w)

def generate2(spec: MachineSpec2, fault: FaultState2, seconds: float,
              fr_hz: float, load: float = 1.0, speed_wander_pct: float = 0.3,
              profile: str = 'steady', running: bool = True,
              rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    fs = spec.fs; n = int(seconds*fs)
    t = np.arange(n)/fs

    if not running:
        x = spec.noise_g*(0.55*rng.normal(0, 1, n)+0.45*_pink(n, rng))
        x += 0.03*spec.base_rms_g*np.sin(2*np.pi*49.8*t+rng.uniform(0, 6.28))
        return x.astype(np.float32)

    fr_t = _speed_profile(profile, fr_hz, n, fs, rng, speed_wander_pct)
    theta = np.cumsum(fr_t)/fs
    x = np.zeros(n)
    A = spec.base_rms_g*(0.6+0.4*load)

    # ---- shaft harmonics; bent shaft loads 1x and 2x like a misalignment
    #      that will not shim out
    h_amp = {1: 0.35+1.9*fault.imbalance+1.3*fault.bent_shaft,
             2: 0.16+2.4*fault.misalignment+0.8*fault.bent_shaft,
             3: 0.08+1.7*fault.looseness,
             4: 0.05+0.9*fault.looseness,
             5: 0.03, 6: 0.02, 7: 0.015, 8: 0.012}
    for k, amp in h_amp.items():
        x += A*amp*np.sin(2*np.pi*k*theta+rng.uniform(0, 6.28))
    if fault.looseness > 0.2:
        for k in (0.5, 1.5, 2.5):
            x += A*0.5*fault.looseness*np.sin(2*np.pi*k*theta+rng.uniform(0, 6.28))

    # ---- rotor bar: current sidebands at 2*slip*f_line in Hz around the
    #      1x and 2x mechanical lines. Order-normalising cannot absorb these:
    #      the offset is fixed in Hz while the carrier moves with speed.
    if fault.rotor_bar > 0.005 and spec.vfd_line_hz > 0:
        s = spec.slip_nominal*(0.6+0.7*load)
        df = 2*s*spec.vfd_line_hz
        for k, base_amp in ((1, 0.9), (2, 0.4)):
            for sgn in (-1, 1):
                x += A*base_amp*1.4*fault.rotor_bar * \
                    np.sin(2*np.pi*(k*theta+sgn*df*t)+rng.uniform(0, 6.28))

    # ---- VFD electrical family: locked to the LINE, not the shaft
    if spec.vfd_line_hz > 0:
        fl = spec.vfd_line_hz
        x += A*0.12*np.sin(2*np.pi*2*fl*t+rng.uniform(0, 6.28))
        x += A*0.05*np.sin(2*np.pi*4*fl*t+rng.uniform(0, 6.28))

    # ---- belt drive: sub-1x comb at the belt pass ratio with flutter
    if spec.belt_ratio > 0:
        fb = spec.belt_ratio
        for k in range(1, 5):
            amp = (0.10+1.5*fault.belt_wear)/k
            ph = rng.uniform(0, 6.28)
            tone = np.sin(2*np.pi*k*fb*theta+ph)
            if fault.belt_wear > 0.1:            # flutter: slow AM
                tone = tone*(1.0+0.5*fault.belt_wear *
                             np.sin(2*np.pi*0.12*fb*theta))
            x += A*amp*tone

    # ---- bearing tones ring resonances that MOVE as the spall grows
    res_f = tuple(f*(1.0-0.12*fault.resonance_shift) for f in spec.resonances_hz)
    res_q = tuple(q*(1.0-0.35*fault.resonance_shift) for q in spec.resonance_q)
    modes = [('outer_race', 'bpfo', fault.outer_race, 'ftf', 0.25, 0.008),
             ('inner_race', 'bpfi', fault.inner_race, None, 0.55, 0.010),
             ('rolling_element', 'bsf', fault.rolling_element, 'ftf', 0.45, 0.015),
             ('cage', 'ftf', fault.cage, None, 0.20, 0.020)]
    for bearing, ratio, _pos in spec.kin.bearings:
        for mode, attr, sev, modattr, depth, jit in modes:
            if sev <= 0.005: continue
            f_def = getattr(bearing, attr)*fr_hz*ratio
            train = _impulse_train(n, fs, f_def, jit, rng)
            ri = rng.integers(0, len(res_f))
            ring = _ring(n, fs, res_f[ri], res_q[ri], rng)
            sig = np.convolve(train, ring)[:n]
            f_mod = fr_hz*ratio if modattr is None else \
                getattr(bearing, modattr)*fr_hz*ratio
            sig = _modulate(sig, fs, f_mod, depth*min(1.0, sev*1.6))
            sig /= (sig.std()+1e-12)
            x += A*(0.9*sev**0.75)*sig

    # ---- gear mesh with 1x sidebands
    if spec.kin.gear_teeth:
        nt = spec.kin.gear_teeth[0]
        gm = A*(0.30+1.4*fault.gear_wear)*np.sin(2*np.pi*nt*theta+rng.uniform(0, 6.28))
        gm = gm*(1.0+(0.15+0.9*fault.gear_wear)*np.sin(2*np.pi*theta))
        x += gm
        x += A*(0.10+0.7*fault.gear_wear)*np.sin(2*np.pi*2*nt*theta+rng.uniform(0, 6.28))

    # ---- vane / blade passing, cavitation, lubrication
    if spec.kin.n_vanes:
        nv = spec.kin.n_vanes
        x += A*(0.22+0.5*fault.cavitation)*np.sin(2*np.pi*nv*theta+rng.uniform(0, 6.28))
        if fault.cavitation > 0.01:
            hs = _pink(n, rng)
            bb, aa = butter(4, [min(1200, 0.45*fs/2)/(fs/2),
                                min(5500, 0.9*fs/2)/(fs/2)], btype='band')
            x += A*2.2*fault.cavitation*filtfilt(bb, aa, hs)
    if fault.lubrication > 0.01:
        hs = _pink(n, rng)
        bb, aa = butter(4, [min(2000, 0.5*fs/2)/(fs/2),
                            min(5800, 0.93*fs/2)/(fs/2)], btype='band')
        x += A*2.8*fault.lubrication*filtfilt(bb, aa, hs)

    # ---- noise floor, then sensor pathologies LAST (they act on the signal
    #      the sensor sees, not on the machine)
    x += spec.noise_g*(0.6*rng.normal(0, 1, n)+0.4*_pink(n, rng))
    if fault.sensor_clip > 0.005:
        lim = np.percentile(np.abs(x), 100-25*fault.sensor_clip)
        x = np.clip(x, -lim, lim)
    if fault.sensor_dropout > 0.005:
        n_drops = rng.poisson(6*fault.sensor_dropout)
        for _ in range(n_drops):
            L = int(fs*rng.uniform(0.004, 0.02))
            i0 = rng.integers(0, max(1, n-L))
            x[i0:i0+L] = 0.0
    return x.astype(np.float32)

# ---------------------------------------------------------------- catalogue
def machine_catalog2():
    """v1's five machines plus four that stress the new physics."""
    from .synth import machine_catalog
    cat = {}
    for name, m in machine_catalog().items():
        cat[name] = MachineSpec2(**{k: getattr(m, k) for k in
                                    ('name', 'asset_class', 'kin', 'base_rms_g',
                                     'resonances_hz', 'resonance_q', 'noise_g', 'fs')})
    cat['belt_fan'] = MachineSpec2('belt_fan', 'fan',
        Kinematics(bearings=[(CATALOG['SKF6203'], 1.0, 'DE'),
                             (CATALOG['SKF6203'], 2.36, 'driven')], n_vanes=9),
        base_rms_g=0.065, resonances_hz=(1500., 2600., 4100.),
        belt_ratio=0.4085)
    cat['vfd_pump'] = MachineSpec2('vfd_pump', 'pump',
        Kinematics(bearings=[(CATALOG['SKF6205'], 1.0, 'DE'),
                             (CATALOG['SKF6203'], 1.0, 'NDE')], n_vanes=5),
        base_rms_g=0.050, resonances_hz=(2800., 3700., 5300.),
        vfd_line_hz=50.0, slip_nominal=0.018)
    cat['recip_compressor'] = MachineSpec2('recip_compressor', 'compressor',
        Kinematics(bearings=[(CATALOG['NU2216'], 1.0, 'DE'),
                             (CATALOG['NU2216'], 1.0, 'NDE')]),
        base_rms_g=0.180, resonances_hz=(900., 2100., 3800.),
        resonance_q=(9., 13., 11.))
    cat['slow_mixer'] = MachineSpec2('slow_mixer', 'mixer',
        Kinematics(bearings=[(CATALOG['SKF22312'], 1.0, 'out'),
                             (CATALOG['NU2216'], 4.8, 'in')],
                   gear_teeth=(17, 82), n_shafts=2, shaft_ratios=(1.0, 4.8)),
        base_rms_g=0.075, resonances_hz=(1100., 2300., 3600.))
    return cat
