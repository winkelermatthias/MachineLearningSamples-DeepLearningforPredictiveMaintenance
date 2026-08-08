"""Synthetic vibration generator.

Builds waveforms from a kinematic model rather than sampling a library, so the
ground truth is exact and every fault mode can be dialled continuously from
absent to severe. That is what makes it useful for evaluating a detector: a
recorded dataset gives you a handful of severities, this gives you a curve.

Signal model, summed in the time domain:
    shaft harmonics          deterministic, phase-locked to theta
    bearing tones            impulse trains, non-synchronous, ringing a
                             structural resonance, amplitude modulated
    gear mesh                carrier at N_teeth x shaft, sidebands at 1x
    vane / blade passing     deterministic
    structural resonances    excited by impacts, decaying
    noise floor              pink + white, sensor + flow
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Bearing:
    code: str; bpfo: float; bpfi: float; bsf: float; ftf: float; n_rollers: int

CATALOG = {
    'SKF6205': Bearing('SKF6205', 3.5848, 5.4152, 4.7135, 0.3983, 9),
    'SKF6203': Bearing('SKF6203', 3.0480, 4.9520, 3.9874, 0.3810, 8),
    'NU2216':  Bearing('NU2216',  6.7241, 9.2759, 4.1234, 0.4203, 16),
    'SKF22312':Bearing('SKF22312',7.2340, 9.7660, 3.8710, 0.4256, 17),
}

@dataclass
class Kinematics:
    bearings: list          # (Bearing, shaft_ratio, position)
    n_vanes: int = 0
    gear_teeth: tuple = ()  # (drive, driven) or ()
    n_shafts: int = 1
    shaft_ratios: tuple = (1.0,)

@dataclass
class FaultState:
    """Severity 0..1 per mode. Continuous, so onset can be ramped."""
    outer_race: float = 0.0
    inner_race: float = 0.0
    rolling_element: float = 0.0
    cage: float = 0.0
    imbalance: float = 0.0
    misalignment: float = 0.0
    looseness: float = 0.0
    gear_wear: float = 0.0
    cavitation: float = 0.0
    lubrication: float = 0.0     # raises the floor and the haystack, few discretes

    def active(self):
        return {k: v for k, v in self.__dict__.items() if v > 0.005}

@dataclass
class MachineSpec:
    name: str
    asset_class: str
    kin: Kinematics
    base_rms_g: float = 0.055
    resonances_hz: tuple = (2900., 3600., 5200.)
    resonance_q: tuple = (18., 24., 14.)
    noise_g: float = 0.0035
    fs: float = 12000.0

# ---------------------------------------------------------------- primitives
def _impulse_train(n, fs, freq, jitter, rng):
    """Non-synchronous impulse train. The random slip is what makes bearing
    tones non-synchronous and is why time synchronous averaging removes shaft
    content but leaves these behind."""
    out = np.zeros(n)
    if freq <= 0: return out
    period = fs/freq
    t = rng.uniform(0, period)
    while t < n-1:
        i = int(t); frac = t-i
        out[i] += (1-frac); out[i+1] += frac
        t += period*(1.0 + jitter*rng.normal())
    return out

def _ring(n, fs, f0, q, rng):
    """Impulse response of a lightly damped structural mode."""
    L = int(min(n, fs*8.0/ f0 * q/10.0))
    tt = np.arange(L)/fs
    return np.exp(-np.pi*f0*tt/q)*np.sin(2*np.pi*f0*tt)

def _modulate(x, fs, f_mod, depth):
    if f_mod <= 0 or depth <= 0: return x
    t = np.arange(len(x))/fs
    return x*(1.0 + depth*np.sin(2*np.pi*f_mod*t))

def _pink(n, rng):
    w = rng.normal(0,1,n)
    b = np.array([0.049922, -0.095993, 0.050612, -0.004408])
    a = np.array([1.0, -2.494956, 2.017265, -0.522189])
    from scipy.signal import lfilter
    p = lfilter(b, a, w)
    return p/(p.std()+1e-12)

# ------------------------------------------------------------------ main gen
def generate(spec: MachineSpec, fault: FaultState, seconds: float,
             fr_hz: float, load: float = 1.0, speed_wander_pct: float = 0.3,
             running: bool = True, rng=None) -> np.ndarray:
    """One acquisition of acceleration in g."""
    rng = rng or np.random.default_rng()
    fs = spec.fs; n = int(seconds*fs)
    t = np.arange(n)/fs

    if not running:
        # stopped: sensor noise floor plus faint coupling from nearby plant
        x = spec.noise_g*(0.55*rng.normal(0,1,n) + 0.45*_pink(n,rng))
        x += 0.03*spec.base_rms_g*np.sin(2*np.pi*49.8*t + rng.uniform(0,6.28))
        return x.astype(np.float32)

    # ---- shaft angle with slow wander; everything deterministic rides this
    w = _pink(n, rng)
    from scipy.signal import butter, filtfilt
    b, a = butter(2, 0.08/(fs/2)); w = filtfilt(b, a, w); w /= (w.std()+1e-12)
    fr_t = fr_hz*(1.0 + speed_wander_pct/100.0*w)
    theta = np.cumsum(fr_t)/fs                    # revolutions

    x = np.zeros(n)
    A = spec.base_rms_g*(0.6 + 0.4*load)

    # ---- shaft harmonics. Imbalance loads 1x, misalignment 2x, looseness 3x+
    h_amp = {1: 0.35 + 1.9*fault.imbalance,
             2: 0.16 + 2.4*fault.misalignment,
             3: 0.08 + 1.7*fault.looseness,
             4: 0.05 + 0.9*fault.looseness,
             5: 0.03, 6: 0.02, 7: 0.015, 8: 0.012}
    for k, amp in h_amp.items():
        x += A*amp*np.sin(2*np.pi*k*theta + rng.uniform(0, 6.28))
    if fault.looseness > 0.2:                    # half-order content
        for k in (0.5, 1.5, 2.5):
            x += A*0.5*fault.looseness*np.sin(2*np.pi*k*theta + rng.uniform(0,6.28))

    # ---- bearing tones
    modes = [('outer_race','bpfo', fault.outer_race, 'ftf', 0.25, 0.008),
             ('inner_race','bpfi', fault.inner_race, None,  0.55, 0.010),
             ('rolling_element','bsf', fault.rolling_element, 'ftf', 0.45, 0.015),
             ('cage','ftf', fault.cage, None, 0.20, 0.020)]
    for bearing, ratio, _pos in spec.kin.bearings:
        for mode, attr, sev, modattr, depth, jit in modes:
            if sev <= 0.005: continue
            f_def = getattr(bearing, attr)*fr_hz*ratio
            train = _impulse_train(n, fs, f_def, jit, rng)
            # each impact rings a structural mode
            ri = rng.integers(0, len(spec.resonances_hz))
            ring = _ring(n, fs, spec.resonances_hz[ri], spec.resonance_q[ri], rng)
            sig = np.convolve(train, ring)[:n]
            # inner race modulates at 1x as the defect passes the load zone;
            # rolling element and outer race modulate at cage rate
            f_mod = fr_hz*ratio if modattr is None else getattr(bearing, modattr)*fr_hz*ratio
            sig = _modulate(sig, fs, f_mod, depth*min(1.0, sev*1.6))
            sig /= (sig.std()+1e-12)
            x += A*(0.9*sev**0.75)*sig

    # ---- gear mesh with 1x sidebands
    if spec.kin.gear_teeth:
        nt = spec.kin.gear_teeth[0]
        gm = A*(0.30 + 1.4*fault.gear_wear)*np.sin(2*np.pi*nt*theta + rng.uniform(0,6.28))
        gm = gm*(1.0 + (0.15+0.9*fault.gear_wear)*np.sin(2*np.pi*theta))
        x += gm
        x += A*(0.10+0.7*fault.gear_wear)*np.sin(2*np.pi*2*nt*theta + rng.uniform(0,6.28))

    # ---- vane / blade passing, and cavitation as broadband above it
    if spec.kin.n_vanes:
        nv = spec.kin.n_vanes
        x += A*(0.22+0.5*fault.cavitation)*np.sin(2*np.pi*nv*theta + rng.uniform(0,6.28))
        if fault.cavitation > 0.01:
            hs = _pink(n, rng)
            bb, aa = butter(4, [1200/(fs/2), 5500/(fs/2)], btype='band')
            x += A*2.2*fault.cavitation*filtfilt(bb, aa, hs)

    # ---- lubrication distress: floor and haystack rise, few discrete lines
    if fault.lubrication > 0.01:
        hs = _pink(n, rng)
        bb, aa = butter(4, [2000/(fs/2), 5800/(fs/2)], btype='band')
        x += A*2.8*fault.lubrication*filtfilt(bb, aa, hs)

    # ---- noise floor
    x += spec.noise_g*(0.6*rng.normal(0,1,n) + 0.4*_pink(n,rng))
    return x.astype(np.float32)

# ---------------------------------------------------------------- catalogue
def machine_catalog():
    return {
     'centrifugal_pump': MachineSpec('centrifugal_pump','pump',
        Kinematics(bearings=[(CATALOG['SKF6205'],1.0,'DE'),(CATALOG['SKF6203'],1.0,'NDE')],
                   n_vanes=7), base_rms_g=0.055),
     'induction_motor': MachineSpec('induction_motor','motor',
        Kinematics(bearings=[(CATALOG['SKF6205'],1.0,'DE'),(CATALOG['SKF6205'],1.0,'NDE')]),
        base_rms_g=0.040, resonances_hz=(3100.,4200.,5600.)),
     'gearbox': MachineSpec('gearbox','gearbox',
        Kinematics(bearings=[(CATALOG['NU2216'],1.0,'input'),(CATALOG['SKF22312'],0.28,'output')],
                   gear_teeth=(23,82), n_shafts=2, shaft_ratios=(1.0,0.28)),
        base_rms_g=0.090, resonances_hz=(2400.,3900.,5100.)),
     'centrifugal_fan': MachineSpec('centrifugal_fan','fan',
        Kinematics(bearings=[(CATALOG['SKF22312'],1.0,'DE'),(CATALOG['SKF22312'],1.0,'NDE')],
                   n_vanes=12), base_rms_g=0.070, resonances_hz=(1800.,2700.,4400.)),
     'screw_compressor': MachineSpec('screw_compressor','compressor',
        Kinematics(bearings=[(CATALOG['NU2216'],1.0,'DE'),(CATALOG['NU2216'],1.0,'NDE')],
                   n_vanes=5), base_rms_g=0.120, resonances_hz=(3300.,4800.,6200.)),
    }

# ------------------------------------------------------------- degradation
def progression(kind: str, u: float) -> float:
    """Severity as a function of normalised time-since-onset u in [0,1].
    Real degradation is not linear: most modes are quiet for a long time then
    accelerate, which is exactly what makes early detection hard and worth
    testing against."""
    u = float(np.clip(u, 0, 1))
    if kind == 'exponential':  return u**2.6
    if kind == 'linear':       return u
    if kind == 'step':         return 1.0 if u > 0.5 else 0.0
    if kind == 'sigmoid':      return 1/(1+np.exp(-11*(u-0.55)))
    if kind == 'intermittent': return u**2 * (0.35 + 0.65*(np.sin(u*17) > 0))
    return u
