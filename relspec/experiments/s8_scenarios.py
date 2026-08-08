"""s8_scenarios.py - build 90-day scenarios and drive the compiled firmware.

Running frames are real CWRU acceleration at random offsets within the source
record. Idle frames are synthesized: accelerometer noise floor plus faint
coupling from nearby machinery, since CWRU contains no stopped-machine data.
Idle synthesis is labelled everywhere it appears in the results.
"""
import numpy as np, scipy.io as sio, struct, subprocess, os, json

FS = 12000.0
NSAMP = 12000              # 1.0 s per acquisition
WAKE_S = 6 * 3600          # 4 acquisitions/day
DAYS = 90
NACQ = DAYS * 24 * 3600 // WAKE_S

POOL = {'normal': ['97','98','99','100'],
        'IR_007': ['105','106','107','108'],
        'IR_014': ['169','170','171','172'],
        'IR_021': ['209','210','211','212']}
LOADIDX = {'97':0,'98':1,'99':2,'100':3,'105':0,'106':1,'107':2,'108':3,
           '169':0,'170':1,'171':2,'172':3,'209':0,'210':1,'211':2,'212':3}

_cache = {}
def rec(fid):
    if fid not in _cache:
        m = sio.loadmat(f'data/{fid}.mat')
        _cache[fid] = m[[k for k in m if k.endswith('_DE_time')][0]].ravel().astype(np.float32)
    return _cache[fid]

def running_frame(state, load, rng):
    fid = [f for f in POOL[state] if LOADIDX[f] == load][0]
    x = rec(fid)
    s = rng.integers(0, len(x) - NSAMP)
    return x[s:s+NSAMP].copy()

def idle_frame(rng, coupling_db=-30.0):
    """Sensor noise floor + faint structural coupling from nearby plant."""
    n = rng.normal(0, 1, NSAMP).astype(np.float32)
    b, a = 0.92, 1.0
    pink = np.zeros(NSAMP, dtype=np.float32); acc = 0.0
    for i in range(NSAMP):
        acc = b*acc + n[i]*0.4; pink[i] = acc
    noise = (0.55*n + 0.45*pink/ (pink.std()+1e-9))
    noise *= 0.0040 / (noise.std()+1e-9)                 # ~ -48 dB re 1 g
    x = rec('97'); s = rng.integers(0, len(x)-NSAMP)
    coup = x[s:s+NSAMP] * (10**(coupling_db/20.0))
    return (noise + coup).astype(np.float32)

# ---------------- duty-cycle generators ----------------
def duty_always(k, rng):    return True
def duty_12h(k, rng):       return (k % 4) in (1, 2)          # 12 h/day
def duty_irregular(k, rng, st={}):
    if 'blk' not in st:
        st['blk'] = rng.random(NACQ) < 0.55
        for i in range(1, NACQ):                              # smooth to blocks
            if rng.random() < 0.75: st['blk'][i] = st['blk'][i-1]
    return bool(st['blk'][k])

def health_step(day):
    return 'normal' if day < 45 else 'IR_021'
def health_gradual(day):
    if day < 30: return 'normal'
    if day < 50: return 'IR_007'
    if day < 70: return 'IR_014'
    return 'IR_021'
def health_healthy(day): return 'normal'
def health_incipient(day):
    return 'normal' if day < 50 else 'IR_007'

def blend_ramp(day):
    """Incipient degradation: blend a fault record into a healthy one with a
    slowly rising mix fraction. CWRU seeded faults are 12+ dB jumps, far too
    large to test detection sensitivity; blending makes the onset gradual and
    subtle while every sample stays real measured data."""
    if day < 25: return ('normal', 0.0)
    a = min(1.20, (day - 25) / 60.0 * 1.20)
    return ('IR_007', a)

def load_fixed(day): return 1
def load_cycle(day):  return int(day // 7) % 4

SCENARIOS = {
 'S7_incipient_blend':  dict(duty=duty_12h, health=health_healthy, load=load_fixed,
                             blend=blend_ramp),
 'S1_always_healthy':   dict(duty=duty_always,   health=health_healthy,   load=load_fixed),
 'S2_duty_healthy':     dict(duty=duty_12h,      health=health_healthy,   load=load_fixed),
 'S3_duty_step_fault':  dict(duty=duty_12h,      health=health_step,      load=load_fixed),
 'S4_duty_gradual':     dict(duty=duty_12h,      health=health_gradual,   load=load_fixed),
 'S5_duty_loadcycle':   dict(duty=duty_12h,      health=health_healthy,   load=load_cycle),
 'S6_irreg_incipient':  dict(duty=duty_irregular,health=health_incipient, load=load_fixed),
}

def build(name, seed=7):
    sc = SCENARIOS[name]; rng = np.random.default_rng(seed)
    frames, meta, st = [], [], {}
    for k in range(NACQ):
        t = k * WAKE_S; day = t / 86400.0
        try:    run = sc['duty'](k, rng, st) if sc['duty'] is duty_irregular else sc['duty'](k, rng)
        except TypeError: run = sc['duty'](k, rng)
        h = sc['health'](day); ld = sc['load'](day)
        if run:
            x = running_frame(h, ld, rng)
            if sc.get('blend'):
                fh, a = sc['blend'](day)
                if a > 0:
                    xf = running_frame(fh, ld, rng)
                    xf = xf * (x.std() / (xf.std() + 1e-9))   # equal-power ref
                    x = x + a * xf                            # additive
        else:
            x = idle_frame(rng)
        frames.append(x)
        meta.append(dict(k=k, t=t, day=day, running=bool(run), health=h, load=ld))
    return frames, meta

def write_bin(path, frames, meta):
    with open(path,'wb') as f:
        f.write(struct.pack('<IIf', len(frames), NSAMP, FS))
        for x, m in zip(frames, meta):
            f.write(struct.pack('<IB', m['t'], 1 if m['running'] else 0))
            f.write(np.asarray(x, dtype=np.float32).tobytes())

def run_sim(binpath, csvpath, cfgpath=None):
    cmd = ['./fw/sim', binpath, csvpath] + ([cfgpath] if cfgpath else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stderr.strip()

if __name__ == '__main__':
    os.makedirs('scen', exist_ok=True)
    summary = {}
    for name in SCENARIOS:
        frames, meta = build(name)
        bp = f'scen/{name}.bin'; cp = f'scen/{name}.csv'
        write_bin(bp, frames, meta)
        err = run_sim(bp, cp)
        nrun = sum(m['running'] for m in meta)
        print(f"{name:22s} acq={len(meta)} truth_running={nrun} ({100*nrun/len(meta):.0f}%)  {err}")
        summary[name] = dict(nacq=len(meta), nrun=int(nrun), sim=err)
        json.dump(meta, open(f'scen/{name}.meta.json','w'))
    json.dump(summary, open('scen/summary.json','w'), indent=1)
