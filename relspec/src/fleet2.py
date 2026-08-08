"""Fleet campaign v2: 48 assets, 90 days, 2 acquisitions a day.

Three times the assets of v1's fleet and nine machine classes instead of
five, including the deliberately hostile ones: VFD drives whose speed
setpoint jumps, belt drives with sub-1x combs, a reciprocating compressor
whose spectrum is all low-order teeth, and two assets with failing sensors.

The question the campaign answers is the compression question at fleet
scale: what does a real mixture of healthy drift, fault onset, speed
changes and sensor garbage cost per day per sensor, for codec v1 vs v2?
And the guardrail: the v1 gate, fed the same extraction, must still detect
the seeded faults - a codec that wins bytes by starving detection is not a
codec, it is a bug with a good ratio.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__) or '.')
import numpy as np
from relspec.pipeline import Gate, NBINS, ENV_BINS
from relspec.pipeline2 import extract2, _band_cache
from relspec.synth2 import (FaultState2, machine_catalog2, generate2,
                            progression)
from eval_real import Rail

DAYS = 90
ACQ_PER_DAY = 2
N_ASSETS = 48
WIN_S = 2.0

NOMINAL_FR = dict(centrifugal_pump=29.5, induction_motor=24.83, gearbox=16.2,
                  centrifugal_fan=12.4, screw_compressor=48.6, belt_fan=18.7,
                  vfd_pump=33.0, recip_compressor=12.15, slow_mixer=6.4)

# fault menu per machine: (field, weight)
FAULT_MENU = dict(
    centrifugal_pump=[('outer_race', 3), ('inner_race', 2), ('imbalance', 2),
                      ('cavitation', 3), ('lubrication', 1)],
    induction_motor=[('outer_race', 3), ('inner_race', 3), ('imbalance', 2),
                     ('misalignment', 2)],
    gearbox=[('gear_wear', 4), ('outer_race', 2), ('inner_race', 2),
             ('looseness', 1)],
    centrifugal_fan=[('imbalance', 4), ('outer_race', 2), ('looseness', 2)],
    screw_compressor=[('rolling_element', 2), ('outer_race', 2),
                      ('lubrication', 2)],
    belt_fan=[('belt_wear', 4), ('imbalance', 2), ('outer_race', 2)],
    vfd_pump=[('rotor_bar', 3), ('outer_race', 2), ('cavitation', 2)],
    recip_compressor=[('looseness', 3), ('bent_shaft', 2), ('outer_race', 2)],
    slow_mixer=[('gear_wear', 3), ('inner_race', 2), ('cage', 1)],
)
PROG_KINDS = ['exponential', 'linear', 'sigmoid', 'intermittent']

def build_assets(seed=11):
    rng = np.random.default_rng(seed)
    cat = machine_catalog2()
    names = list(cat)
    assets = []
    for i in range(N_ASSETS):
        mtype = names[i % len(names)]
        spec = cat[mtype]
        fr = NOMINAL_FR[mtype]*(1+rng.uniform(-0.03, 0.03))
        a = dict(aid=f'A{i:02d}', mtype=mtype, fr=fr,
                 wander=float(rng.uniform(0.2, 1.4)),
                 vfd=spec.vfd_line_hz > 0,
                 seed=int(rng.integers(1, 2**31)))
        # ~40% get a progressing fault; two assets get a failing sensor
        if i in (17, 41):
            a['fault'] = ('sensor_clip' if i == 17 else 'sensor_dropout')
            a['onset'] = int(rng.uniform(30, 55)); a['prog'] = 'linear'
            a['sev_max'] = float(rng.uniform(0.6, 1.0))
        elif rng.random() < 0.40:
            menu = FAULT_MENU[mtype]
            w = np.array([m[1] for m in menu], dtype=float)
            a['fault'] = menu[rng.choice(len(menu), p=w/w.sum())][0]
            a['onset'] = int(rng.uniform(25, 70))
            a['prog'] = PROG_KINDS[rng.integers(0, len(PROG_KINDS))]
            a['sev_max'] = float(rng.uniform(0.5, 1.0))
        else:
            a['fault'] = None
        assets.append(a)
    return assets

def run_asset(a):
    cat = machine_catalog2()
    spec = cat[a['mtype']]
    rng = np.random.default_rng(a['seed'])
    rails = {k: Rail(k) for k in ('v1', 'v2anchor', 'v2best')}
    gate = Gate()
    rows = []
    detect_day = None
    setpoint = a['fr']
    for day in range(DAYS):
        if a['vfd'] and day % 10 == 3:
            setpoint = a['fr']*(1+rng.uniform(-0.12, 0.12))
        load = 0.6+0.35*np.sin(2*np.pi*day/7.0)**2+rng.uniform(-0.05, 0.05)
        for k in range(ACQ_PER_DAY):
            t_acq = (day*ACQ_PER_DAY+k)
            fr_true = setpoint*(1+rng.uniform(-0.015, 0.015))
            f = FaultState2()
            if a['fault'] and day >= a['onset']:
                u = (day-a['onset'])/max(DAYS-a['onset'], 1)
                setattr(f, a['fault'],
                        a['sev_max']*progression(a['prog'], u))
            profile = 'ramp' if rng.random() < 0.05 else 'steady'
            x = generate2(spec, f, WIN_S, fr_true, load=np.clip(load, 0.3, 1.0),
                          speed_wander_pct=a['wander'], profile=profile,
                          rng=rng)
            e = extract2(x.astype(np.float64), spec.fs, band_key=a['aid'],
                         fr_nominal=a['fr'])
            row = dict(aid=a['aid'], mtype=a['mtype'], day=day, k=k,
                       faulty=bool(a['fault']) and day >= a['onset'],
                       fr_true=fr_true, fr_est=e.fr, tier=e.tier)
            for name, rail in rails.items():
                b, kind, _, _ = rail.step(e)
                row[name] = b
                row[name+'_anchor'] = int('anchor' in kind)
            if gate.n == 0: gate.init(e)
            else:
                s = gate.decide(e, t_acq*43200)
                if s['decision'] == 'up_change' and detect_day is None:
                    detect_day = day
            rows.append(row)
    return dict(aid=a['aid'], mtype=a['mtype'], fault=a['fault'],
                onset=a.get('onset'), detect_day=detect_day, rows=rows)

def main():
    from multiprocessing import Pool
    assets = build_assets()
    t0 = time.time()
    with Pool(4) as p:
        results = p.map(run_asset, assets)
    rows = [r for res in results for r in res['rows']]
    meta = [{k: v for k, v in res.items() if k != 'rows'} for res in results]
    print(f'{len(rows)} acquisitions from {len(assets)} assets '
          f'in {time.time()-t0:.0f}s')
    json.dump(dict(meta=meta, rows=rows),
              open('../fleet2.json', 'w'), default=float)

    import pandas as pd
    F = pd.DataFrame(rows); M = pd.DataFrame(meta)
    print('\n== bytes/frame and bytes/day/sensor ==')
    for k in ('v1', 'v2anchor', 'v2best'):
        bpf = F[k].mean()
        print(f'  {k:<9} {bpf:7.1f} B/frame  {bpf*ACQ_PER_DAY:7.0f} B/day  '
              f'anchor rate {F[k+"_anchor"].mean()*100:.1f}%')
    print('\n== by machine type (v2best B/frame) ==')
    print(F.groupby("mtype")[["v1", "v2best"]].mean().round(1).to_string())
    print('\n== speed estimator ==')
    err = 100*np.abs(F.fr_est-F.fr_true)/F.fr_true
    print(f'  median {err.median():.3f}%  p95 {err.quantile(0.95):.3f}%  '
          f'tier1 {100*(F.tier == 1).mean():.1f}%')
    print('\n== detection sanity (v1 gate on v2 extraction) ==')
    faulted = M[M.fault.notna() & ~M.fault.str.startswith('sensor').fillna(False)]
    det = faulted[faulted.detect_day.notna()]
    print(f'  {len(det)}/{len(faulted)} seeded faults detected; '
          f'median latency {(det.detect_day-det.onset).median():.1f} d')
    healthy = M[M.fault.isna()]
    fa = healthy[healthy.detect_day.notna()]
    print(f'  false alarms on healthy assets: {len(fa)}/{len(healthy)}')

if __name__ == '__main__':
    main()
