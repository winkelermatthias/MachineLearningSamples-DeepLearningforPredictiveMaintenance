"""Seed a demo workspace through the public API: two sensors on one pump,
30 days at one spectrum a day, an outer-race fault growing on the drive end
from day 22. Usage:

    python3 scripts/seed_demo.py http://127.0.0.1:8000 "demo passphrase ..."
"""
import base64, sys, pathlib
import numpy as np
import httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'src'))
from relspec.synth2 import FaultState2, machine_catalog2, generate2  # noqa: E402

def b64f32(x):
    return base64.b64encode(np.asarray(x, dtype='<f4').tobytes()).decode()

def main(base, phrase):
    c = httpx.Client(base_url=base, timeout=120)
    ws = c.post('/v1/workspaces',
                json=dict(passphrase=phrase, name='demo',
                          params=dict(event_budget_per_week=2))).json()
    tok = {'Authorization': 'Bearer '+ws['token']}
    tree = dict(plants=[dict(name='Demo plant', assets=[dict(
        tag='PUMP-201', machine_type='pump', components=[
            dict(name='drive-end bearing', kind='bearing',
                 sensors=[dict(code='DE-V', units='g', fs=12000,
                               fr_nominal=29.5)]),
            dict(name='non-drive-end bearing', kind='bearing',
                 sensors=[dict(code='NDE-V', units='g', fs=12000,
                               fr_nominal=29.5)])])])])
    r = c.post('/v1/hierarchy:ensure', json=tree, headers=tok).json()
    comps = r['plants'][0]['assets'][0]['components']
    sensors = {cc['name']: cc['sensors'][0]['sensor_id'] for cc in comps}
    spec = machine_catalog2()['centrifugal_pump']
    for name, sid in sensors.items():
        faulty = 'drive-end' in name and 'non' not in name
        for day in range(1, 31):
            sev = 0.0
            if faulty and day >= 22: sev = min(0.9, 0.15*(day-21))
            rng = np.random.default_rng(hash((name, day)) % 2**31)
            x = generate2(spec, FaultState2(outer_race=sev), 2.0,
                          29.5*(1+rng.uniform(-.01, .01)),
                          load=0.7+0.2*np.sin(day/3), speed_wander_pct=0.5,
                          rng=rng).astype('float64')
            v = c.post('/v1/waveforms', json=dict(
                sensor_id=sid, ts=f'2026-03-{day:02d}T06:00:00Z', fs=12000,
                units='g', encoding='float32', data_b64=b64f32(x),
                client_ref=f'{name}-{day}'), headers=tok).json()
            if 'acq_id' not in v:
                raise SystemExit(f'ingest failed day {day}: {v}')
        print(f'{name}: 30 days ingested')
    ov = c.get('/v1/fleet/overview', headers=tok).json()
    for s in ov:
        print(f"  {s['path']}: vel {s['vel_rms']:.2f} mm/s, "
              f"gate {s['gate_score']:.1f} ({s['gate_decision']})")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
