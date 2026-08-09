"""A 30-day campaign through the HTTP API: healthy warm-up, then a growing
outer-race fault. Verifies the gate fires a change event, the significance
policy spends its event budget on it, and every read surface tells the
same story."""
import numpy as np
from relspec.synth2 import FaultState2
from conftest import make_wave, b64f32, make_ws, make_tree, hdr

def test_thirty_day_campaign(client):
    ws = make_ws(client, params=dict(event_budget_per_week=2))
    tok = ws['token']
    sid = make_tree(client, tok)

    stored = []
    for day in range(1, 31):
        sev = 0.0 if day < 22 else min(0.9, 0.15*(day-21))
        fault = FaultState2(outer_race=sev) if sev else None
        x = make_wave(fault=fault, seed=100+day)
        body = dict(sensor_id=sid, ts=f'2026-03-{day:02d}T06:00:00Z',
                    fs=12000, units='g', encoding='float32',
                    data_b64=b64f32(x), client_ref=f'd{day}')
        v = client.post('/v1/waveforms', json=body, headers=hdr(tok)).json()
        assert 'acq_id' in v, v
        if v['waveform_stored']: stored.append((day, v['sig_reason']))

    # weekly baselines happened
    reasons = [r for _, r in stored]
    assert reasons[0] == 'first_waveform'
    assert 'weekly_baseline' in reasons
    # the fault fired the gate and the event budget kept the raw waveform
    ev = client.get('/v1/events', headers=hdr(tok)).json()
    changes = [e for e in ev if e['type'] == 'gate_change']
    assert changes, 'no gate change on a 0->0.9 outer race fault'
    assert 'gate_change' in reasons
    # never more than the weekly cap (1 baseline + 2 events)
    days = [d for d, _ in stored]
    for i in range(len(days)):
        assert sum(1 for d in days if 0 <= days[i]-d < 7) <= 3

    # trend shows the level rise
    tr = client.get(f'/v1/sensors/{sid}/trend', headers=hdr(tok)).json()
    assert len(tr) == 30
    early = np.mean([t['acc_avg'] for t in tr[:10]])
    late = np.mean([t['acc_avg'] for t in tr[-3:]])
    assert late > early*1.5, f'fault not visible in trend ({early} -> {late})'

    # overview reflects the last (faulty) state
    ov = client.get('/v1/fleet/overview', headers=hdr(tok)).json()[0]
    assert ov['gate_score'] > 2.0

    # bundle serves the whole month, oldest first, with patterns
    b = client.get(f'/v1/sensors/{sid}/explorer-bundle?limit=60',
                   headers=hdr(tok)).json()
    assert len(b['frames']) == 30
    n_pats_late = len(b['frames'][-1]['env_pat'])
    assert n_pats_late >= 1
    # payload economics: residual frames dominate
    kinds = [f['kind'] for f in b['frames']]
    assert kinds.count('A') <= 6
    mean_bytes = np.mean([f['bytes'] for f in b['frames']])
    assert mean_bytes < 700
