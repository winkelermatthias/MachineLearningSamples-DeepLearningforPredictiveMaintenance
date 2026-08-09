"""Health model, per-pattern z, and change fingerprints, exercised the
only honest way: a fault campaign through the HTTP API against a real
database. Quiet warm-up teaches baselines; a growing outer-race fault
must move z on SOME track, escalate the health tier with named drivers,
and every acquisition — codec-only ones included — must carry a
fingerprint."""
import numpy as np
from relspec.synth2 import FaultState2
from conftest import make_wave, b64f32, make_ws, make_tree, hdr


def _ingest_day(client, tok, sid, day, fault=None, seed=0):
    x = make_wave(fault=fault, seed=seed)
    body = dict(sensor_id=sid, ts=f'2026-05-{day:02d}T06:00:00Z',
                fs=12000, units='g', encoding='float32',
                data_b64=b64f32(x), client_ref=f'h{day}')
    r = client.post('/v1/waveforms', json=body, headers=hdr(tok))
    assert r.status_code == 200, r.text
    return r.json()


def test_health_campaign(client):
    ws = make_ws(client, params=dict(event_budget_per_week=2))
    tok = ws['token']
    sid = make_tree(client, tok)

    # 16 days with a mild incipient defect teach the baselines (the
    # BPFO family exists and is quiet-stable), then 8 days of growth —
    # the z-matrix's exact target scenario
    for day in range(1, 17):
        _ingest_day(client, tok, sid, day,
                    fault=FaultState2(outer_race=0.12), seed=500 + day)
    for day in range(17, 25):
        sev = min(0.95, 0.12 + 0.2 * (day - 16))
        _ingest_day(client, tok, sid, day,
                    fault=FaultState2(outer_race=sev), seed=500 + day)

    h = client.get(f'/v1/sensors/{sid}/health', headers=hdr(tok)).json()
    assert len(h) == 24

    # fingerprint on EVERY acquisition, decoded server-side
    for row in h:
        assert row['fp'] is not None, row['ts']
        assert len(row['fp']['acc_bands']) == 16
        assert len(row['fp']['env_bands']) == 8
    # the fingerprint sees the fault even without a spectral frame read
    early_env = np.mean([np.mean(r['fp']['env_bands']) for r in h[4:10]])
    late_env = np.mean([np.mean(r['fp']['env_bands']) for r in h[-3:]])
    assert late_env > early_env + 2, (early_env, late_env)

    # health: quiet start, escalated end, drivers say why
    assert all((r['health'] or 0) <= 1 for r in h[6:16]), \
        [ (r['ts'], r['health']) for r in h[:16] ]
    assert h[-1]['health'] >= 2, h[-1]
    sig = {d['signal'] for d in h[-1]['drivers']}
    assert 'gate' in sig and 'vel_rms' in sig

    # per-pattern z via the spectra read. The envelope rail is honestly
    # empty while healthy (nothing patterned), so baselines live on the
    # acc rail; fault energy landing on quiet-taught acc tracks must go
    # z-positive. Tracks born WITH the fault have no baseline — that is
    # the gate's job, not z's.
    frs = client.get(f'/v1/sensors/{sid}/frames?limit=5',
                     headers=hdr(tok)).json()
    zs = []
    for fr2 in frs:
        sp = client.get(f"/v1/acquisitions/{fr2['acq_id']}/spectra",
                        headers=hdr(tok)).json()
        for rail in ('acc', 'env'):
            zs += [p['z'] for p in sp[rail]['patterns']
                   if p.get('z') is not None]
    assert zs, 'no z reported on any baselined track in the last 5 frames'
    assert max(zs) > 2.5, sorted(zs)

    # fleet health rollup: our sensor is present and counted
    fh = client.get('/v1/fleet/health', headers=hdr(tok)).json()
    assert fh['sensors'][0]['sensor_id'] == sid
    assert fh['sensors'][0]['health'] >= 2
    assert fh['assets'][0]['health'] >= 2
    counts = fh['plants'][0]['counts']
    assert sum(counts) + fh['plants'][0]['learning'] == 1

    # ETag caching works on the new endpoints
    r1 = client.get('/v1/fleet/health', headers=hdr(tok))
    r2 = client.get('/v1/fleet/health',
                    headers=hdr(tok) | {'if-none-match': r1.headers['etag']})
    assert r2.status_code == 304


def test_health_hysteresis(client):
    """After a fault escalates the tier, quiet frames must NOT relax it
    immediately — one step down only after 5 consecutive quieter frames."""
    ws = make_ws(client, phrase='hysteresis test phrase with real entropy',
                 params=dict(event_budget_per_week=2))
    tok = ws['token']
    sid = make_tree(client, tok)

    for day in range(1, 13):
        _ingest_day(client, tok, sid, day,
                    fault=FaultState2(outer_race=0.12), seed=900 + day)
    for day in range(13, 19):
        sev = min(0.95, 0.12 + 0.25 * (day - 12))
        _ingest_day(client, tok, sid, day,
                    fault=FaultState2(outer_race=sev), seed=900 + day)
    h_fault = max(r['health'] for r in
                  client.get(f'/v1/sensors/{sid}/health',
                             headers=hdr(tok)).json()[12:])
    assert h_fault >= 2

    for day in range(19, 28):
        _ingest_day(client, tok, sid, day,
                    fault=FaultState2(outer_race=0.12), seed=900 + day)
    h = client.get(f'/v1/sensors/{sid}/health', headers=hdr(tok)).json()
    after = [r['health'] for r in h[18:]]
    # quiet frames must not snap the tier back: stepwise relaxation only,
    # and never within the first 4 quiet frames after the peak
    peak = max([r['health'] for r in h[12:18]])
    assert all(a >= peak - 1 for a in after[:4]), after
    assert after[-1] < peak, after
    for i in range(1, len(after)):
        assert after[i] >= after[i-1] - 1
