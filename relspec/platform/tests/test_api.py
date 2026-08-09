import base64
import numpy as np
from conftest import make_wave, b64f32, make_ws, make_tree, hdr, PHRASE

def test_workspace_lifecycle(client):
    r = client.post('/v1/workspaces', json=dict(passphrase='weak'))
    assert r.status_code == 400
    ws = make_ws(client)
    assert ws['created'] is True
    ws2 = make_ws(client)                       # same phrase reconnects
    assert ws2['created'] is False
    assert ws2['workspace_id'] == ws['workspace_id']
    # bearer works
    r = client.get('/v1/workspaces/me', headers=hdr(ws['token']))
    assert r.status_code == 200 and r.json()['sensors'] == 0
    # passphrase header works
    p = base64.b64encode(PHRASE.encode()).decode()
    r = client.get('/v1/workspaces/me',
                   headers={'Authorization': f'Passphrase {p}'})
    assert r.status_code == 200
    # wrong phrase rejected
    bad = base64.b64encode(b'utterly wrong phrase of words').decode()
    r = client.get('/v1/workspaces/me',
                   headers={'Authorization': f'Passphrase {bad}'})
    assert r.status_code == 401
    # no header
    assert client.get('/v1/workspaces/me').status_code == 401

def test_hierarchy_idempotent(client):
    ws = make_ws(client)
    s1 = make_tree(client, ws['token'])
    s2 = make_tree(client, ws['token'])
    assert s1 == s2
    r = client.get('/v1/hierarchy', headers=hdr(ws['token']))
    tree = r.json()
    assert tree['plants'][0]['assets'][0]['components'][0]['sensors'][0]['sensor_id'] == s1

def _submit(client, tok, sid, x, day, **kw):
    body = dict(sensor_id=sid, ts=f'2026-01-{day:02d}T12:00:00Z', fs=12000,
                units='g', encoding='float32', data_b64=b64f32(x), **kw)
    return client.post('/v1/waveforms', json=body, headers=hdr(tok))

def test_ingest_policy_and_reads(client):
    ws = make_ws(client)
    tok = ws['token']
    sid = make_tree(client, tok)

    r = _submit(client, tok, sid, make_wave(seed=1), 1, client_ref='w1')
    v = r.json()
    assert r.status_code == 200, r.text
    assert v['waveform_stored'] and v['sig_reason'] == 'first_waveform'
    assert v['frame_kind'] == 'A' and v['payload_bytes'] > 200
    acq1 = v['acq_id']

    # day 2: chain continues (codec state persisted -> residual, not anchor)
    v2 = _submit(client, tok, sid, make_wave(seed=2), 2, client_ref='w2').json()
    assert v2['frame_kind'] in ('R', 'P')
    assert not v2['waveform_stored'] and v2['sig_reason'] in (
        'weekly_budget_exhausted', 'not_significant')
    assert v2['payload_bytes'] < v['payload_bytes']

    # duplicate client_ref returns original verdict
    d = _submit(client, tok, sid, make_wave(seed=2), 2, client_ref='w2').json()
    assert d['duplicate'] is True and d['acq_id'] == v2['acq_id']

    # out-of-order rejected
    r = _submit(client, tok, sid, make_wave(seed=3), 1)
    assert r.status_code == 409

    # day 9: weekly baseline stores again
    v9 = _submit(client, tok, sid, make_wave(seed=9), 9).json()
    assert v9['waveform_stored'] and v9['sig_reason'] == 'weekly_baseline'

    # units conversion accepts m/s2
    x = make_wave(seed=10)*9.80665
    body = dict(sensor_id=sid, ts='2026-01-10T12:00:00Z', fs=12000,
                units='m/s2', encoding='float32', data_b64=b64f32(x))
    v10 = client.post('/v1/waveforms', json=body, headers=hdr(tok)).json()
    assert abs(v10['features']['acc_rms']-v9['features']['acc_rms']) < 0.05

    # validation errors
    assert _submit(client, tok, sid, make_wave(seed=1)[:100], 11).status_code == 400
    bad = dict(sensor_id=sid, ts='2026-01-12T12:00:00Z', fs=500, units='g',
               encoding='float32', data_b64=b64f32(make_wave(seed=1)))
    assert client.post('/v1/waveforms', json=bad, headers=hdr(tok)).status_code == 400

    # ---- reads
    r = client.get('/v1/fleet/overview', headers=hdr(tok))
    ov = r.json()
    assert len(ov) == 1 and ov[0]['sensor_id'] == sid
    assert r.headers.get('etag')

    # ETag round trip -> 304
    r2 = client.get('/v1/fleet/overview', headers={**hdr(tok),
                    'If-None-Match': r.headers['etag']})
    assert r2.status_code == 304

    tr = client.get(f'/v1/sensors/{sid}/trend', headers=hdr(tok)).json()
    assert len(tr) == 4 and all(row['vel_avg'] > 0 for row in tr)

    fr = client.get(f'/v1/sensors/{sid}/frames', headers=hdr(tok)).json()
    assert len(fr) == 4 and fr[0]['ts'] > fr[-1]['ts']

    sp = client.get(f'/v1/acquisitions/{acq1}/spectra', headers=hdr(tok))
    s = sp.json()
    assert len(base64.b64decode(s['acc']['o'])) == 768
    assert len(base64.b64decode(s['env']['dec'])) == 384
    assert 'immutable' in sp.headers['cache-control']
    assert s['acc']['kind'] == 'anchor'
    # anchors are lossless
    assert s['acc']['o'] == s['acc']['dec']

    b = client.get(f'/v1/sensors/{sid}/explorer-bundle', headers=hdr(tok)).json()
    assert len(b['frames']) == 4
    assert b['frames'][0]['ts'] < b['frames'][-1]['ts']   # oldest first
    assert {'a_o', 'a_d', 'e_o', 'e_d', 'acc_pat', 'acc_own'} <= set(b['frames'][0])
    assert b['title'].startswith('P1 / PUMP-1')

    # waveform round trip: stored raw decodes back to the submitted signal
    import zstandard
    w = client.get(f'/v1/acquisitions/{acq1}/waveform', headers=hdr(tok)).json()
    raw = zstandard.ZstdDecompressor().decompress(base64.b64decode(w['data_b64']))
    got = np.frombuffer(raw, dtype='<i2').astype(np.float64)*w['scale']
    orig = make_wave(seed=1)
    assert len(got) == len(orig)
    assert float(np.max(np.abs(got-orig))) < w['scale']*1.01
    # unstored acquisition has no waveform
    assert client.get(f'/v1/acquisitions/{v2["acq_id"]}/waveform',
                      headers=hdr(tok)).status_code == 404

def test_batch_and_isolation(client):
    ws = make_ws(client)
    tok = ws['token']
    sid = make_tree(client, tok)
    import json as J
    lines = []
    for day in (1, 2):
        lines.append(J.dumps(dict(sensor_id=sid, ts=f'2026-02-0{day}T00:00:00Z',
                                  fs=12000, units='g', encoding='float32',
                                  data_b64=b64f32(make_wave(seed=day)))))
    lines.append('{"sensor_id": "nope", "ts": "2026-02-03T00:00:00Z"}')
    r = client.post('/v1/waveforms:batch', content='\n'.join(lines),
                    headers=hdr(tok))
    out = [J.loads(l) for l in r.text.splitlines()]
    assert len(out) == 3
    assert out[0]['sig_reason'] == 'first_waveform'
    assert 'error' in out[2]

    # a second workspace cannot see the first one's data
    ws2 = make_ws(client, phrase='a completely different passphrase here')
    r = client.get('/v1/fleet/overview', headers=hdr(ws2['token']))
    assert r.json() == []
    r = client.get(f'/v1/sensors/{sid}/trend', headers=hdr(ws2['token']))
    assert r.json() == []
