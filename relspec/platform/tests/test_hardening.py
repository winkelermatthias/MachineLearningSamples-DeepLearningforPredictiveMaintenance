"""Stage-1 hardening: per-sensor ingest serialization, scoped API keys,
and ingest quotas/bounds."""
import json
import threading
from conftest import make_wave, b64f32, make_ws, make_tree, hdr

def _body(sid, ts, seed=1, **kw):
    return dict(sensor_id=sid, ts=ts, fs=12000, units='g', encoding='float32',
                data_b64=b64f32(make_wave(seed=seed)), **kw)

# ------------------------------------------------------- (a) serialization
def test_concurrent_same_sensor_serialized(client):
    ws = make_ws(client)
    tok = ws['token']
    sid = make_tree(client, tok)
    N = 8
    results = [None]*N
    barrier = threading.Barrier(N)

    def worker(k):
        barrier.wait()
        r = client.post('/v1/waveforms',
                        json=_body(sid, f'2026-05-01T00:{k:02d}:00Z', seed=20+k),
                        headers=hdr(tok))
        results[k] = r

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(N)]
    for t in threads: t.start()
    for t in threads: t.join()

    # serialized: every request either landed or was a clean 409 (a later ts
    # committed first); never a 500 / corrupted chain
    codes = [r.status_code for r in results]
    assert all(c in (200, 409) for c in codes), codes
    ok = codes.count(200)
    assert ok >= 1
    # the 409'd ones retry with fresh, later timestamps and all land
    for j in range(N-ok):
        r = client.post('/v1/waveforms',
                        json=_body(sid, f'2026-05-02T00:{j:02d}:00Z', seed=40+j),
                        headers=hdr(tok))
        assert r.status_code == 200, r.text

    fr = client.get(f'/v1/sensors/{sid}/frames', headers=hdr(tok)).json()
    assert len(fr) == N
    from relspec_service import db
    with db.pool().connection() as conn:
        n_seen, = conn.execute(
            'SELECT n_seen FROM gate_state WHERE sensor_id=%s', (sid,)).fetchone()
        n_ts, = conn.execute(
            'SELECT count(DISTINCT ts) FROM acquisition WHERE sensor_id=%s',
            (sid,)).fetchone()
        anchors, = conn.execute(
            """SELECT count(*) FROM spectrum_frame f
               JOIN acquisition a ON a.acq_id=f.acq_id
               WHERE a.sensor_id=%s AND f.rail='acc' AND f.kind='anchor'""",
            (sid,)).fetchone()
    # n_seen counts every commit exactly once: lost updates would show here
    assert n_seen == N and n_ts == N
    # chain intact: one initial anchor, everything after rides the chain
    assert anchors <= 2
    kinds = [f['kind'] for f in fr]
    assert kinds[-1] == 'A'   # oldest frame (frames come newest-first)

def test_batch_groups_per_sensor(client):
    ws = make_ws(client)
    tok = ws['token']
    tree = dict(plants=[dict(name='P1', assets=[dict(tag='PUMP-1',
                machine_type='pump', components=[dict(name='DE-brg',
                kind='bearing', sensors=[dict(code='V1', fs=12000, fr_nominal=29.5),
                                         dict(code='V2', fs=12000, fr_nominal=29.5)])])])])
    r = client.post('/v1/hierarchy:ensure', json=tree, headers=hdr(tok))
    sensors = r.json()['plants'][0]['assets'][0]['components'][0]['sensors']
    s1, s2 = sensors[0]['sensor_id'], sensors[1]['sensor_id']
    # interleaved lines; within each sensor the order is oldest-first
    lines = [json.dumps(_body(s, f'2026-05-0{d}T00:00:00Z', seed=d))
             for d in (1, 2, 3) for s in (s1, s2)]
    r = client.post('/v1/waveforms:batch', content='\n'.join(lines),
                    headers=hdr(tok))
    out = [json.loads(l) for l in r.text.splitlines()]
    assert len(out) == 6 and all('acq_id' in o for o in out), out
    # first per sensor is the anchor; later ones ride the chain
    assert out[0]['frame_kind'] == 'A' and out[1]['frame_kind'] == 'A'
    assert all(o['frame_kind'] in ('R', 'P') for o in out[2:])

# ------------------------------------------------------------ (b) API keys
def test_key_mint_scopes_revocation(client):
    ws = make_ws(client)
    tok = ws['token']
    sid = make_tree(client, tok)

    mk = lambda scope, label: client.post(
        '/v1/keys', json=dict(scope=scope, label=label), headers=hdr(tok))
    read = mk('read', 'dashboard').json()
    ing = mk('ingest', 'edge-gw').json()
    full = mk('full', 'admin').json()
    assert {'key_id', 'token'} <= set(read)
    assert mk('nope', 'x').status_code == 400

    # read key: GETs work, writes and key management are forbidden
    assert client.get('/v1/fleet/overview', headers=hdr(read['token'])).status_code == 200
    b1 = _body(sid, '2026-06-01T00:00:00Z')
    assert client.post('/v1/waveforms', json=b1, headers=hdr(read['token'])).status_code == 403
    assert client.post('/v1/waveforms:batch', content=json.dumps(b1),
                       headers=hdr(read['token'])).status_code == 403
    assert client.post('/v1/hierarchy:ensure', json=dict(plants=[]),
                       headers=hdr(read['token'])).status_code == 403
    assert client.post('/v1/keys', json=dict(scope='read'),
                       headers=hdr(read['token'])).status_code == 403
    assert client.get('/v1/keys', headers=hdr(read['token'])).status_code == 403
    # /me works but never hands a scoped key a workspace token
    me = client.get('/v1/workspaces/me', headers=hdr(read['token']))
    assert me.status_code == 200 and 'token' not in me.json()

    # ingest key: POST waveforms only
    r = client.post('/v1/waveforms', json=b1, headers=hdr(ing['token']))
    assert r.status_code == 200, r.text
    assert client.get('/v1/fleet/overview', headers=hdr(ing['token'])).status_code == 403
    assert client.get(f'/v1/sensors/{sid}/trend', headers=hdr(ing['token'])).status_code == 403

    # full key does both
    assert client.get('/v1/fleet/overview', headers=hdr(full['token'])).status_code == 200
    r = client.post('/v1/waveforms', json=_body(sid, '2026-06-02T00:00:00Z', seed=2),
                    headers=hdr(full['token']))
    assert r.status_code == 200

    keys = client.get('/v1/keys', headers=hdr(tok)).json()
    assert len(keys) == 3 and not any(k['revoked'] for k in keys)
    assert {k['scope'] for k in keys} == {'read', 'ingest', 'full'}

    # revocation kills the token immediately
    assert client.delete(f"/v1/keys/{read['key_id']}", headers=hdr(tok)).status_code == 200
    assert client.get('/v1/fleet/overview', headers=hdr(read['token'])).status_code == 401
    keys = {k['key_id']: k for k in client.get('/v1/keys', headers=hdr(tok)).json()}
    assert keys[read['key_id']]['revoked'] is True
    assert client.delete('/v1/keys/doesnotexist', headers=hdr(tok)).status_code == 404
    # tampered key token rejected
    bad = ing['token'][:-4]+'AAAA'
    assert client.post('/v1/waveforms', json=b1, headers=hdr(bad)).status_code == 401

# ------------------------------------------------------- (c) quotas/bounds
def test_bounds_and_rate_cap(client, monkeypatch):
    from relspec_service import config as cfg, ingest as ing
    ws = make_ws(client, phrase='a dedicated quota test phrase')
    tok = ws['token']
    sid = make_tree(client, tok)

    # fs bounds
    bad = _body(sid, '2026-07-01T00:00:00Z'); bad['fs'] = 500_000
    assert client.post('/v1/waveforms', json=bad, headers=hdr(tok)).status_code == 400
    bad['fs'] = 0.5
    assert client.post('/v1/waveforms', json=bad, headers=hdr(tok)).status_code == 400

    # oversize body -> 413
    monkeypatch.setattr(cfg, 'MAX_SAMPLES', 1000)
    r = client.post('/v1/waveforms', json=_body(sid, '2026-07-01T00:00:00Z'),
                    headers=hdr(tok))
    assert r.status_code == 413
    monkeypatch.setattr(cfg, 'MAX_SAMPLES', 8_000_000)

    # rate cap: 3/min then 429 + Retry-After
    monkeypatch.setattr(cfg, 'INGEST_RATE_PER_MIN', 3)
    ing._rate.pop(ws['workspace_id'], None)
    for d in (1, 2, 3):
        r = client.post('/v1/waveforms',
                        json=_body(sid, f'2026-07-0{d}T12:00:00Z', seed=d),
                        headers=hdr(tok))
        assert r.status_code == 200, r.text
    r = client.post('/v1/waveforms', json=_body(sid, '2026-07-04T12:00:00Z'),
                    headers=hdr(tok))
    assert r.status_code == 429
    assert int(r.headers['retry-after']) >= 1
    # batch reserves its lines up front, so it hits the same wall
    r = client.post('/v1/waveforms:batch',
                    content=json.dumps(_body(sid, '2026-07-05T12:00:00Z')),
                    headers=hdr(tok))
    assert r.status_code == 429
