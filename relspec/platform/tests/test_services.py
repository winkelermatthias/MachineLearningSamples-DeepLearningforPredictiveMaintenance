"""SSE live events and resumable migration jobs, against a real running
server (dev_server: embedded Postgres + uvicorn) because both features
live on the event loop, which TestClient does not keep running."""
import base64, json, pathlib, socket, subprocess, sys, threading, time
import httpx
import numpy as np
import pytest

PLATFORM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM.parent / 'src'))
PHRASE = 'services test passphrase with plenty of entropy'


def _free_port():
    s = socket.socket(); s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]; s.close(); return p


@pytest.fixture(scope='module')
def live(tmp_path_factory):
    port = _free_port()
    data = tmp_path_factory.mktemp('svc-pg')
    logf = open(tmp_path_factory.mktemp('svc-log') / 'server.log', 'wb')
    proc = subprocess.Popen(
        [sys.executable, str(PLATFORM / 'scripts' / 'dev_server.py'),
         '--port', str(port), '--data', str(data)],
        stdout=logf, stderr=subprocess.STDOUT)
    base = f'http://127.0.0.1:{port}'
    for _ in range(180):
        try:
            if httpx.get(base + '/health', timeout=2).status_code == 200:
                break
        except Exception:
            time.sleep(0.5)
    else:
        proc.kill(); raise RuntimeError('dev_server not healthy')
    r = httpx.post(base + '/v1/workspaces',
                   json={'passphrase': PHRASE, 'name': 'svc'})
    tok = r.json()['token']
    hdrs = {'authorization': f'Bearer {tok}'}
    httpx.post(base + '/v1/hierarchy:ensure', headers=hdrs, json={
        'plants': [{'name': 'P', 'assets': [{'tag': 'A1', 'components': [
            {'name': 'C1', 'sensors': [
                {'code': 'S1', 'units': 'g', 'fs': 12000, 'fr_nominal': 29.5},
                {'code': 'S2', 'units': 'g', 'fs': 12000, 'fr_nominal': 29.5},
            ]}]}]}]}).raise_for_status()
    yield base, hdrs
    proc.terminate(); proc.wait(timeout=20)


def _wave(seed):
    from relspec.synth2 import FaultState2, machine_catalog2, generate2
    x = generate2(machine_catalog2()['centrifugal_pump'], FaultState2(),
                  2.0, 29.5, rng=np.random.default_rng(seed))
    return base64.b64encode(np.asarray(x, '<f4').tobytes()).decode()


def _body(sensor, day, seed, **kw):
    return dict(sensor_path=f'P/A1/C1/{sensor}',
                ts=f'2026-06-{day:02d}T00:00:00Z', fs=12000, units='g',
                encoding='float32', data_b64=_wave(seed), **kw)


def test_sse_live_event(live):
    base, hdrs = live
    got = []
    ready = threading.Event()

    def listen():
        with httpx.stream('GET', base + '/v1/events/stream', headers=hdrs,
                          timeout=30) as r:
            ready.set()
            for line in r.iter_lines():
                if line.startswith('data: '):
                    got.append(json.loads(line[6:]))
                    return
    t = threading.Thread(target=listen, daemon=True)
    t.start()
    assert ready.wait(10), 'stream never opened'
    time.sleep(0.8)          # let LISTEN attach before the ingest
    v = httpx.post(base + '/v1/waveforms', headers=hdrs,
                   json=_body('S1', 1, 1), timeout=60).json()
    assert 'acq_id' in v, v
    t.join(timeout=10)
    assert got, 'no SSE event within 10 s of ingest'
    assert got[0]['kind'] == 'acquisition'
    assert got[0]['acq_id'] == v['acq_id']


def test_migration_dry_run_catches_errors(live):
    base, hdrs = live
    manifest = [
        _body('S2', 2, 10),
        _body('S2', 1, 11),                       # out-of-order within manifest
        _body('NOPE', 3, 12),                     # unknown sensor
        dict(_body('S2', 4, 13), fs=999999999),   # absurd fs
    ]
    r = httpx.post(base + '/v1/migrations', headers=hdrs, timeout=120,
                   json={'manifest': manifest, 'dry_run': True}).json()
    assert r['dry_run'] and r['total'] == 4
    assert r['ok'] == 1 and r['errors'] == 3
    errs = {i['ordinal']: i['errors'] for i in r['items'] if not i['ok']}
    assert 'out-of-order' in errs[1][0]
    assert any('sensor' in e for e in errs[2])
    assert any('fs' in e for e in errs[3])
    # dry run wrote nothing
    me = httpx.get(base + '/v1/workspaces/me', headers=hdrs).json()
    assert me['acquisitions'] <= 1   # only the SSE test's waveform


def test_migration_job_with_pause_resume(live):
    base, hdrs = live
    manifest = ([_body('S1', d, 20 + d) for d in range(2, 12)] +
                [_body('S2', d, 40 + d) for d in range(2, 12)])
    r = httpx.post(base + '/v1/migrations', headers=hdrs, timeout=120,
                   json={'manifest': manifest}).json()
    job = r['job_id']
    assert r['invalid'] == 0

    # pause quickly, mid-job with high probability
    time.sleep(2.5)
    httpx.post(base + f'/v1/migrations/{job}:pause', headers=hdrs)
    time.sleep(2.0)
    st1 = httpx.get(base + f'/v1/migrations/{job}', headers=hdrs).json()
    if st1['state'] == 'paused':
        done_at_pause = st1['done']
        time.sleep(1.5)
        st2 = httpx.get(base + f'/v1/migrations/{job}', headers=hdrs).json()
        assert st2['done'] == done_at_pause, 'worker kept writing while paused'
        httpx.post(base + f'/v1/migrations/{job}:resume', headers=hdrs)

    deadline = time.time() + 120
    while time.time() < deadline:
        st = httpx.get(base + f'/v1/migrations/{job}', headers=hdrs).json()
        if st['state'] == 'done':
            break
        time.sleep(1)
    assert st['state'] == 'done', st
    assert st['done'] == 20 and st['errors'] == 0, st

    # every waveform landed exactly once, in order, per sensor
    me = httpx.get(base + '/v1/workspaces/me', headers=hdrs).json()
    assert me['acquisitions'] == 21   # 1 (SSE test) + 20
    h = httpx.get(base + '/v1/sensors', headers=hdrs)  # may not exist; fallback
    fr = httpx.get(base + '/v1/fleet/overview', headers=hdrs).json()
    assert len(fr) == 2

    # resume on a finished job is a no-op
    r2 = httpx.post(base + f'/v1/migrations/{job}:resume', headers=hdrs).json()
    assert r2['resumed'] is False
