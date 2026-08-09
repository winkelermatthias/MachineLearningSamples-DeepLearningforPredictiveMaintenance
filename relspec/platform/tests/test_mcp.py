"""MCP tools against a real running service: dev_server subprocess
(embedded Postgres + API), then the tool functions end to end — connect,
hierarchy, submit, duplicate replay, status, trend, prompt."""
import base64, pathlib, socket, subprocess, sys, time
import numpy as np
import pytest

PLATFORM = pathlib.Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def live_api(tmp_path_factory):
    port = _free_port()
    data = tmp_path_factory.mktemp('mcp-pg')
    proc = subprocess.Popen(
        [sys.executable, str(PLATFORM / 'scripts' / 'dev_server.py'),
         '--port', str(port), '--data', str(data)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = f'http://127.0.0.1:{port}'
    import httpx
    for _ in range(120):
        try:
            if httpx.get(base+'/health', timeout=2).status_code == 200: break
        except Exception: time.sleep(0.5)
    else:
        proc.kill()
        raise RuntimeError('dev_server did not become healthy')
    yield base
    proc.terminate(); proc.wait(timeout=20)

def _free_port():
    s = socket.socket(); s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]; s.close(); return p

def test_mcp_tools_roundtrip(live_api, monkeypatch):
    sys.path.insert(0, str(PLATFORM / 'mcpserver'))
    from relspec_mcp import server as S
    monkeypatch.setattr(S, 'API', live_api)
    S._state['token'] = None

    out = S.relspec_create_workspace('a very fine mcp test passphrase', 'mcp')
    assert out['created'] and 'token' not in out          # token kept private

    S.relspec_ensure_hierarchy({'plants': [{'name': 'P', 'assets': [
        {'tag': 'A1', 'components': [{'name': 'C1', 'sensors': [
            {'code': 'S1', 'units': 'g', 'fs': 12000,
             'fr_nominal': 29.5}]}]}]}]})
    tree = S.relspec_get_hierarchy()
    assert tree['plants'][0]['assets'][0]['components'][0]['sensors'][0]['code'] == 'S1'

    sys.path.insert(0, str(PLATFORM.parent / 'src'))
    from relspec.synth2 import FaultState2, machine_catalog2, generate2
    x = generate2(machine_catalog2()['centrifugal_pump'], FaultState2(),
                  2.0, 29.5, rng=np.random.default_rng(3)).astype('float64')
    b64 = base64.b64encode(np.asarray(x, dtype='<f4').tobytes()).decode()
    v = S.relspec_submit_waveform('P/A1/C1/S1', '2026-04-01T00:00:00Z',
                                  12000, b64, client_ref='r1')
    assert v['waveform_stored'] and v['sig_reason'] == 'first_waveform'
    d = S.relspec_submit_waveform('P/A1/C1/S1', '2026-04-01T00:00:00Z',
                                  12000, b64, client_ref='r1')
    assert d['duplicate'] is True

    st = S.relspec_get_status('P/A1/C1/S1')
    assert st['workspace']['acquisitions'] == 1
    assert len(st['sensors']) == 1

    tr = S.relspec_get_trend('P/A1/C1/S1')
    assert len(tr) == 1 and tr[0]['n'] == 1

    with pytest.raises(RuntimeError, match='no sensor at path'):
        S.relspec_get_trend('P/A1/C1/NOPE')

    p = S.ingest_migration()
    assert 'oldest-first' in p or 'OLDEST FIRST' in p


def test_mcp_analysis_and_migration_tools(live_api, monkeypatch):
    """The diagnosis-copilot surface: health, ledger, spectrum (size-
    bounded), fingerprint compare, dry-run-first migrations, scoped key
    minting — all through the tool functions against the live server."""
    sys.path.insert(0, str(PLATFORM / 'mcpserver'))
    from relspec_mcp import server as S
    monkeypatch.setattr(S, 'API', live_api)
    S._state['token'] = None
    S.relspec_connect('a very fine mcp test passphrase')

    sys.path.insert(0, str(PLATFORM.parent / 'src'))
    from relspec.synth2 import FaultState2, machine_catalog2, generate2

    def b64(seed, sev=0.0):
        f = FaultState2(outer_race=sev) if sev else FaultState2()
        x = generate2(machine_catalog2()['centrifugal_pump'], f,
                      2.0, 29.5, rng=np.random.default_rng(seed))
        return base64.b64encode(np.asarray(x, '<f4').tobytes()).decode()

    # 12 more waveforms on the sensor test_mcp_tools_roundtrip created
    # (it left one at 2026-04-01); mild defect then growth for z
    for d in range(2, 14):
        sev = 0.12 if d < 11 else 0.12 + 0.25 * (d - 10)
        S.relspec_submit_waveform('P/A1/C1/S1', f'2026-04-{d:02d}T00:00:00Z',
                                  12000, b64(40 + d, sev),
                                  client_ref=f'm{d}')

    h = S.relspec_get_health('P/A1/C1/S1')
    assert h['tier'] in ('healthy', 'monitor', 'alert', 'critical')
    assert h['trend'] and h['drivers']

    fh = S.relspec_get_fleet_health()
    assert fh['assets'][0]['asset'] == 'A1'
    assert 'healthy' in fh['plants'][0]

    led = S.relspec_get_pattern_ledger('P/A1/C1/S1')
    assert led['tracks'], 'ledger empty'
    zs = [t['z'] for t in led['tracks'] if t['z'] is not None]
    if zs:   # sorted most-anomalous first when any z exists
        assert abs(led['tracks'][0]['z']) == max(abs(z) for z in zs)

    sp = S.relspec_get_spectrum('P/A1/C1/S1', 'env')
    assert len(sp['points']) <= 128
    assert 1 <= len(sp['peaks']) <= 8
    assert all(len(pt) == 2 for pt in sp['points'])

    cmp_ = S.relspec_compare_frames('P/A1/C1/S1', n_back=10)
    assert len(cmp_['acc_band_delta_db']) == 16
    assert len(cmp_['env_band_delta_db']) == 8
    assert cmp_['then']['ts'] < cmp_['now']['ts']

    # migrations: dry-run catches a bad item, clean job completes
    bad = S.relspec_migrate_dry_run([
        dict(sensor_path='P/A1/C1/S1', ts='2026-04-20T00:00:00Z',
             fs=12000, units='g', encoding='float32', data_b64=b64(90)),
        dict(sensor_path='P/A1/C1/NOPE', ts='2026-04-21T00:00:00Z',
             fs=12000, units='g', encoding='float32', data_b64=b64(91)),
    ])
    assert bad['ok'] == 1 and bad['errors'] == 1
    assert bad['failed_items'][0]['ordinal'] == 1

    job = S.relspec_migrate_start([
        dict(sensor_path='P/A1/C1/S1', ts=f'2026-04-2{d}T00:00:00Z',
             fs=12000, units='g', encoding='float32', data_b64=b64(95 + d))
        for d in range(0, 3)])
    for _ in range(120):
        st = S.relspec_migrate_status(job['job_id'])
        if st['state'] == 'done':
            break
        time.sleep(1)
    assert st['state'] == 'done' and st['done'] == 3 and st['errors'] == 0

    # scoped key: ingest can write but not read
    key = S.relspec_mint_key('ingest', 'test-gw')
    assert 'warning' in key
    import httpx
    kh = {'authorization': f"Bearer {key['token']}"}
    r = httpx.get(live_api + '/v1/fleet/overview', headers=kh)
    assert r.status_code == 403
    r = httpx.post(live_api + '/v1/waveforms', headers=kh, timeout=60,
                   json=dict(sensor_path='P/A1/C1/S1',
                             ts='2026-04-25T00:00:00Z', fs=12000, units='g',
                             encoding='float32', data_b64=b64(99)))
    assert r.status_code == 200, r.text

    p = S.ingest_migration()
    assert 'DRY-RUN' in p or 'dry-run' in p.lower()
