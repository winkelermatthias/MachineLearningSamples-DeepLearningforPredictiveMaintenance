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
