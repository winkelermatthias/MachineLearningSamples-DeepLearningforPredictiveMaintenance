"""End-to-end edge chain: compiled C device simulator -> gateway
(buffering, edge features, OPC UA) -> cloud platform (FastAPI + real
embedded Postgres) -> verified via cloud API and an OPC UA client.

Also unit-covers the spool's ordering/marking discipline.
"""
import base64, json, pathlib, socket, subprocess, sys, time
import httpx
import pytest

EDGE = pathlib.Path(__file__).resolve().parents[1]
PLATFORM = EDGE.parent / 'platform'
PHRASE = 'edge chain test passphrase with plenty of entropy'
DEV_KEY = '00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff'


def free_port():
    s = socket.socket(); s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]; s.close(); return p


@pytest.fixture(scope='module')
def firmware():
    subprocess.run(['make', 'test', 'device-sim'], cwd=EDGE / 'firmware',
                   check=True, capture_output=True)
    return EDGE / 'firmware' / 'build' / 'device-sim'


@pytest.fixture(scope='module')
def cloud(tmp_path_factory):
    port = free_port()
    data = tmp_path_factory.mktemp('chain-pg')
    proc = subprocess.Popen(
        [sys.executable, str(PLATFORM / 'scripts' / 'dev_server.py'),
         '--port', str(port), '--data', str(data)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = f'http://127.0.0.1:{port}'
    for _ in range(180):
        try:
            if httpx.get(base + '/health', timeout=2).status_code == 200:
                break
        except Exception:
            time.sleep(0.5)
    else:
        proc.kill(); raise RuntimeError('cloud did not become healthy')
    r = httpx.post(base + '/v1/workspaces',
                   json={'passphrase': PHRASE, 'name': 'edge-test'})
    r.raise_for_status()
    token = r.json()['token']
    hdrs = {'authorization': f'Bearer {token}'}
    httpx.post(base + '/v1/hierarchy:ensure', headers=hdrs, json={
        'plants': [{'name': 'P', 'assets': [{'tag': 'A1', 'components': [
            {'name': 'C1', 'sensors': [
                {'code': 'S1', 'units': 'g', 'fs': 8192, 'fr_nominal': 29.5},
                {'code': 'S2', 'units': 'g', 'fs': 8192, 'fr_nominal': 29.5},
                {'code': 'S3', 'units': 'g', 'fs': 8192, 'fr_nominal': 29.5},
            ]}]}]}]}).raise_for_status()
    yield base, hdrs
    proc.terminate(); proc.wait(timeout=20)


@pytest.fixture(scope='module')
def gateway(cloud, tmp_path_factory):
    base, _ = cloud
    tmp = tmp_path_factory.mktemp('gw')
    mapping = tmp / 'devices.json'
    mapping.write_text(json.dumps({
        'RS-000A01': 'P/A1/C1/S1',
        'RS-000B02': 'P/A1/C1/S2',
        'RS-000C03': {'path': 'P/A1/C1/S3', 'key': DEV_KEY},
        'RS-000D04': {'path': 'P/A1/C1/S3', 'key': DEV_KEY},
    }))
    ports = dict(beacon=free_port(), stream=free_port(),
                 http=free_port(), opcua=free_port())
    env = dict(RS_CLOUD=base, RS_PASSPHRASE=PHRASE,
               RS_MAPPING=str(mapping), RS_SPOOL=str(tmp / 'spool.db'),
               RS_BIND='127.0.0.1',
               RS_BEACON_PORT=str(ports['beacon']),
               RS_STREAM_PORT=str(ports['stream']),
               RS_HTTP_PORT=str(ports['http']),
               RS_OPCUA_ENDPOINT=f"opc.tcp://127.0.0.1:{ports['opcua']}/relspec/",
               PATH='/usr/bin:/bin', PYTHONPATH=str(EDGE / 'gateway'))
    logf = open(tmp / 'gateway.log', 'wb')
    proc = subprocess.Popen(
        [sys.executable, '-m', 'relspec_gateway.main'],
        cwd=tmp, env=env, stdout=logf, stderr=subprocess.STDOUT)
    gw_url = f"http://127.0.0.1:{ports['http']}"
    for _ in range(120):
        try:
            if httpx.get(gw_url + '/status', timeout=2).status_code == 200:
                break
        except Exception:
            time.sleep(0.5)
    else:
        proc.kill()
        raise RuntimeError('gateway did not start:\n' +
                           (tmp / 'gateway.log').read_text()[-2000:])
    yield dict(url=gw_url, ports=ports, proc=proc)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_sim(firmware, ports, dev, count=3, extra=()):
    return subprocess.run(
        [str(firmware), '--id', dev,
         '--gateway', f"127.0.0.1:{ports['stream']}",
         '--beacon', f"127.0.0.1:{ports['beacon']}",
         '--fs', '8192', '--acq-ms', '500', '--period-ms', '400',
         '--count', str(count), '--speed', '100', '--growth', '0.5',
         *extra],
        capture_output=True, timeout=120)


def wait_until(fn, timeout=60, every=0.5):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v: return v
        time.sleep(every)
    return None


def test_chain_stream_to_cloud(firmware, cloud, gateway):
    base, hdrs = cloud
    r = run_sim(firmware, gateway['ports'], 'RS-000A01', count=3)
    assert r.returncode == 0, r.stderr.decode()

    # every acquisition must reach the cloud (spool fully drained)
    def sent():
        st = httpx.get(gateway['url'] + '/status').json()
        return st if st['spool'].get('sent', 0) >= 3 else None
    st = wait_until(sent, timeout=90)
    assert st, 'spool never drained: ' + json.dumps(
        httpx.get(gateway['url'] + '/status').json())
    assert st['spool'].get('pending', 0) == 0
    assert st['cloud_online'] is True

    me = httpx.get(base + '/v1/workspaces/me', headers=hdrs).json()
    assert me['acquisitions'] >= 3

    # first waveform of a sensor is significant -> stored raw
    ov = httpx.get(base + '/v1/fleet/overview', headers=hdrs).json()
    assert any(s.get('code') == 'S1' or 'S1' in json.dumps(s)
               for s in (ov if isinstance(ov, list) else ov.get('sensors', [])))

    # device registry saw beacons (battery + fw metadata)
    dev = next(d for d in st['devices'] if d['dev_id'] == 'RS-000A01')
    assert dev['battery_mv'] == 3650
    assert dev['acq_count'] == 3
    assert dev['rms_g'] > 0.05
    assert dev['sensor_path'] == 'P/A1/C1/S1'


def test_chain_opcua_readout(gateway):
    import asyncio
    from asyncua import Client

    async def read():
        ep = f"opc.tcp://127.0.0.1:{gateway['ports']['opcua']}/relspec/"
        async with Client(url=ep) as c:
            idx = await c.get_namespace_index('urn:relspec:gateway')
            dev = await c.nodes.objects.get_child(
                [f'{idx}:Devices', f'{idx}:RS-000A01'])
            rms = await (await dev.get_child(f'{idx}:RmsG')).read_value()
            n = await (await dev.get_child(f'{idx}:AcqCount')).read_value()
            path = await (await dev.get_child(f'{idx}:SensorPath')).read_value()
            gw = await c.nodes.objects.get_child([f'{idx}:Gateway'])
            sent = await (await gw.get_child(f'{idx}:SpoolSent')).read_value()
            online = await (await gw.get_child(f'{idx}:CloudOnline')).read_value()
            return rms, n, path, sent, online

    rms, n, path, sent, online = asyncio.get_event_loop().run_until_complete(read()) \
        if False else asyncio.run(read())
    assert rms > 0.05
    assert n >= 3
    assert path == 'P/A1/C1/S1'
    assert sent >= 3
    assert online is True


def test_chain_control_start(firmware, cloud, gateway):
    """Device boots idle (--wait-start); the gateway's control channel
    starts it; its acquisition then flows all the way to the cloud."""
    base, hdrs = cloud
    import threading
    res = {}

    def run():
        res['r'] = run_sim(firmware, gateway['ports'], 'RS-000B02',
                           count=1, extra=('--wait-start',))
    t = threading.Thread(target=run); t.start()

    def connected():
        st = httpx.get(gateway['url'] + '/status').json()
        return any(d['dev_id'] == 'RS-000B02' and d['connected']
                   for d in st['devices'])
    assert wait_until(connected, timeout=30), 'device never connected'
    r = httpx.post(gateway['url'] + '/ctrl',
                   json={'dev': 'RS-000B02', 'cmd': 'start'})
    assert r.status_code == 200
    t.join(timeout=120)
    assert res['r'].returncode == 0, res['r'].stderr.decode()

    def has_s2():
        st = httpx.get(gateway['url'] + '/status').json()
        d = [x for x in st['devices'] if x['dev_id'] == 'RS-000B02']
        return d and d[0]['acq_count'] >= 1
    assert wait_until(has_s2, timeout=30)


def test_chain_authenticated_device(firmware, cloud, gateway):
    """Keyed device: gateway CHALLENGE -> device F_AUTH -> data flows
    all the way to the cloud."""
    base, hdrs = cloud
    sent0 = httpx.get(gateway['url'] + '/status').json()['spool'].get('sent', 0)
    r = run_sim(firmware, gateway['ports'], 'RS-000C03', count=2,
                extra=('--key', DEV_KEY))
    assert r.returncode == 0, r.stderr.decode()

    def drained():
        st = httpx.get(gateway['url'] + '/status').json()
        return st if st['spool'].get('sent', 0) >= sent0 + 2 else None
    st = wait_until(drained, timeout=90)
    assert st, 'authed acquisitions never reached cloud: ' + json.dumps(
        httpx.get(gateway['url'] + '/status').json())
    dev = next(d for d in st['devices'] if d['dev_id'] == 'RS-000C03')
    assert dev['authenticated'] is True
    assert dev['quarantined'] is False
    assert dev['acq_count'] == 2
    me = httpx.get(base + '/v1/workspaces/me', headers=hdrs).json()
    assert me['acquisitions'] >= 6            # 3 + 1 + 2 from prior tests


def test_chain_wrong_key_quarantined(firmware, cloud, gateway):
    """A device answering the challenge with a wrong key is quarantined:
    registered and counted, but nothing spooled or uplinked."""
    st0 = httpx.get(gateway['url'] + '/status').json()
    sent0 = st0['spool'].get('sent', 0)
    r = run_sim(firmware, gateway['ports'], 'RS-000D04', count=2,
                extra=('--key', 'deadbeef' * 8))
    assert r.returncode == 0, r.stderr.decode()

    def quarantined():
        st = httpx.get(gateway['url'] + '/status').json()
        d = [x for x in st['devices'] if x['dev_id'] == 'RS-000D04']
        return st if d and d[0]['quarantined'] and \
            d[0]['quarantine_drops'] >= 2 else None
    st = wait_until(quarantined, timeout=30)
    assert st, 'device never quarantined: ' + json.dumps(
        httpx.get(gateway['url'] + '/status').json())
    dev = next(d for d in st['devices'] if d['dev_id'] == 'RS-000D04')
    assert dev['authenticated'] is False
    assert dev['acq_count'] == 0
    time.sleep(1.0)
    st = httpx.get(gateway['url'] + '/status').json()
    assert st['spool'].get('sent', 0) == sent0    # nothing new spooled
    assert st['spool'].get('pending', 0) == 0


def test_uplink_cloud_key(tmp_path):
    """RS_CLOUD_KEY mode: the configured token is sent verbatim as
    Bearer on /v1/waveforms — no passphrase exchange."""
    import asyncio
    sys.path.insert(0, str(EDGE / 'gateway'))
    from relspec_gateway.spool import Spool
    from relspec_gateway.uplink import Uplink
    seen = {}

    async def run():
        async def handle(reader, writer):
            req = await reader.readuntil(b'\r\n\r\n')
            seen['path'] = req.split(b'\r\n')[0].split(b' ')[1].decode()
            for h in req.split(b'\r\n'):
                if h.lower().startswith(b'authorization:'):
                    seen['auth'] = h.split(b':', 1)[1].strip().decode()
            payload = json.dumps({'sig_reason': 'ok'}).encode()
            writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: application/'
                         b'json\r\nContent-Length: %d\r\n\r\n'
                         % len(payload) + payload)
            await writer.drain()
            writer.close()
        srv = await asyncio.start_server(handle, '127.0.0.1', 0)
        port = srv.sockets[0].getsockname()[1]
        sp = Spool(str(tmp_path / 'k.db'))
        sp.add('D1', 'P/A/C/S', '2026-01-01T00:00:00+00:00', 8192, 0.001,
               b'\x01\x00' * 64)
        up = Uplink(f'http://127.0.0.1:{port}', 'unused-phrase', sp,
                    cloud_key='tok-EDGE-KEY')
        async with httpx.AsyncClient(timeout=10) as client:
            assert await up.submit_one(client, sp.next_pending())
        srv.close()
        await srv.wait_closed()
        assert sp.counts().get('sent') == 1

    asyncio.run(run())
    assert seen['path'] == '/v1/waveforms'    # never hit /v1/workspaces
    assert seen['auth'] == 'Bearer tok-EDGE-KEY'


def test_spool_discipline(tmp_path):
    sys.path.insert(0, str(EDGE / 'gateway'))
    from relspec_gateway.spool import Spool
    sp = Spool(str(tmp_path / 's.db'))
    a = sp.add('D1', 'P/A/C/S', '2026-01-02T00:00:00+00:00', 8192, 0.001,
               b'\x01\x00' * 100)
    b = sp.add('D1', 'P/A/C/S', '2026-01-01T00:00:00+00:00', 8192, 0.001,
               b'\x02\x00' * 100)
    # oldest-first regardless of insert order (cloud rejects out-of-order)
    first = sp.next_pending()
    assert first['id'] == b and first['n'] == 100
    assert first['data'][0:2] == b'\x02\x00'
    sp.mark(b, 'sent', {'ok': True})
    assert sp.next_pending()['id'] == a
    sp.mark(a, 'skipped', {'error': 'x'})
    assert sp.next_pending() is None
    assert sp.counts() == {'sent': 1, 'skipped': 1}
