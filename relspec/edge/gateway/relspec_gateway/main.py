"""Gateway orchestrator: beacons + streams + spool + uplink + OPC UA +
a tiny HTTP status/control surface.

    python -m relspec_gateway.main --cloud http://api:8000 \
        --passphrase "$RS_PASSPHRASE" --mapping devices.json

Env fallbacks: RS_CLOUD, RS_PASSPHRASE, RS_MAPPING, RS_SPOOL, RS_HTTP_PORT,
RS_OPCUA_ENDPOINT, RS_BEACON_PORT, RS_STREAM_PORT, RS_BIND.

HTTP surface (JSON): GET /status; POST /ctrl {dev,cmd,arg}
"""
from __future__ import annotations
import argparse, asyncio, json, logging, os, signal
from . import proto
from .ingestd import start_servers
from .opcua_server import OpcUa
from .registry import Registry
from .spool import Spool
from .uplink import Uplink

log = logging.getLogger('gateway')

CMDS = {'start': proto.C_START, 'stop': proto.C_STOP,
        'set_fs': proto.C_SET_FS, 'ident': proto.C_IDENT}


def parse_args(argv=None):
    e = os.environ.get
    ap = argparse.ArgumentParser()
    ap.add_argument('--cloud', default=e('RS_CLOUD', 'http://127.0.0.1:8000'))
    ap.add_argument('--passphrase', default=e('RS_PASSPHRASE', ''))
    ap.add_argument('--mapping', default=e('RS_MAPPING', 'devices.json'),
                    help='JSON file: {"<dev_id>": "Plant/Asset/Comp/Sensor"}')
    ap.add_argument('--spool', default=e('RS_SPOOL', 'gateway-spool.db'))
    ap.add_argument('--bind', default=e('RS_BIND', '0.0.0.0'))
    ap.add_argument('--beacon-port', type=int,
                    default=int(e('RS_BEACON_PORT', str(proto.PORT_BEACON))))
    ap.add_argument('--stream-port', type=int,
                    default=int(e('RS_STREAM_PORT', str(proto.PORT_STREAM))))
    ap.add_argument('--http-port', type=int,
                    default=int(e('RS_HTTP_PORT', '8080')))
    ap.add_argument('--opcua', default=e('RS_OPCUA_ENDPOINT',
                    'opc.tcp://0.0.0.0:4840/relspec/'))
    ap.add_argument('--no-uplink', action='store_true')
    return ap.parse_args(argv)


class Http:
    """Deliberately tiny: two JSON endpoints, no framework."""

    def __init__(self, gw): self.gw = gw

    async def handle(self, reader, writer):
        try:
            req = await asyncio.wait_for(reader.readuntil(b'\r\n\r\n'), 10)
            line = req.split(b'\r\n', 1)[0].decode()
            method, path, _ = line.split(' ', 2)
            body = b''
            for h in req.split(b'\r\n'):
                if h.lower().startswith(b'content-length:'):
                    body = await reader.readexactly(int(h.split(b':')[1]))
            if method == 'GET' and path == '/status':
                out, code = self.gw.status(), 200
            elif method == 'POST' and path == '/ctrl':
                out, code = await self.gw.ctrl(json.loads(body or b'{}'))
            else:
                out, code = {'error': 'not found'}, 404
            payload = json.dumps(out).encode()
            writer.write(b'HTTP/1.1 %d OK\r\nContent-Type: application/json'
                         b'\r\nContent-Length: %d\r\n\r\n'
                         % (code, len(payload)) + payload)
            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()


class Gateway:
    def __init__(self, args):
        self.args = args
        mapping = {}
        if os.path.exists(args.mapping):
            mapping = json.loads(open(args.mapping).read())
        self.registry = Registry(mapping)
        self.spool = Spool(args.spool)
        self.uplink = Uplink(args.cloud, args.passphrase, self.spool,
                             self.registry)
        self.opcua = OpcUa(self.registry, args.opcua)
        self.stream_srv = None
        self._loop = None

    def status(self) -> dict:
        c = self.spool.counts()
        return dict(
            devices=[vars(d) | {'extra': d.extra} for d in
                     self.registry.all()],
            spool=c, cloud_online=self.uplink.online,
            uplink_sent=self.uplink.sent_total)

    async def ctrl(self, body: dict):
        cmd = CMDS.get(str(body.get('cmd', '')).lower())
        if cmd is None:
            return {'error': f"unknown cmd {body.get('cmd')!r}"}, 400
        ok = await self.stream_srv.send_ctrl(
            body.get('dev', ''), cmd, int(body.get('arg', 0)))
        return ({'sent': True} if ok else
                {'error': 'device not connected'}), (200 if ok else 404)

    def _on_change(self, dev):
        if self._loop:
            self._loop.create_task(self.opcua.update_device(dev))
            c = self.spool.counts()
            self._loop.create_task(self.opcua.update_gateway(
                c.get('pending', 0), c.get('sent', 0), self.uplink.online))

    async def run(self):
        self._loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:
                pass
        await self.opcua.start()
        self.registry.on_change = self._on_change
        transport, tcp, self.stream_srv = await start_servers(
            self.registry, self.spool, self.args.bind,
            self.args.beacon_port, self.args.stream_port)
        http = await asyncio.start_server(
            Http(self).handle, self.args.bind, self.args.http_port)
        log.info('http status on :%d', self.args.http_port)
        tasks = []
        if not self.args.no_uplink:
            tasks.append(asyncio.create_task(self.uplink.run(stop)))
        tasks.append(asyncio.create_task(self._gauge_loop(stop)))
        await stop.wait()
        for t in tasks: t.cancel()
        transport.close(); tcp.close(); http.close()
        await self.opcua.stop()

    async def _gauge_loop(self, stop):
        while not stop.is_set():
            try:
                c = self.spool.counts()
                await self.opcua.update_gateway(
                    c.get('pending', 0), c.get('sent', 0), self.uplink.online)
            except Exception:
                log.exception('gauge update failed')
            try:
                await asyncio.wait_for(stop.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s')
    logging.getLogger('asyncua').setLevel(logging.WARNING)
    asyncio.run(Gateway(parse_args(argv)).run())


if __name__ == '__main__':
    main()
