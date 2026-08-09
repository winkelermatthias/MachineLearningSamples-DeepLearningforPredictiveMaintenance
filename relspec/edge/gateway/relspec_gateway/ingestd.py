"""RSP/1 network servers: UDP beacon listener + TCP stream server.

The stream server reassembles DATA frames into acquisition windows;
ACQ_END closes a window, which is feature-extracted and spooled in one
synchronous step — an acquisition is durable on gateway disk before the
next frame is read. Control frames can be pushed back to any connected
device (used by the /ctrl HTTP endpoint and auto-STOP policies).
"""
from __future__ import annotations
import asyncio, datetime as dt, hmac, logging, secrets
import numpy as np
from . import proto
from .edgeproc import features
from .registry import Registry
from .spool import Spool

log = logging.getLogger('ingestd')


class BeaconProtocol(asyncio.DatagramProtocol):
    def __init__(self, registry: Registry):
        self.registry = registry

    def datagram_received(self, data: bytes, addr):
        b = proto.Beacon.parse(data)
        if b is None:
            return
        d = self.registry.beacon(b)
        d.extra['addr'] = addr[0]


class StreamServer:
    def __init__(self, registry: Registry, spool: Spool):
        self.registry = registry
        self.spool = spool
        self.conns: dict[str, asyncio.StreamWriter] = {}
        self.on_acq = None            # callback(dev) after spooling

    async def handle(self, reader: asyncio.StreamReader,
                     writer: asyncio.StreamWriter):
        dev_id = ''
        dev = None
        window: list[bytes] = []
        w_t0_us = 0; w_fs = 0.0; w_scale = 0.0
        key = None          # configured device key, if any
        nonce = None        # outstanding CHALLENGE nonce
        authed = False      # this connection may spool acquisitions
        try:
            while True:
                head = await reader.readexactly(8)
                total = proto.peek_frame_len(head)
                if total is None:
                    log.warning('bad frame header from %s; closing',
                                writer.get_extra_info('peername'))
                    return
                frame = head + await reader.readexactly(total - 8)
                parsed = proto.parse_frame(frame)
                if parsed is None:
                    log.warning('CRC fail from %s; closing', dev_id or '?')
                    return
                hdr, payload = parsed
                if hdr.type == proto.F_HELLO:
                    dev_id = hdr.dev_id
                    dev = self.registry.get(dev_id)
                    dev.connected = True
                    self.conns[dev_id] = writer
                    key = self.registry.key_for(dev_id)
                    authed = False
                    if key:
                        nonce = secrets.randbits(32)
                        writer.write(proto.encode_ctrl(proto.C_CHALLENGE,
                                                       nonce))
                        await writer.drain()
                        log.info('device %s connected; challenge sent',
                                 dev_id)
                    elif self.registry.known(dev_id):
                        authed = True
                        log.warning('device %s has no key configured; '
                                    'accepting unauthenticated', dev_id)
                        log.info('device %s connected (fs=%s)',
                                 dev_id, hdr.fs_hz)
                    else:
                        dev.quarantined = True
                        log.warning('unknown device %s quarantined '
                                    '(not in mapping)', dev_id)
                elif hdr.type == proto.F_AUTH:
                    if dev and key and nonce is not None and \
                            hmac.compare_digest(
                                payload[:proto.AUTH_TAG_LEN],
                                proto.auth_tag(key, dev_id, nonce)):
                        authed = True
                        dev.authenticated = True
                        log.info('device %s authenticated', dev_id)
                    elif dev:
                        dev.quarantined = True
                        log.warning('device %s failed auth; quarantined',
                                    dev_id)
                    nonce = None
                    if dev and self.registry.on_change:
                        self.registry.on_change(dev)
                elif hdr.type == proto.F_DATA:
                    if dev is None or dev.quarantined or not authed:
                        continue
                    if not window:
                        w_t0_us, w_fs, w_scale = hdr.t0_us, hdr.fs_hz, hdr.scale_g
                    window.append(payload)
                elif hdr.type == proto.F_ACQ_END:
                    if dev is None or dev.quarantined or not authed:
                        if dev:
                            dev.quarantine_drops += 1
                            log.warning('%s: dropped window %d '
                                        '(unauthenticated/quarantined)',
                                        dev.dev_id, dev.quarantine_drops)
                            if self.registry.on_change:
                                self.registry.on_change(dev)
                        window = []
                        continue
                    self._finish(hdr, window, w_t0_us, w_fs, w_scale, payload)
                    window = []
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            if dev_id:
                self.registry.get(dev_id).connected = False
                self.conns.pop(dev_id, None)
                log.info('device %s disconnected', dev_id)
            writer.close()

    def _finish(self, hdr, window, t0_us, fs, scale, aux: bytes):
        dev = self.registry.get(hdr.dev_id)
        samples = b''.join(window)
        claimed = int.from_bytes(aux[0:4], 'little') if len(aux) >= 8 else 0
        dropped = int.from_bytes(aux[4:8], 'little') if len(aux) >= 8 else 0
        got = len(samples) // 2
        if claimed and got != claimed:
            log.warning('%s: window incomplete (%d/%d) — discarding',
                        hdr.dev_id, got, claimed)
            return
        ts = dt.datetime.fromtimestamp(t0_us / 1e6, dt.timezone.utc)
        f = features(samples, scale)
        dev.rms_g, dev.peak_g, dev.crest = f['rms_g'], f['peak_g'], f['crest']
        dev.last_acq_ts = ts.isoformat()
        dev.acq_count += 1
        dev.drop_total += dropped
        self.spool.add(hdr.dev_id, dev.sensor_path, ts.isoformat(),
                       float(fs), float(scale), samples)
        log.info('%s: acq %s  n=%d fs=%g  rms=%.3f g crest=%.1f -> spooled',
                 hdr.dev_id, ts.isoformat(), got, fs, f['rms_g'], f['crest'])
        if self.registry.on_change: self.registry.on_change(dev)
        if self.on_acq: self.on_acq(dev)

    async def send_ctrl(self, dev_id: str, cmd: int, arg: int = 0) -> bool:
        w = self.conns.get(dev_id)
        if w is None: return False
        w.write(proto.encode_ctrl(cmd, arg))
        await w.drain()
        return True


async def start_servers(registry: Registry, spool: Spool,
                        host: str, beacon_port: int, stream_port: int):
    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(
        lambda: BeaconProtocol(registry), local_addr=(host, beacon_port))
    srv = StreamServer(registry, spool)
    tcp = await asyncio.start_server(srv.handle, host, stream_port)
    log.info('listening: beacons udp/%d, streams tcp/%d',
             beacon_port, stream_port)
    return transport, tcp, srv
