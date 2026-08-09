"""OPC UA readout server (asyncua) — the plant-floor face of the gateway.

Namespace `urn:relspec:gateway`:

  Objects/
    Gateway/            SpoolPending, SpoolSent, CloudOnline, DeviceCount
    Devices/<dev_id>/   SensorPath, Streaming, Connected, BatteryMv,
                        DropTotal, RmsG, PeakG, Crest, LastAcqTime,
                        AcqCount

Values refresh on every completed acquisition and beacon; SCADA/
historian clients subscribe like to any other OPC UA server.
"""
from __future__ import annotations
import logging
from asyncua import Server, ua
from .registry import Registry, Device

log = logging.getLogger('opcua')

DEV_VARS = [
    ('SensorPath', 'sensor_path', ua.VariantType.String),
    ('Streaming', 'streaming', ua.VariantType.Boolean),
    ('Connected', 'connected', ua.VariantType.Boolean),
    ('BatteryMv', 'battery_mv', ua.VariantType.UInt16),
    ('DropTotal', 'drop_total', ua.VariantType.UInt32),
    ('RmsG', 'rms_g', ua.VariantType.Double),
    ('PeakG', 'peak_g', ua.VariantType.Double),
    ('Crest', 'crest', ua.VariantType.Double),
    ('LastAcqTime', 'last_acq_ts', ua.VariantType.String),
    ('AcqCount', 'acq_count', ua.VariantType.UInt32),
]


class OpcUa:
    def __init__(self, registry: Registry, endpoint: str):
        self.registry = registry
        self.endpoint = endpoint
        self.server = Server()
        self.idx = 0
        self._dev_nodes: dict[str, dict] = {}
        self._gw: dict = {}

    async def start(self):
        await self.server.init()
        self.server.set_endpoint(self.endpoint)
        self.server.set_server_name('relspec edge gateway')
        self.idx = await self.server.register_namespace('urn:relspec:gateway')
        objects = self.server.nodes.objects
        gw = await objects.add_object(self.idx, 'Gateway')
        for name, vtype, init in [
                ('SpoolPending', ua.VariantType.UInt32, 0),
                ('SpoolSent', ua.VariantType.UInt32, 0),
                ('CloudOnline', ua.VariantType.Boolean, False),
                ('DeviceCount', ua.VariantType.UInt32, 0)]:
            n = await gw.add_variable(self.idx, name,
                                      ua.Variant(init, vtype))
            await n.set_writable(False)
            self._gw[name] = n
        self._devices_obj = await objects.add_object(self.idx, 'Devices')
        await self.server.start()
        log.info('OPC UA serving at %s', self.endpoint)

    async def stop(self):
        await self.server.stop()

    async def _ensure_device(self, d: Device) -> dict:
        if d.dev_id in self._dev_nodes:
            return self._dev_nodes[d.dev_id]
        obj = await self._devices_obj.add_object(self.idx, d.dev_id)
        nodes = {}
        for name, attr, vtype in DEV_VARS:
            val = getattr(d, attr)
            n = await obj.add_variable(self.idx, name, ua.Variant(val, vtype))
            await n.set_writable(False)
            nodes[name] = n
        self._dev_nodes[d.dev_id] = nodes
        return nodes

    async def update_device(self, d: Device):
        nodes = await self._ensure_device(d)
        for name, attr, vtype in DEV_VARS:
            await nodes[name].write_value(
                ua.Variant(getattr(d, attr), vtype))

    async def update_gateway(self, pending: int, sent: int, online: bool):
        await self._gw['SpoolPending'].write_value(
            ua.Variant(pending, ua.VariantType.UInt32))
        await self._gw['SpoolSent'].write_value(
            ua.Variant(sent, ua.VariantType.UInt32))
        await self._gw['CloudOnline'].write_value(bool(online))
        await self._gw['DeviceCount'].write_value(
            ua.Variant(len(self.registry.all()), ua.VariantType.UInt32))
