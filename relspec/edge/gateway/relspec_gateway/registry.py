"""Live device registry, fed by beacons and stream activity."""
from __future__ import annotations
import time
from dataclasses import dataclass, field


@dataclass
class Device:
    dev_id: str
    sensor_path: str = ''
    fw_ver: int = 0
    fs_hz: int = 0
    battery_mv: int = 0
    uptime_s: int = 0
    drop_total: int = 0
    streaming: bool = False
    connected: bool = False
    last_beacon: float = 0.0
    last_acq_ts: str = ''
    rms_g: float = 0.0
    peak_g: float = 0.0
    crest: float = 0.0
    acq_count: int = 0
    authenticated: bool = False
    quarantined: bool = False
    quarantine_drops: int = 0      # acquisition windows discarded
    extra: dict = field(default_factory=dict)


class Registry:
    """Mapping file values are either "Plant/Asset/Comp/Sensor" (legacy,
    keyless) or {"path": ..., "key": "<hex>"} for authenticated devices."""

    def __init__(self, mapping: dict):
        self._devs: dict[str, Device] = {}
        self._paths: dict[str, str] = {}
        self._keys: dict[str, bytes] = {}
        for dev, v in mapping.items():
            if isinstance(v, dict):
                self._paths[dev] = v.get('path', '')
                if v.get('key'):
                    self._keys[dev] = bytes.fromhex(v['key'])
            else:
                self._paths[dev] = v
        self.on_change = None          # callback(dev) for OPC UA refresh

    def mapping_for(self, dev_id: str) -> str:
        return self._paths.get(dev_id, '')

    def key_for(self, dev_id: str) -> bytes | None:
        return self._keys.get(dev_id)

    def known(self, dev_id: str) -> bool:
        return dev_id in self._paths

    def get(self, dev_id: str) -> Device:
        if dev_id not in self._devs:
            self._devs[dev_id] = Device(dev_id,
                                        sensor_path=self.mapping_for(dev_id))
        return self._devs[dev_id]

    def all(self) -> list[Device]:
        return list(self._devs.values())

    def beacon(self, b) -> Device:
        d = self.get(b.dev_id)
        d.fw_ver, d.fs_hz = b.fw_ver, b.fs_hz
        d.battery_mv, d.uptime_s = b.battery_mv, b.uptime_s
        d.drop_total = b.drop_total
        d.streaming = bool(b.flags & 1)
        d.last_beacon = time.time()
        if self.on_change: self.on_change(d)
        return d
