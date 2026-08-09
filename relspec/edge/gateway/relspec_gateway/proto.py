"""Python side of RSP/1 — mirrors firmware/core/rs_proto.h exactly.

Byte-for-byte the same wire format; the C unit tests and
tests/test_chain.py both pin it, so a drift on either side fails CI.
"""
from __future__ import annotations
import hmac, struct, zlib
from dataclasses import dataclass

PORT_BEACON = 47700
PORT_STREAM = 47701

MAGIC_BEACON = 0x31425352
MAGIC_DATA = 0x31445352
MAGIC_CTRL = 0x31435352

F_HELLO, F_DATA, F_ACQ_END, F_STATUS, F_AUTH = 1, 2, 3, 4, 5
(C_START, C_STOP, C_SET_FS, C_SET_ACQ_MS, C_IDENT, C_REBOOT,
 C_CHALLENGE) = 1, 2, 3, 4, 5, 6, 7

BF_STREAMING = 0x01

BEACON_LEN = 40
DATA_HDR_LEN = 40
CTRL_LEN = 16
MAX_SAMPLES = 1024


def crc32(b: bytes) -> int:
    return zlib.crc32(b) & 0xFFFFFFFF


@dataclass
class Beacon:
    ver: int; flags: int; fw_ver: int; dev_id: str
    fs_hz: int; channels: int; battery_mv: int
    uptime_s: int; drop_total: int

    @classmethod
    def parse(cls, buf: bytes) -> 'Beacon | None':
        if len(buf) != BEACON_LEN: return None
        if struct.unpack_from('<I', buf, 0)[0] != MAGIC_BEACON: return None
        if struct.unpack_from('<I', buf, 36)[0] != crc32(buf[:36]): return None
        ver, flags, fw = struct.unpack_from('<BBH', buf, 4)
        dev = buf[8:20].rstrip(b'\0').decode('ascii', 'replace')
        fs, ch, _r, batt = struct.unpack_from('<IBBH', buf, 20)
        up, drops = struct.unpack_from('<II', buf, 28)
        return cls(ver, flags, fw, dev, fs, ch, batt, up, drops)


@dataclass
class DataHdr:
    type: int; n: int; seq: int; dev_id: str
    t0_us: int; fs_hz: int; scale_g: float


def frame_len(n: int) -> int:
    return DATA_HDR_LEN + 2 * n + 4


def parse_frame(buf: bytes) -> tuple[DataHdr, bytes] | None:
    """buf holds exactly one frame; returns (header, payload bytes)."""
    if len(buf) < DATA_HDR_LEN + 4: return None
    if struct.unpack_from('<I', buf, 0)[0] != MAGIC_DATA: return None
    n = struct.unpack_from('<H', buf, 6)[0]
    if n > MAX_SAMPLES or len(buf) != frame_len(n): return None
    body = len(buf) - 4
    if struct.unpack_from('<I', buf, body)[0] != crc32(buf[:body]):
        return None
    typ = buf[4]
    seq = struct.unpack_from('<I', buf, 8)[0]
    dev = buf[12:24].rstrip(b'\0').decode('ascii', 'replace')
    t0, fs, scale = struct.unpack_from('<QIf', buf, 24)
    return DataHdr(typ, n, seq, dev, t0, fs, scale), buf[DATA_HDR_LEN:body]


def peek_frame_len(hdr8: bytes) -> int | None:
    """First 8 bytes -> total frame length, or None if not a data frame."""
    if len(hdr8) < 8: return None
    if struct.unpack_from('<I', hdr8, 0)[0] != MAGIC_DATA: return None
    n = struct.unpack_from('<H', hdr8, 6)[0]
    if n > MAX_SAMPLES: return None
    return frame_len(n)


def encode_ctrl(cmd: int, arg: int = 0) -> bytes:
    head = struct.pack('<IBBBBI', MAGIC_CTRL, cmd, 0, 0, 0, arg)
    return head + struct.pack('<I', crc32(head))


AUTH_TAG_LEN = 16


def auth_tag(key: bytes, dev_id: str, nonce: int) -> bytes:
    """F_AUTH payload — mirrors rs_auth_tag() in core/rs_sha256.c."""
    msg = dev_id.encode('ascii')[:12].ljust(12, b'\0') + \
        struct.pack('<I', nonce)
    return hmac.new(key, msg, 'sha256').digest()[:AUTH_TAG_LEN]
