"""Store-and-forward spool: every completed acquisition lands in SQLite
before anything network-dependent happens. The uplink drains it
oldest-first per device; the cloud being down just makes the spool grow.
Samples are zstd-compressed int16 (the codec's own input format), so a
day of 1 s/2 min acquisitions at 8 kHz is ~15 MB on the gateway disk.
"""
from __future__ import annotations
import json, sqlite3, threading, time
import zstandard

SCHEMA = """
CREATE TABLE IF NOT EXISTS acq(
    id INTEGER PRIMARY KEY,
    dev TEXT NOT NULL,
    sensor_path TEXT NOT NULL,
    ts_iso TEXT NOT NULL,
    fs REAL NOT NULL,
    scale REAL NOT NULL,
    n INTEGER NOT NULL,
    data BLOB NOT NULL,           -- zstd(int16 LE)
    state TEXT NOT NULL DEFAULT 'pending',  -- pending|sent|skipped
    tries INTEGER NOT NULL DEFAULT 0,
    verdict TEXT,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS acq_state ON acq(state, dev, ts_iso);
"""


class Spool:
    def __init__(self, path: str):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()
        self._z = zstandard.ZstdCompressor(level=3)

    def add(self, dev: str, sensor_path: str, ts_iso: str, fs: float,
            scale: float, samples: bytes) -> int:
        blob = self._z.compress(samples)
        with self._lock, self._db:
            cur = self._db.execute(
                'INSERT INTO acq(dev,sensor_path,ts_iso,fs,scale,n,data,created)'
                ' VALUES(?,?,?,?,?,?,?,?)',
                (dev, sensor_path, ts_iso, fs, scale,
                 len(samples) // 2, blob, time.time()))
            return int(cur.lastrowid)

    def next_pending(self) -> dict | None:
        """Oldest pending acquisition, globally time-ordered so the
        cloud's per-sensor ordering constraint is respected."""
        with self._lock:
            row = self._db.execute(
                "SELECT id,dev,sensor_path,ts_iso,fs,scale,n,data,tries "
                "FROM acq WHERE state='pending' ORDER BY ts_iso LIMIT 1"
            ).fetchone()
        if not row: return None
        d = zstandard.ZstdDecompressor().decompress(row[7])
        return dict(id=row[0], dev=row[1], sensor_path=row[2], ts_iso=row[3],
                    fs=row[4], scale=row[5], n=row[6], data=d, tries=row[8])

    def mark(self, acq_id: int, state: str, verdict: dict | None = None):
        with self._lock, self._db:
            self._db.execute(
                'UPDATE acq SET state=?, verdict=?, tries=tries+1 WHERE id=?',
                (state, json.dumps(verdict) if verdict else None, acq_id))

    def bump_tries(self, acq_id: int):
        with self._lock, self._db:
            self._db.execute('UPDATE acq SET tries=tries+1 WHERE id=?',
                             (acq_id,))

    def counts(self) -> dict:
        with self._lock:
            rows = self._db.execute(
                'SELECT state, count(*) FROM acq GROUP BY state').fetchall()
        return {s: n for s, n in rows}
