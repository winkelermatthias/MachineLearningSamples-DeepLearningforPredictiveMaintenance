"""Read cache: a small per-process LRU keyed by (workspace, route, params,
data_version). Ingest bumps data_version, so invalidation is implicit and
exact; the same version doubles as the HTTP ETag.
"""
from __future__ import annotations
from collections import OrderedDict
import threading

class LRU:
    def __init__(self, maxsize=256):
        self.d: OrderedDict = OrderedDict()
        self.maxsize = maxsize
        self.lock = threading.Lock()
        self.hits = self.misses = 0
    def get(self, key):
        with self.lock:
            if key in self.d:
                self.d.move_to_end(key); self.hits += 1
                return self.d[key]
            self.misses += 1
            return None
    def put(self, key, value):
        with self.lock:
            self.d[key] = value; self.d.move_to_end(key)
            while len(self.d) > self.maxsize:
                self.d.popitem(last=False)

CACHE = LRU()

def data_version(cur, wsid: str) -> int:
    row = cur.execute('SELECT data_version FROM workspace WHERE workspace_id=%s',
                      (wsid,)).fetchone()
    return row[0] if row else 0
