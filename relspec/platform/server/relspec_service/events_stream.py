"""Live events: one LISTEN connection per process fans pg_notify
payloads out to any number of SSE subscribers, filtered by workspace.

NOTIFY is sent inside the ingest transaction, so a subscriber hears
about an acquisition exactly when it becomes visible to reads — never
before, never instead. The listener reconnects with backoff; missed
notifications during a reconnect are acceptable by design (SSE is a
freshness channel, the tables are the record; clients reconcile with an
ordinary GET on reconnect, which Last-Event-ID makes cheap).
"""
from __future__ import annotations
import asyncio, json, logging
import psycopg
from . import config

log = logging.getLogger('events')

CHANNEL = 'relspec_events'


class Broker:
    def __init__(self):
        self._subs: set[tuple[str, asyncio.Queue]] = set()
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def subscribe(self, wsid: str) -> asyncio.Queue:
        async with self._lock:
            if self._task is None or self._task.done():
                self._task = asyncio.create_task(self._listen())
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subs.add((wsid, q))
        return q

    def unsubscribe(self, wsid: str, q: asyncio.Queue):
        self._subs.discard((wsid, q))

    async def _listen(self):
        backoff = 0.5
        while True:
            try:
                conn = await psycopg.AsyncConnection.connect(
                    config.DATABASE_URL, autocommit=True)
                await conn.execute(f'LISTEN {CHANNEL}')
                log.info('event listener up')
                backoff = 0.5
                async for n in conn.notifies():
                    try:
                        ws = json.loads(n.payload).get('ws')
                    except json.JSONDecodeError:
                        continue
                    for wsid, q in list(self._subs):
                        if wsid != ws:
                            continue
                        try:
                            q.put_nowait(n.payload)
                        except asyncio.QueueFull:
                            # slow consumer: drop oldest, keep the stream live
                            try: q.get_nowait()
                            except asyncio.QueueEmpty: pass
                            q.put_nowait(n.payload)
            except Exception as e:
                log.warning('event listener down (%s); retrying in %.1fs',
                            type(e).__name__, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 15.0)


BROKER = Broker()


async def sse_generator(wsid: str, q: asyncio.Queue):
    yield ': connected\n\n'
    seq = 0
    try:
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=15.0)
            except asyncio.TimeoutError:
                yield ': hb\n\n'
                continue
            seq += 1
            yield f'id: {seq}\ndata: {item}\n\n'
    finally:
        BROKER.unsubscribe(wsid, q)
