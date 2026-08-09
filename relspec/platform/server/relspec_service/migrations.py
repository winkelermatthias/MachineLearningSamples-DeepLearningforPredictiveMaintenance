"""Resumable bulk-ingest ("migration") jobs.

Dry-run first is the contract: validate the ENTIRE manifest — sensors
resolve, timestamps parse, fs in bounds, per-sensor order strictly
increasing both within the manifest and against history — and return a
per-item report without writing a byte. A real job stores every item,
then a background worker walks them in manifest order (callers submit
per-sensor oldest-first; validation enforced it) through the ordinary
ingest path. Item state is the checkpoint; client_ref defaults to
job:ordinal, so pause/crash + resume is exactly-once by construction.
"""
from __future__ import annotations
import asyncio, datetime as dt, json, logging, uuid
from . import config, db, ingest

log = logging.getLogger('migrations')

_workers: dict[str, asyncio.Task] = {}


def _parse_ts(raw) -> dt.datetime | None:
    try:
        ts = dt.datetime.fromisoformat(str(raw).replace('Z', '+00:00'))
        return ts if ts.tzinfo else ts.replace(tzinfo=dt.timezone.utc)
    except Exception:
        return None


def validate(cur, wsid: str, items: list[dict]) -> list[dict]:
    """Per-item report; item i valid iff report[i]['ok']."""
    report = []
    last_by_sensor: dict[str, dt.datetime] = {}
    for i, body in enumerate(items):
        errs = []
        ref = str(body.get('sensor_id') or body.get('sensor_path') or '')
        sensor = None
        if not ref:
            errs.append('missing sensor_id/sensor_path')
        else:
            try:
                sensor = ingest._resolve_sensor(cur, wsid, body)
            except ingest.IngestError as e:
                errs.append(e.detail)
        ts = _parse_ts(body.get('ts'))
        if ts is None:
            errs.append(f'bad ts {body.get("ts")!r}')
        fs = float(body.get('fs') or 0)
        if not (config.FS_MIN <= fs <= config.FS_MAX):
            errs.append(f'fs {fs:g} outside [{config.FS_MIN:g}, {config.FS_MAX:g}]')
        if not (body.get('data_b64') or body.get('values')):
            errs.append('no data_b64/values')
        if sensor and ts:
            sid = sensor['sensor_id']
            if sid not in last_by_sensor:
                row = cur.execute(
                    'SELECT max(ts) FROM acquisition WHERE sensor_id=%s',
                    (sid,)).fetchone()
                last_by_sensor[sid] = row[0]
            prev = last_by_sensor[sid]
            if prev is not None and ts <= prev:
                errs.append(f'out-of-order: sensor already has data at '
                            f'{prev.isoformat()}')
            else:
                last_by_sensor[sid] = ts
        report.append(dict(ordinal=i, sensor=ref,
                           ts=ts.isoformat() if ts else None,
                           ok=not errs, errors=errs))
    return report


def create(cur, wsid: str, items: list[dict], report: list[dict]) -> str:
    job_id = uuid.uuid4().hex[:12]
    cur.execute('INSERT INTO migration_job (job_id, workspace_id, total) '
                'VALUES (%s,%s,%s)', (job_id, wsid, len(items)))
    for i, (body, rep) in enumerate(zip(items, report)):
        body = dict(body)
        body.setdefault('client_ref', f'mig:{job_id}:{i}')
        cur.execute(
            '''INSERT INTO migration_item (job_id, ordinal, sensor_ref, ts,
               body, state, error)
               VALUES (%s,%s,%s,%s,%s,%s,%s)''',
            (job_id, i, rep['sensor'], rep['ts'], json.dumps(body),
             'pending' if rep['ok'] else 'error',
             None if rep['ok'] else '; '.join(rep['errors'])))
        if not rep['ok']:
            cur.execute('UPDATE migration_job SET errors=errors+1 '
                        'WHERE job_id=%s', (job_id,))
    return job_id


def status(cur, wsid: str, job_id: str) -> dict | None:
    row = cur.execute(
        'SELECT state, created, total, done, errors FROM migration_job '
        'WHERE job_id=%s AND workspace_id=%s', (job_id, wsid)).fetchone()
    if not row:
        return None
    bad = cur.execute(
        'SELECT ordinal, sensor_ref, error FROM migration_item '
        'WHERE job_id=%s AND state=%s ORDER BY ordinal LIMIT 20',
        (job_id, 'error')).fetchall()
    return dict(job_id=job_id, state=row[0], created=row[1].isoformat(),
                total=row[2], done=row[3], errors=row[4],
                failed_items=[dict(ordinal=o, sensor=s, error=e)
                              for o, s, e in bad])


def set_state(cur, wsid: str, job_id: str, state: str) -> bool:
    return cur.execute(
        'UPDATE migration_job SET state=%s '
        'WHERE job_id=%s AND workspace_id=%s AND state NOT IN (%s, %s)',
        (state, job_id, wsid, 'done', 'failed')).rowcount > 0


def start_worker(wsid: str, job_id: str):
    """Idempotent: one worker per job per process."""
    t = _workers.get(job_id)
    if t and not t.done():
        return
    _workers[job_id] = asyncio.get_running_loop().create_task(
        _run(wsid, job_id))


async def _run(wsid: str, job_id: str):
    log.info('migration %s: worker started', job_id)
    while True:
        with db.pool().connection() as conn:
            cur = conn.cursor()
            state, = cur.execute(
                'SELECT state FROM migration_job WHERE job_id=%s',
                (job_id,)).fetchone()
            if state != 'running':
                log.info('migration %s: %s; worker stopping', job_id, state)
                return
            item = cur.execute(
                'SELECT ordinal, body FROM migration_item '
                'WHERE job_id=%s AND state=%s ORDER BY ordinal LIMIT 1',
                (job_id, 'pending')).fetchone()
        if item is None:
            with db.pool().connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    'UPDATE migration_job SET state=%s WHERE job_id=%s',
                    ('done', job_id))
                conn.commit()
            log.info('migration %s: done', job_id)
            return
        ordinal, body = item
        body = body if isinstance(body, dict) else json.loads(body)
        try:
            verdict = await asyncio.to_thread(ingest.ingest_one, wsid, body)
            st, err = 'done', None
            _ = verdict
        except ingest.IngestError as e:
            st, err = 'error', e.detail
        except Exception as e:               # keep the job alive; record it
            st, err = 'error', f'{type(e).__name__}: {e}'
        with db.pool().connection() as conn:
            cur = conn.cursor()
            cur.execute('UPDATE migration_item SET state=%s, error=%s '
                        'WHERE job_id=%s AND ordinal=%s',
                        (st, err, job_id, ordinal))
            cur.execute(
                'UPDATE migration_job SET done=done+%s, errors=errors+%s '
                'WHERE job_id=%s',
                (1 if st == 'done' else 0, 1 if st == 'error' else 0, job_id))
            conn.commit()
