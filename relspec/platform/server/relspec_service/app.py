"""relspec service API. See docs/PLATFORM_PLAN.md for the design.

Run locally:  DATABASE_URL=postgresql://... uvicorn relspec_service.app:app
"""
from __future__ import annotations
import base64, datetime as dt, json, uuid
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
import pathlib
from . import auth, cache, config, db, ies, ingest, migrations, queries

app = FastAPI(title='relspec service', version='1.0')

# Cross-origin dashboards (?api= mode) authenticate with explicit Bearer
# headers, never cookies, so a permissive CORS policy leaks nothing.
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=['*'],
                   allow_methods=['*'], allow_headers=['*'])

# ------------------------------------------------------------------ plumbing
@app.on_event('startup')
def _startup():
    if config.DATABASE_URL:
        db.migrate()

@app.exception_handler(auth.AuthError)
def _auth_err(_, exc: auth.AuthError):
    return JSONResponse(status_code=exc.status, content={'detail': exc.detail})

@app.exception_handler(ingest.IngestError)
def _ing_err(_, exc: ingest.IngestError):
    headers = ({'Retry-After': str(exc.retry_after)}
               if getattr(exc, 'retry_after', None) else None)
    return JSONResponse(status_code=exc.status, content={'detail': exc.detail},
                        headers=headers)

def principal(request: Request, cur, need: str = 'full') -> auth.Principal:
    client = request.client.host if request.client else 'unknown'
    pr = auth.authenticate(request.headers.get('authorization'), cur, client)
    auth.require_scope(pr, need)
    return pr

def cached_json(request: Request, wsid: str, cur, key_extra, builder):
    """ETag + LRU wrapper for range queries. Immutable resources skip this
    and set long-lived cache headers directly."""
    ver = cache.data_version(cur, wsid)
    etag = f'"{ver}"'
    if request.headers.get('if-none-match') == etag:
        return Response(status_code=304)
    key = (wsid, request.url.path, str(key_extra), ver)
    body = cache.CACHE.get(key)
    if body is None:
        body = json.dumps(builder()).encode()
        cache.CACHE.put(key, body)
    return Response(body, media_type='application/json',
                    headers={'ETag': etag, 'Cache-Control': 'max-age=30'})

def _t01(request: Request):
    q = request.query_params
    t0 = q.get('from', '1970-01-01T00:00:00+00:00')
    t1 = q.get('to', '9999-01-01T00:00:00+00:00')
    return t0, t1

# ---------------------------------------------------------------- workspaces
@app.post('/v1/workspaces')
async def create_workspace(request: Request):
    body = await request.json()
    phrase = auth.normalize(body.get('passphrase', ''))
    auth.validate_strength(phrase)
    wsid = auth.workspace_id(phrase)
    with db.pool().connection() as conn:
        cur = conn.cursor()
        row = cur.execute('SELECT auth_hash FROM workspace WHERE workspace_id=%s',
                          (wsid,)).fetchone()
        if row:
            # same phrase -> same workspace: connecting, not creating
            if not auth.check_verifier(phrase, row[0]):
                raise auth.AuthError(409, 'workspace id collision')  # ~impossible
            created = False
        else:
            cur.execute(
                'INSERT INTO workspace (workspace_id, name, auth_hash, params) '
                'VALUES (%s,%s,%s,%s)',
                (wsid, body.get('name', ''), auth.make_verifier(phrase),
                 json.dumps(body.get('params', {}))))
            conn.commit()
            created = True
    return dict(workspace_id=wsid, created=created,
                token=auth.mint_token(wsid),
                note='The passphrase is the only key to this data. '
                     'Losing it loses the workspace; anyone holding it has '
                     'full read/write access.')

@app.get('/v1/workspaces/me')
def me(request: Request):
    with db.pool().connection() as conn:
        cur = conn.cursor()
        pr = principal(request, cur, 'read')
        row = cur.execute(
            '''SELECT name, data_version, created_at,
               (SELECT count(*) FROM sensor s WHERE s.workspace_id=w.workspace_id),
               (SELECT count(*) FROM acquisition a WHERE a.workspace_id=w.workspace_id),
               (SELECT count(*) FROM waveform wf JOIN acquisition a2
                  ON a2.acq_id=wf.acq_id WHERE a2.workspace_id=w.workspace_id)
               FROM workspace w WHERE workspace_id=%s''',
            (pr.workspace_id,)).fetchone()
    out = dict(workspace_id=pr.workspace_id, name=row[0], data_version=row[1],
               created_at=row[2].isoformat(), sensors=row[3],
               acquisitions=row[4], waveforms_stored=row[5])
    if pr.key_id is None:   # a scoped key must not escalate to a ws token
        out['token'] = auth.mint_token(pr.workspace_id)
    return out

# ----------------------------------------------------------------- hierarchy
@app.post('/v1/hierarchy:ensure')
async def ensure_hierarchy(request: Request):
    body = await request.json()
    out = []
    with db.pool().connection() as conn:
        cur = conn.cursor()
        pr = principal(request, cur)
        for pl in body.get('plants', []):
            pid = _ensure(cur, 'plant', dict(workspace_id=pr.workspace_id,
                                             name=pl['name']),
                          dict(tz=pl.get('tz', 'UTC')), 'plant_id')
            pnode = dict(name=pl['name'], plant_id=pid, assets=[])
            for a in pl.get('assets', []):
                aid = _ensure(cur, 'asset',
                              dict(workspace_id=pr.workspace_id, plant_id=pid,
                                   tag=a['tag']),
                              dict(machine_type=a.get('machine_type', ''),
                                   criticality=a.get('criticality', 3)),
                              'asset_id')
                anode = dict(tag=a['tag'], asset_id=aid, components=[])
                for c in a.get('components', []):
                    cid = _ensure(cur, 'component',
                                  dict(workspace_id=pr.workspace_id,
                                       asset_id=aid, name=c['name']),
                                  dict(kind=c.get('kind', ''),
                                       shaft_ratio=c.get('shaft_ratio', 1.0)),
                                  'component_id')
                    cnode = dict(name=c['name'], component_id=cid, sensors=[])
                    for s in c.get('sensors', []):
                        sid = _ensure(cur, 'sensor',
                                      dict(workspace_id=pr.workspace_id,
                                           component_id=cid, code=s['code']),
                                      dict(position=s.get('position', ''),
                                           units=s.get('units', 'g'),
                                           fs_nominal=s.get('fs'),
                                           fr_nominal=s.get('fr_nominal')),
                                      'sensor_id')
                        cnode['sensors'].append(dict(code=s['code'],
                                                     sensor_id=sid))
                    anode['components'].append(cnode)
                pnode['assets'].append(anode)
            out.append(pnode)
        db.bump_version(cur, pr.workspace_id)
        conn.commit()
    return dict(plants=out)

def _ensure(cur, table: str, nat: dict, extra: dict, idcol: str) -> str:
    where = ' AND '.join(f'{k}=%s' for k in nat)
    row = cur.execute(f'SELECT {idcol} FROM {table} WHERE {where}',
                      tuple(nat.values())).fetchone()
    if row: return row[0]
    new_id = uuid.uuid4().hex
    cols = {**nat, idcol: new_id,
            **{k: v for k, v in extra.items() if v is not None}}
    names = ', '.join(cols)
    ph = ', '.join(['%s']*len(cols))
    cur.execute(f'INSERT INTO {table} ({names}) VALUES ({ph})',
                tuple(cols.values()))
    return new_id

@app.get('/v1/hierarchy')
def get_hierarchy(request: Request):
    with db.pool().connection() as conn:
        cur = conn.cursor()
        pr = principal(request, cur, 'read')
        return cached_json(request, pr.workspace_id, cur, '',
                           lambda: queries.hierarchy(cur, pr.workspace_id))

# -------------------------------------------------------------------- ingest
@app.post('/v1/waveforms')
async def post_waveform(request: Request):
    body = await request.json()
    with db.pool().connection() as conn:
        pr = principal(request, conn.cursor(), 'ingest')
    ingest.reserve_rate(pr.workspace_id)
    # DSP + codec work runs off the event loop
    return await run_in_threadpool(ingest.ingest_one, pr.workspace_id, body)

@app.post('/v1/waveforms:batch')
async def post_batch(request: Request):
    """NDJSON in, NDJSON out; one waveform per line, verdict or error per
    line, processing continues past bad lines. Lines are grouped per sensor:
    order is preserved within a sensor (the chain demands it), while
    different sensors process concurrently in a small thread pool."""
    with db.pool().connection() as conn:
        pr = principal(request, conn.cursor(), 'ingest')
    raw = await request.body()
    out: dict[int, dict] = {}
    parsed: list[tuple[int, dict]] = []
    for i, line in enumerate(raw.decode().splitlines()):
        line = line.strip()
        if not line: continue
        try:
            parsed.append((i, json.loads(line)))
        except json.JSONDecodeError as e:
            out[i] = dict(line=i, error=str(e))
    ingest.reserve_rate(pr.workspace_id, len(parsed))
    groups: dict[str, list[tuple[int, dict]]] = {}
    for i, body in parsed:
        key = str(body.get('sensor_id') or body.get('sensor_path') or '')
        groups.setdefault(key, []).append((i, body))

    def run_group(items):
        for i, body in items:
            try:
                out[i] = ingest.ingest_one(pr.workspace_id, body)
            except ingest.IngestError as e:
                out[i] = dict(line=i, error=e.detail)

    def run_all():
        if not groups: return
        with ThreadPoolExecutor(max_workers=min(4, len(groups))) as ex:
            list(ex.map(run_group, groups.values()))

    await run_in_threadpool(run_all)
    return Response('\n'.join(json.dumps(out[i]) for i in sorted(out)),
                    media_type='application/x-ndjson')

# ---------------------------------------------------------------------- keys
@app.post('/v1/keys')
async def create_key(request: Request):
    body = await request.json()
    scope = body.get('scope', 'read')
    if scope not in auth.SCOPES:
        raise auth.AuthError(400, f'scope must be one of {list(auth.SCOPES)}')
    with db.pool().connection() as conn:
        cur = conn.cursor()
        pr = principal(request, cur, 'full')
        key_id = uuid.uuid4().hex
        cur.execute(
            'INSERT INTO api_key (key_id, workspace_id, scope, label) '
            'VALUES (%s,%s,%s,%s)',
            (key_id, pr.workspace_id, scope, str(body.get('label') or '')))
        conn.commit()
    return dict(key_id=key_id, scope=scope,
                token=auth.mint_key_token(pr.workspace_id, key_id, scope),
                note='store the token now; it is not retrievable later')

@app.get('/v1/keys')
def list_keys(request: Request):
    with db.pool().connection() as conn:
        cur = conn.cursor()
        pr = principal(request, cur, 'full')
        rows = cur.execute(
            'SELECT key_id, scope, label, created, revoked FROM api_key '
            'WHERE workspace_id=%s ORDER BY created', (pr.workspace_id,)).fetchall()
    return [dict(key_id=k, scope=s, label=l, created=c.isoformat(), revoked=r)
            for k, s, l, c, r in rows]

@app.delete('/v1/keys/{key_id}')
def revoke_key(key_id: str, request: Request):
    with db.pool().connection() as conn:
        cur = conn.cursor()
        pr = principal(request, cur, 'full')
        row = cur.execute(
            'UPDATE api_key SET revoked=TRUE '
            'WHERE key_id=%s AND workspace_id=%s RETURNING key_id',
            (key_id, pr.workspace_id)).fetchone()
        conn.commit()
    if not row:
        return JSONResponse(status_code=404, content={'detail': 'no such key'})
    return dict(key_id=key_id, revoked=True)

# --------------------------------------------------------------------- reads
@app.get('/v1/fleet/overview')
def overview(request: Request):
    with db.pool().connection() as conn:
        cur = conn.cursor()
        pr = principal(request, cur, 'read')
        return cached_json(request, pr.workspace_id, cur, '',
                           lambda: queries.fleet_overview(cur, pr.workspace_id))

@app.get('/v1/sensors/{sensor_id}/trend')
def trend(sensor_id: str, request: Request):
    t0, t1 = _t01(request)
    with db.pool().connection() as conn:
        cur = conn.cursor()
        pr = principal(request, cur, 'read')
        return cached_json(request, pr.workspace_id, cur, (sensor_id, t0, t1),
                           lambda: queries.trend(cur, pr.workspace_id,
                                                 sensor_id, t0, t1))

@app.get('/v1/sensors/{sensor_id}/health')
def sensor_health(sensor_id: str, request: Request):
    t0, t1 = _t01(request)
    with db.pool().connection() as conn:
        cur = conn.cursor()
        pr = principal(request, cur, 'read')
        return cached_json(request, pr.workspace_id, cur,
                           ('health', sensor_id, t0, t1),
                           lambda: queries.sensor_health(
                               cur, pr.workspace_id, sensor_id, t0, t1))

@app.get('/v1/fleet/health')
def fleet_health(request: Request):
    with db.pool().connection() as conn:
        cur = conn.cursor()
        pr = principal(request, cur, 'read')
        return cached_json(request, pr.workspace_id, cur, ('fleet_health',),
                           lambda: queries.fleet_health(cur, pr.workspace_id))

@app.get('/v1/sensors/{sensor_id}/frames')
def frames(sensor_id: str, request: Request, limit: int = 500):
    t0, t1 = _t01(request)
    limit = min(limit, 2000)
    with db.pool().connection() as conn:
        cur = conn.cursor()
        pr = principal(request, cur, 'read')
        return cached_json(request, pr.workspace_id, cur,
                           (sensor_id, t0, t1, limit),
                           lambda: queries.frames(cur, pr.workspace_id,
                                                  sensor_id, t0, t1, limit))

@app.get('/v1/acquisitions/{acq_id}/spectra')
def spectra(acq_id: str, request: Request):
    with db.pool().connection() as conn:
        cur = conn.cursor()
        pr = principal(request, cur, 'read')
        out = queries.spectra(cur, pr.workspace_id, acq_id)
    if out is None:
        return JSONResponse(status_code=404, content={'detail': 'not found'})
    return Response(json.dumps(out), media_type='application/json',
                    headers={'Cache-Control': 'max-age=31536000, immutable'})

@app.get('/v1/sensors/{sensor_id}/explorer-bundle')
def bundle(sensor_id: str, request: Request, limit: int = 400):
    limit = min(limit, 1500)
    with db.pool().connection() as conn:
        cur = conn.cursor()
        pr = principal(request, cur, 'read')
        resp = cached_json(request, pr.workspace_id, cur, (sensor_id, limit),
                           lambda: _bundle_or_404(cur, pr.workspace_id,
                                                  sensor_id, limit))
        return resp

def _bundle_or_404(cur, wsid, sensor_id, limit):
    out = queries.explorer_bundle(cur, wsid, sensor_id, limit)
    return out if out is not None else {'detail': 'not found'}

@app.get('/v1/events')
def events(request: Request, limit: int = 500):
    t0, t1 = _t01(request)
    with db.pool().connection() as conn:
        cur = conn.cursor()
        pr = principal(request, cur, 'read')
        return cached_json(request, pr.workspace_id, cur, (t0, t1, limit),
                           lambda: queries.events(cur, pr.workspace_id,
                                                  t0, t1, limit))

@app.get('/v1/events/stream')
async def events_stream(request: Request):
    """SSE: live acquisition/gate events for this workspace. Heartbeat
    comments every 15 s; reconcile with GET /v1/events on reconnect."""
    from fastapi.responses import StreamingResponse
    from . import events_stream as es
    with db.pool().connection() as conn:
        pr = principal(request, conn.cursor())
    q = await es.BROKER.subscribe(pr.workspace_id)
    return StreamingResponse(es.sse_generator(pr.workspace_id, q),
                             media_type='text/event-stream',
                             headers={'cache-control': 'no-cache',
                                      'x-accel-buffering': 'no'})

# ---------------------------------------------------------------- migrations
@app.post('/v1/migrations')
async def create_migration(request: Request):
    body = await request.json()
    items = body.get('manifest') or []
    if not isinstance(items, list) or not items:
        return JSONResponse({'detail': 'manifest must be a non-empty list'},
                            status_code=400)
    with db.pool().connection() as conn:
        cur = conn.cursor()
        pr = principal(request, cur, 'ingest')
        report = migrations.validate(cur, pr.workspace_id, items)
        if body.get('dry_run'):
            ok = sum(1 for r in report if r['ok'])
            return dict(dry_run=True, total=len(report), ok=ok,
                        errors=len(report) - ok, items=report)
        job_id = migrations.create(cur, pr.workspace_id, items, report)
        conn.commit()
    migrations.start_worker(pr.workspace_id, job_id)
    return dict(job_id=job_id, total=len(items),
                invalid=sum(1 for r in report if not r['ok']))

@app.get('/v1/migrations/{job_id}')
def migration_status(job_id: str, request: Request):
    with db.pool().connection() as conn:
        cur = conn.cursor()
        pr = principal(request, cur, 'read')
        out = migrations.status(cur, pr.workspace_id, job_id)
    return out if out else JSONResponse({'detail': 'not found'},
                                        status_code=404)

@app.post('/v1/migrations/{job_id}:pause')
def migration_pause(job_id: str, request: Request):
    with db.pool().connection() as conn:
        cur = conn.cursor()
        pr = principal(request, cur, 'ingest')
        ok = migrations.set_state(cur, pr.workspace_id, job_id, 'paused')
        conn.commit()
    return {'paused': ok}

@app.post('/v1/migrations/{job_id}:resume')
async def migration_resume(job_id: str, request: Request):
    with db.pool().connection() as conn:
        cur = conn.cursor()
        pr = principal(request, cur, 'ingest')
        ok = migrations.set_state(cur, pr.workspace_id, job_id, 'running')
        conn.commit()
    if ok:
        migrations.start_worker(pr.workspace_id, job_id)
    return {'resumed': ok}

@app.get('/v1/acquisitions/{acq_id}/waveform')
def waveform(acq_id: str, request: Request):
    with db.pool().connection() as conn:
        cur = conn.cursor()
        pr = principal(request, cur, 'read')
        out = queries.waveform(cur, pr.workspace_id, acq_id)
    if out is None:
        return JSONResponse(status_code=404,
                            content={'detail': 'no stored waveform for this acquisition'})
    return Response(json.dumps(out), media_type='application/json',
                    headers={'Cache-Control': 'max-age=31536000, immutable'})

@app.get('/v1/acquisitions/{acq_id}/ies')
def ies_view(acq_id: str, request: Request):
    """Fast-SC Improved Envelope Spectrum — a read-only second opinion
    computed from the stored raw waveform. Codec-only acquisitions 404
    honestly: the codec payload cannot reconstruct the raw signal."""
    with db.pool().connection() as conn:
        cur = conn.cursor()
        pr = principal(request, cur, 'read')
        own = cur.execute(
            'SELECT fr_est FROM acquisition WHERE acq_id=%s AND workspace_id=%s',
            (acq_id, pr.workspace_id)).fetchone()
        if not own:
            return JSONResponse(status_code=404, content={'detail': 'not found'})
        etag = f'"ies-{acq_id}"'   # waveforms are immutable, so the id is the tag
        if request.headers.get('if-none-match') == etag:
            return Response(status_code=304)
        row = cur.execute(
            'SELECT encoding, scale, fs, data FROM waveform WHERE acq_id=%s',
            (acq_id,)).fetchone()
    if row is None:
        return JSONResponse(status_code=404, content={'detail':
            'no stored raw waveform for this acquisition (codec frames and '
            'features only) — the IES needs the raw signal'})
    key = ('ies', acq_id)
    body = cache.CACHE.get(key)
    if body is None:
        try:
            x = ies.decode_stored(row[0], row[1], row[3])
            out = ies.compute(x, float(row[2]), own[0])
        except ies.IesError as e:
            return JSONResponse(status_code=e.status, content={'detail': e.detail})
        out['acq_id'] = acq_id
        body = json.dumps(out).encode()
        cache.CACHE.put(key, body)
    return Response(body, media_type='application/json',
                    headers={'ETag': etag,
                             'Cache-Control': 'max-age=31536000, immutable'})

@app.get('/v1/meta')
def meta():
    from relspec.pipeline import CENTERS, ENV_CENTERS
    return dict(centers=[round(float(c), 4) for c in CENTERS],
                env_centers=[round(float(c), 4) for c in ENV_CENTERS],
                q_db=0.5, db_off=-128.0)

@app.get('/health')
def health():
    with db.pool().connection() as conn:
        conn.execute('SELECT 1')
    return dict(ok=True)

# static explorer app (nginx does this in compose; harmless duplicate here)
_web = pathlib.Path(__file__).resolve().parents[2] / 'web' / 'app'
if _web.is_dir():
    app.mount('/app', StaticFiles(directory=str(_web), html=True), name='app')
