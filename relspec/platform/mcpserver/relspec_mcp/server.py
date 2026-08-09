"""relspec MCP server: Claude as the ingestion analyst.

Thin by design — every tool is one API call plus honest errors; every
policy decision (significance, storage cadence, codec state) stays on the
server. Run over stdio for local Claude Code / Desktop:

    RELSPEC_API=http://127.0.0.1:8000 python3 -m relspec_mcp.server

or containerised with streamable-http (see docker-compose.yml).
"""
from __future__ import annotations
import base64, os
import httpx
from mcp.server.mcpserver import MCPServer

API = os.environ.get('RELSPEC_API', 'http://127.0.0.1:8000').rstrip('/')

mcp = MCPServer('relspec', instructions='Condition-monitoring ingestion service: create/connect a passphrase workspace, ensure the plant/asset/component/sensor hierarchy, then submit time waveforms oldest-first per sensor.')
_state = {'token': None}

def _client() -> httpx.Client:
    return httpx.Client(base_url=API, timeout=180)

def _hdr() -> dict:
    if not _state['token']:
        raise RuntimeError('not connected: call relspec_connect or '
                           'relspec_create_workspace first')
    return {'Authorization': 'Bearer ' + _state['token']}

def _call(method: str, path: str, **kw):
    with _client() as c:
        r = c.request(method, path, **kw)
    if r.status_code >= 400:
        try: detail = r.json().get('detail', r.text)
        except Exception: detail = r.text
        raise RuntimeError(f'{method} {path} -> {r.status_code}: {detail}')
    return r.json() if 'json' in r.headers.get('content-type', '') else r.text

TIERS = ('healthy', 'monitor', 'alert', 'critical')
FP_Q_DB = 0.5           # dB per fingerprint band count (server Q_DB)

def _tier(h) -> str:
    return 'learning' if h is None else TIERS[min(int(h), len(TIERS) - 1)]

def _sensor_id(sensor_path: str) -> str:
    ov = _call('GET', '/v1/fleet/overview', headers=_hdr())
    match = [s for s in ov if s['path'] == sensor_path]
    if not match:
        raise RuntimeError(f'no sensor at path {sensor_path!r}; '
                           f'known: {[s["path"] for s in ov]}')
    return match[0]['sensor_id']

@mcp.tool()
def relspec_create_workspace(passphrase: str, name: str = '') -> dict:
    """Create (or reconnect to) a workspace. The passphrase is the only key:
    it locates the workspace and unlocks it, for writing and for reading.
    Have the USER choose and keep it; never invent one silently, never write
    it into files or logs."""
    out = _call('POST', '/v1/workspaces',
                json=dict(passphrase=passphrase, name=name))
    _state['token'] = out.pop('token')
    return out

@mcp.tool()
def relspec_connect(passphrase: str) -> dict:
    """Connect to an existing workspace with its passphrase."""
    out = _call('POST', '/v1/workspaces', json=dict(passphrase=passphrase))
    _state['token'] = out.pop('token')
    out['connected'] = True
    return out

@mcp.tool()
def relspec_ensure_hierarchy(tree: dict) -> dict:
    """Idempotently create plant -> asset -> component -> sensor structure.
    tree = {"plants":[{"name":..., "assets":[{"tag":..., "components":
    [{"name":..., "kind":..., "sensors":[{"code":..., "units":"g",
    "fs":12000, "fr_nominal":29.5}]}]}]}]}. Re-running with the same names
    returns the same ids."""
    return _call('POST', '/v1/hierarchy:ensure', json=tree, headers=_hdr())

@mcp.tool()
def relspec_get_hierarchy() -> dict:
    """The workspace's full hierarchy with ids."""
    return _call('GET', '/v1/hierarchy', headers=_hdr())

@mcp.tool()
def relspec_submit_waveform(sensor_path: str, ts: str, fs: float,
                            data_b64: str, units: str = 'g',
                            encoding: str = 'float32',
                            scale: float | None = None,
                            client_ref: str | None = None) -> dict:
    """Submit one time waveform. sensor_path = plant/asset/component/sensor.
    ts ISO-8601; data_b64 = base64 of little-endian float32 (or int16 with
    scale). Submit each sensor's history OLDEST FIRST. The server keeps
    features, spectra and patterns for every submission and decides on its
    own significance policy whether the raw waveform is stored — report the
    verdict (waveform_stored, sig_reason) honestly."""
    body = dict(sensor_path=sensor_path, ts=ts, fs=fs, units=units,
                encoding=encoding, data_b64=data_b64)
    if scale is not None: body['scale'] = scale
    if client_ref: body['client_ref'] = client_ref
    return _call('POST', '/v1/waveforms', json=body, headers=_hdr())

@mcp.tool()
def relspec_submit_batch(ndjson: str) -> str:
    """Submit many waveforms: one JSON object per line, same fields as
    relspec_submit_waveform. Returns one verdict per line; lines with errors
    do not stop the batch. Keep batches at or under 200 lines and in time
    order per sensor."""
    with _client() as c:
        r = c.post('/v1/waveforms:batch', content=ndjson, headers=_hdr())
    return r.text

@mcp.tool()
def relspec_get_status(sensor_path: str = '') -> dict:
    """Workspace summary plus the fleet overview (per-sensor latest state);
    filter to one sensor_path if given."""
    me = _call('GET', '/v1/workspaces/me', headers=_hdr())
    me.pop('token', None)
    ov = _call('GET', '/v1/fleet/overview', headers=_hdr())
    if sensor_path:
        ov = [s for s in ov if s['path'] == sensor_path]
    return dict(workspace=me, sensors=ov)

@mcp.tool()
def relspec_get_trend(sensor_path: str, date_from: str = '1970-01-01',
                      date_to: str = '9999-01-01') -> list:
    """Daily velocity / acceleration / envelope trend for one sensor —
    use it after migrating to verify counts and level ranges against the
    source database."""
    return _call('GET', f'/v1/sensors/{_sensor_id(sensor_path)}/trend',
                 params={'from': date_from, 'to': date_to}, headers=_hdr())

@mcp.tool()
def relspec_get_health(sensor_path: str) -> dict:
    """Latest health verdict for one sensor: tier (healthy / monitor /
    alert / critical, or 'learning' before the model has one), the driver
    signals explaining WHY, and a compact trend of the last 30
    acquisitions (ts, tier, vel_rms)."""
    sid = _sensor_id(sensor_path)
    rows = _call('GET', f'/v1/sensors/{sid}/health', headers=_hdr())
    if not rows:
        return dict(sensor_path=sensor_path, tier='learning',
                    ts=None, drivers=[], trend=[])
    last = rows[-1]
    trend = [dict(ts=r['ts'], health=_tier(r['health']),
                  vel_rms=None if r['vel_rms'] is None
                  else round(r['vel_rms'], 3))
             for r in rows[-30:]]
    return dict(sensor_path=sensor_path, tier=_tier(last['health']),
                ts=last['ts'], drivers=last.get('drivers', []), trend=trend)

@mcp.tool()
def relspec_get_fleet_health() -> dict:
    """Fleet health rollup, deliberately small: each asset's worst-sensor
    tier plus per-plant tier counts (a distribution, not just the worst).
    Drill into a sensor with relspec_get_health."""
    fh = _call('GET', '/v1/fleet/health', headers=_hdr())
    assets = [dict(asset=a['asset'], plant=a['plant'],
                   sensors=a['sensors'], tier=_tier(a['health']))
              for a in fh.get('assets', [])]
    plants = [dict(plant=p['plant'],
                   **{TIERS[i]: p['counts'][i] for i in range(len(TIERS))},
                   learning=p['learning'])
              for p in fh.get('plants', [])]
    return dict(assets=assets, plants=plants)

@mcp.tool()
def relspec_get_pattern_ledger(sensor_path: str, limit: int = 30) -> dict:
    """Ledger of the spectral pattern tracks seen in the last `limit`
    frames of one sensor: per track its rail, kind, key (order), latest
    z-score against its learned baseline (None while learning), latest
    energy share and observation count — sorted most anomalous (|z|)
    first. Never returns raw spectra."""
    sid = _sensor_id(sensor_path)
    b = _call('GET', f'/v1/sensors/{sid}/explorer-bundle',
              params={'limit': limit}, headers=_hdr())
    if 'frames' not in b:
        raise RuntimeError(f'no data yet for sensor {sensor_path!r}')
    tracks: dict[tuple, dict] = {}
    for fr in b['frames']:                     # oldest -> newest
        for rail in ('acc', 'env'):
            for p in fr.get(rail + '_pat', []):
                t = tracks.setdefault((rail, p['t']), dict(
                    rail=rail, track_id=p['t'], n_obs=0))
                t['n_obs'] += 1
                t.update(kind=p['kind'], key=round(p['key'], 3),
                         share=round(p['share'], 4), last_ts=fr['ts'],
                         _acq=fr['acq_id'])
    # z lives on the pattern rows of the spectra resource, not in the
    # bundle: fetch it for each track's latest frame (usually one call,
    # since live tracks share the newest acquisition)
    zmap = {}
    for acq in {t['_acq'] for t in tracks.values()}:
        sp = _call('GET', f'/v1/acquisitions/{acq}/spectra', headers=_hdr())
        for rail in ('acc', 'env'):
            for p in sp.get(rail, {}).get('patterns', []):
                zmap[(acq, rail, p['t'])] = p.get('z')
    out = []
    for (rail, tid), t in tracks.items():
        z = zmap.get((t.pop('_acq'), rail, tid))
        t['z'] = None if z is None else round(z, 2)
        out.append(t)
    out.sort(key=lambda t: (t['z'] is None,
                            -abs(t['z']) if t['z'] is not None else 0))
    return dict(sensor_path=sensor_path, frames_scanned=len(b['frames']),
                tracks=out)

@mcp.tool()
def relspec_get_spectrum(sensor_path: str, rail: str = 'env') -> dict:
    """Latest spectrum of one rail ('env' envelope or 'acc' acceleration)
    for one sensor, downsampled to at most 128 (order, dB) points with the
    top 8 peaks labelled by order, plus the detected patterns with their z.
    The full-resolution array is never returned."""
    if rail not in ('acc', 'env'):
        raise RuntimeError("rail must be 'acc' or 'env'")
    sid = _sensor_id(sensor_path)
    frames = _call('GET', f'/v1/sensors/{sid}/frames',
                   params={'limit': 1}, headers=_hdr())
    if not frames:
        raise RuntimeError(f'no frames yet for sensor {sensor_path!r}')
    fr = frames[0]
    sp = _call('GET', f"/v1/acquisitions/{fr['acq_id']}/spectra",
               headers=_hdr())
    if rail not in sp:
        raise RuntimeError(f'no {rail} spectrum stored on the latest frame')
    meta = _call('GET', '/v1/meta')
    centers = meta['centers'] if rail == 'acc' else meta['env_centers']
    u8 = base64.b64decode(sp[rail]['o'])
    n = min(len(u8), len(centers))
    db = [u8[i] * meta['q_db'] + meta['db_off'] for i in range(n)]
    # peak-preserving downsample: <=128 chunks, keep each chunk's maximum
    step = max(1, -(-n // 128))
    points = []
    for i0 in range(0, n, step):
        j = max(range(i0, min(i0 + step, n)), key=lambda i: db[i])
        points.append([round(centers[j], 3), round(db[j], 1)])
    # top 8 local maxima of the full-resolution curve, labelled by order
    peaks = sorted((i for i in range(1, n - 1)
                    if db[i] > db[i - 1] and db[i] >= db[i + 1]),
                   key=lambda i: -db[i])[:8]
    pats = [dict(track_id=p['t'], kind=p['kind'], key=round(p['key'], 3),
                 f0=round(p['f0'], 3), share=round(p['share'], 4),
                 z=None if p.get('z') is None else round(p['z'], 2))
            for p in sp[rail].get('patterns', [])]
    return dict(sensor_path=sensor_path, rail=rail, ts=fr['ts'],
                acq_id=fr['acq_id'], n_bins_full=n,
                points=points,
                peaks=[dict(order=round(centers[i], 3),
                            db=round(db[i], 1)) for i in peaks],
                patterns=pats)

@mcp.tool()
def relspec_compare_frames(sensor_path: str, n_back: int = 10) -> dict:
    """Compare the latest acquisition against the one n_back frames
    earlier using the 30-byte banded fingerprint every acquisition
    carries: per-band delta dB (16 acceleration + 8 envelope bands),
    vel_rms delta, and the health tier then vs now."""
    sid = _sensor_id(sensor_path)
    rows = _call('GET', f'/v1/sensors/{sid}/health', headers=_hdr())
    if len(rows) < 2:
        raise RuntimeError('need at least 2 acquisitions to compare')
    now = rows[-1]
    then = rows[max(0, len(rows) - 1 - n_back)]
    fp0, fp1 = then.get('fp'), now.get('fp')
    if not fp0 or not fp1:
        raise RuntimeError('fingerprint missing on one of the frames')
    acc = [round((b1 - b0) * FP_Q_DB, 1)
           for b0, b1 in zip(fp0['acc_bands'], fp1['acc_bands'])]
    env = [round((b1 - b0) * FP_Q_DB, 1)
           for b0, b1 in zip(fp0['env_bands'], fp1['env_bands'])]
    return dict(sensor_path=sensor_path,
                frames_apart=len(rows) - 1 - max(0, len(rows) - 1 - n_back),
                then=dict(ts=then['ts'], tier=_tier(then['health']),
                          vel_rms=then['vel_rms']),
                now=dict(ts=now['ts'], tier=_tier(now['health']),
                         vel_rms=now['vel_rms']),
                vel_rms_delta=round((now['vel_rms'] or 0.0)
                                    - (then['vel_rms'] or 0.0), 3),
                acc_band_delta_db=acc, env_band_delta_db=env)

@mcp.tool()
def relspec_migrate_dry_run(manifest: list) -> dict:
    """Validate a migration manifest WITHOUT writing anything. Each item
    is a waveform body: {sensor_path|sensor_id, ts ISO-8601, fs, units,
    encoding float32|int16(+scale), data_b64, client_ref?}, oldest-first
    per sensor. Returns counts plus every failing item with its errors —
    fix those, dry-run again, then relspec_migrate_start."""
    out = _call('POST', '/v1/migrations',
                json=dict(manifest=manifest, dry_run=True), headers=_hdr())
    bad = [r for r in out.get('items', []) if not r['ok']]
    return dict(dry_run=True, total=out['total'], ok=out['ok'],
                errors=out['errors'], failed_items=bad[:50])

@mcp.tool()
def relspec_migrate_start(manifest: list) -> dict:
    """Start a real, resumable migration job for a manifest that already
    passed relspec_migrate_dry_run (same item shape, oldest-first per
    sensor). Returns the job_id; poll relspec_migrate_status until state
    is 'done'. Items are checkpointed server-side, so a crash or pause
    loses nothing — relspec_migrate_resume picks up exactly where it
    stopped."""
    out = _call('POST', '/v1/migrations',
                json=dict(manifest=manifest, dry_run=False), headers=_hdr())
    out['note'] = "poll relspec_migrate_status until state is 'done'"
    return out

@mcp.tool()
def relspec_migrate_status(job_id: str) -> dict:
    """Progress of a migration job: state (running/paused/done/failed),
    done/errors/total counts and up to 20 failed items with reasons."""
    return _call('GET', f'/v1/migrations/{job_id}', headers=_hdr())

@mcp.tool()
def relspec_migrate_resume(job_id: str) -> dict:
    """Resume a paused or interrupted migration job; already-ingested
    items are never redone (exactly-once by checkpoint)."""
    return _call('POST', f'/v1/migrations/{job_id}:resume', headers=_hdr())

@mcp.tool()
def relspec_mint_key(scope: str, label: str = '') -> dict:
    """Mint a scoped API key for this workspace: scope 'read' (queries
    only), 'ingest' (submit waveforms only) or 'full'. Requires a full
    workspace connection. The returned token is shown ONCE and cannot be
    retrieved later — hand it to the user to store immediately."""
    out = _call('POST', '/v1/keys', json=dict(scope=scope, label=label),
                headers=_hdr())
    out['warning'] = ('this token is shown ONCE and is not retrievable '
                      'later — store it now; revoke via the API if leaked')
    return out

@mcp.prompt()
def ingest_migration() -> str:
    """Migrate a customer's waveform database into a relspec workspace."""
    return (
        "You are migrating a customer's vibration time-waveform database "
        "into a relspec workspace. Work in this order and confirm each "
        "stage with the user before executing it.\n"
        "1. INSPECT their database read-only. Find the tables holding "
        "waveforms, sample rates, timestamps, units, and whatever "
        "identifies the measurement point. Show the user what you found "
        "and how many waveforms / sensors / what date range it covers.\n"
        "2. PROPOSE the hierarchy mapping: plant -> asset -> component -> "
        "sensor, built from their naming. Never invent structure they "
        "don't have — a flat plant with assets is fine. Show the tree; "
        "adjust until they approve; then call relspec_ensure_hierarchy "
        "once.\n"
        "3. DRY-RUN FIRST: build the manifest oldest-first per sensor "
        "(order matters: baselines, codec chains and weekly storage "
        "budgets accrue in time order, and the server rejects "
        "out-of-order timestamps per sensor) and call "
        "relspec_migrate_dry_run. It validates every item — sensor "
        "resolves, ts parses, fs in bounds, per-sensor order — without "
        "writing a byte. Fix each reported per-item error at the source "
        "and dry-run again until errors is 0. Use a stable client_ref "
        "(their primary key) so re-runs are idempotent.\n"
        "4. START the job with relspec_migrate_start on the clean "
        "manifest, then POLL relspec_migrate_status until state is "
        "'done', reporting done/errors/total progress along the way. "
        "After ANY interruption — pause, crash, network drop, a new "
        "session — call relspec_migrate_status and then "
        "relspec_migrate_resume: items are checkpointed server-side, so "
        "resuming is exactly-once and nothing is redone or lost. "
        "(relspec_submit_batch remains available for small ad-hoc loads "
        "of <= 200 lines, same oldest-first rule.)\n"
        "5. Do NOT decide which waveforms deserve full storage — submit "
        "everything in range; the server keeps every waveform's features, "
        "spectra and patterns, and stores raw waveforms on its own "
        "significance policy (weekly baseline + change events). Report "
        "the verdicts honestly, including how many raw waveforms were "
        "kept and how many change events fired.\n"
        "6. VERIFY: after each sensor, call relspec_get_trend and compare "
        "row counts and level ranges against their source. Summarise "
        "per sensor: submitted, ingested, duplicates skipped, waveforms "
        "stored, events raised. Never report success you have not read "
        "back.\n"
        "The passphrase is the only key to this data: have the user "
        "choose it and store it themselves; never write it into files or "
        "logs, and never echo it back in your replies.")

if __name__ == '__main__':
    transport = os.environ.get('RELSPEC_MCP_TRANSPORT', 'stdio')
    if transport == 'http':
        mcp.run(transport='streamable-http',
                host='0.0.0.0', port=int(os.environ.get('PORT', 8765)))
    else:
        mcp.run()
