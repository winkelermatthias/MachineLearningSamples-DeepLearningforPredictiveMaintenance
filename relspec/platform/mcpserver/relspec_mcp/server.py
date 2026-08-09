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
    ov = _call('GET', '/v1/fleet/overview', headers=_hdr())
    match = [s for s in ov if s['path'] == sensor_path]
    if not match:
        raise RuntimeError(f'no sensor at path {sensor_path!r}; '
                           f'known: {[s["path"] for s in ov]}')
    return _call('GET', f"/v1/sensors/{match[0]['sensor_id']}/trend",
                 params={'from': date_from, 'to': date_to}, headers=_hdr())

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
        "3. MIGRATE oldest-first per sensor, in batches of <= 200 lines "
        "via relspec_submit_batch. Order matters: baselines, codec chains "
        "and weekly storage budgets accrue in time order, and history "
        "submitted backwards is history wasted (the server rejects "
        "out-of-order timestamps per sensor). Use a stable client_ref "
        "(their primary key) so re-runs are idempotent.\n"
        "4. Do NOT decide which waveforms deserve full storage — submit "
        "everything in range; the server keeps every waveform's features, "
        "spectra and patterns, and stores raw waveforms on its own "
        "significance policy (weekly baseline + change events). Report "
        "the verdicts honestly, including how many raw waveforms were "
        "kept and how many change events fired.\n"
        "5. VERIFY: after each sensor, call relspec_get_trend and compare "
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
