"""One waveform in: decode payload, run the pipeline, decide significance,
persist everything in a single transaction. The server owns every policy
decision; the response tells the client honestly what happened.
"""
from __future__ import annotations
import base64, collections, datetime as dt, hashlib, json, threading, time, uuid
import numpy as np
import psycopg.errors
import zstandard
from . import config, db, health, pipeline

class IngestError(Exception):
    def __init__(self, status: int, detail: str, retry_after: int | None = None):
        self.status, self.detail, self.retry_after = status, detail, retry_after

G = 9.80665

# Per-workspace sliding-window rate cap (in-memory, per replica). Batch
# reserves all its lines up front so it cannot sidestep the cap.
_rate: dict[str, collections.deque] = {}
_rate_lock = threading.Lock()

def reserve_rate(wsid: str, n: int = 1):
    cap = config.INGEST_RATE_PER_MIN
    now = time.time()
    with _rate_lock:
        q = _rate.setdefault(wsid, collections.deque())
        while q and now-q[0] >= 60: q.popleft()
        if len(q)+n > cap:
            retry = max(1, int(60-(now-q[0]))+1) if q else 60
            raise IngestError(429, f'ingest rate cap {cap}/min exceeded',
                              retry_after=retry)
        q.extend([now]*n)

def _sensor_lock(cur, sensor_id: str):
    """Serialize read-process-write per sensor: the codec/gate chain is a
    strict sequence, so concurrent posts for one sensor must queue. The
    xact lock releases on commit/rollback; different sensors don't collide
    (beyond a 2^-64 hash coincidence, which only costs concurrency)."""
    h = int.from_bytes(hashlib.sha256(sensor_id.encode()).digest()[:8],
                       'big', signed=True)
    cur.execute('SELECT pg_advisory_xact_lock(%s)', (h,))

def decode_samples(body: dict) -> np.ndarray:
    enc = body.get('encoding', 'float32')
    if 'data_b64' in body:
        raw = base64.b64decode(body['data_b64'])
        if enc == 'float32':
            x = np.frombuffer(raw, dtype='<f4').astype(np.float64)
        elif enc == 'int16':
            scale = float(body.get('scale') or 0)
            if scale <= 0: raise IngestError(400, 'int16 encoding requires scale')
            x = np.frombuffer(raw, dtype='<i2').astype(np.float64)*scale
        else:
            raise IngestError(400, f'unknown encoding {enc!r}')
    elif 'values' in body:
        x = np.asarray(body['values'], dtype=np.float64)
    else:
        raise IngestError(400, 'provide data_b64 or values')
    units = (body.get('units') or 'g').lower()
    if units in ('g',): pass
    elif units in ('m/s2', 'm/s^2', 'mps2'): x = x/G
    elif units in ('mm/s2', 'mm/s^2'): x = x/(1000*G)
    else: raise IngestError(400, f'unsupported units {body.get("units")!r}')
    if len(x) > config.MAX_SAMPLES:
        raise IngestError(413,
            f'{len(x)} samples exceeds the {config.MAX_SAMPLES} limit')
    if not np.isfinite(x).all():
        raise IngestError(400, 'waveform contains NaN or Inf')
    return x

def validate(x: np.ndarray, fs: float):
    if not (1000 <= fs <= 100_000):
        raise IngestError(400, 'fs must be between 1 kHz and 100 kHz')
    dur = len(x)/fs
    if not (0.25 <= dur <= 60):
        raise IngestError(400, f'duration {dur:.2f}s outside 0.25-60 s')
    clipped = np.mean(np.abs(x) >= np.max(np.abs(x))*0.999) if len(x) else 0
    if clipped > 0.2:
        raise IngestError(400, 'waveform looks clipped (>20% of samples at rail)')

def _sig_params(ws_params: dict) -> dict:
    p = dict(config.SIG_DEFAULTS)
    p.update({k: v for k, v in (ws_params or {}).items() if k in p})
    return p

def significance(cur, sensor_id: str, ts: dt.datetime, gate: dict,
                 params: dict) -> tuple[bool, str]:
    """Server-side decision. Features/frames/patterns are always stored;
    this only gates the RAW waveform."""
    week_ago = ts-dt.timedelta(days=7)
    n_week, = cur.execute(
        'SELECT count(*) FROM waveform WHERE sensor_id=%s AND ts>%s AND ts<=%s',
        (sensor_id, week_ago, ts)).fetchone()
    last = cur.execute(
        'SELECT max(ts) FROM waveform WHERE sensor_id=%s AND ts<=%s',
        (sensor_id, ts)).fetchone()[0]
    if last is None:
        return True, 'first_waveform'
    budget = 1+int(params['event_budget_per_week'])
    if n_week >= budget:
        return False, 'weekly_budget_exhausted'
    if (ts-last).days >= int(params['baseline_days']):
        return True, 'weekly_baseline'
    if gate['decision'] == 'up_change' and n_week < budget \
            and int(params['event_budget_per_week']) > 0:
        return True, 'gate_change'
    return False, 'not_significant'

def compress_waveform(x: np.ndarray) -> tuple[bytes, float]:
    scale = float(np.max(np.abs(x)))/32767.0 or 1e-12
    i16 = np.clip(np.round(x/scale), -32767, 32767).astype('<i2')
    return zstandard.ZstdCompressor(level=9).compress(i16.tobytes()), scale

def ingest_one(wsid: str, body: dict) -> dict:
    ts_raw = body.get('ts')
    try:
        ts = dt.datetime.fromisoformat(str(ts_raw).replace('Z', '+00:00'))
        if ts.tzinfo is None: ts = ts.replace(tzinfo=dt.timezone.utc)
    except Exception:
        raise IngestError(400, f'bad ts {ts_raw!r} (ISO-8601 required)')
    fs = float(body.get('fs') or 0)
    if not (config.FS_MIN <= fs <= config.FS_MAX):
        raise IngestError(400,
            f'fs must be between {config.FS_MIN:g} and {config.FS_MAX:g} Hz')
    x = decode_samples(body)
    validate(x, fs)
    client_ref = body.get('client_ref')

    with db.pool().connection() as conn:
        cur = conn.cursor()
        sensor = _resolve_sensor(cur, wsid, body)
        _sensor_lock(cur, sensor['sensor_id'])
        # idempotency: same (sensor, client_ref) returns the original verdict
        if client_ref:
            row = cur.execute(
                'SELECT acq_id, waveform_stored, sig_reason, payload_bytes '
                'FROM acquisition WHERE sensor_id=%s AND client_ref=%s',
                (sensor['sensor_id'], client_ref)).fetchone()
            if row:
                return dict(acq_id=row[0], duplicate=True,
                            waveform_stored=row[1], sig_reason=row[2],
                            payload_bytes=row[3])
        # chain discipline: history must arrive in time order per sensor
        last_ts = cur.execute(
            'SELECT max(ts) FROM acquisition WHERE sensor_id=%s',
            (sensor['sensor_id'],)).fetchone()[0]
        if last_ts is not None and ts <= last_ts:
            raise IngestError(409,
                f'out-of-order: sensor already has data at {last_ts.isoformat()}; '
                'submit waveforms oldest-first')

        codec_blobs = {r: b for r, b in cur.execute(
            'SELECT rail, state FROM codec_state WHERE sensor_id=%s',
            (sensor['sensor_id'],)).fetchall()}
        grow = cur.execute('SELECT state FROM gate_state WHERE sensor_id=%s',
                           (sensor['sensor_id'],)).fetchone()
        tracks = {r: [dict(track_id=t, kind=k, key=ky)
                      for t, k, ky in cur.execute(
                          'SELECT track_id, kind, key FROM pattern_track '
                          'WHERE sensor_id=%s AND rail=%s',
                          (sensor['sensor_id'], r)).fetchall()]
                  for r in pipeline.RAILS}

        res = pipeline.process(x, fs, sensor, codec_blobs,
                               grow[0] if grow else None, tracks,
                               int(ts.timestamp()))

        ws_params = cur.execute(
            'SELECT params FROM workspace WHERE workspace_id=%s',
            (wsid,)).fetchone()[0]
        store, reason = significance(cur, sensor['sensor_id'], ts,
                                     res['gate'], _sig_params(ws_params))

        # health model: z per pattern (mutates res), baselines, tier —
        # all inside the sensor's advisory lock like the codec state
        zmax, vanished = health.update_z(cur, sensor['sensor_id'], ts, res)
        tier_h, drivers = health.compute(cur, sensor['sensor_id'], res,
                                         zmax, vanished)
        fp = health.fingerprint(res)

        acq_id = uuid.uuid4().hex
        pb = sum(len(res['frames'][r]['payload']) for r in pipeline.RAILS)
        kinds = [res['frames'][r]['kind'] for r in pipeline.RAILS]
        fk = 'A' if 'anchor' in kinds else ('P' if 'residual_p' in kinds else 'R')
        try:
            cur.execute(
                '''INSERT INTO acquisition (workspace_id, acq_id, sensor_id, ts,
                   client_ref, fs, n_samples, fr_est, tier, conf, vel_rms, acc_rms,
                   env_rms, acc_kurt, acc_crest, gate_score, gate_drift,
                   gate_decision, gate_kind, frame_kind, payload_bytes,
                   waveform_stored, sig_reason, health, health_drivers,
                   fingerprint)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                           %s,%s,%s,%s,%s,%s,%s,%s,%s)''',
                (wsid, acq_id, sensor['sensor_id'], ts, client_ref, fs, len(x),
                 res['fr'], res['tier'], res['conf'], res['vel_rms'],
                 res['acc_rms'], res['env_rms'], res['acc_kurt'], res['acc_crest'],
                 res['gate']['score'], res['gate']['drift'],
                 res['gate']['decision'], res['gate']['kind'], fk, pb, store,
                 reason, tier_h, drivers, fp))
        except psycopg.errors.UniqueViolation:
            # backstop under the advisory lock: cannot normally trigger
            raise IngestError(409, 'sensor already has an acquisition at this ts')
        for rail in pipeline.RAILS:
            f = res['frames'][rail]
            cur.execute(
                'INSERT INTO spectrum_frame (acq_id, rail, kind, payload, o, dec, own, res_share) '
                'VALUES (%s,%s,%s,%s,%s,%s,%s,%s)',
                (acq_id, rail, f['kind'], f['payload'], f['o'], f['dec'],
                 f['own'], f['res_share']))
            for p in res['patterns'][rail]:
                cur.execute(
                    '''INSERT INTO pattern (acq_id, rail, idx, track_id, kind,
                       key, f0, spacing, energy, energy_dec, share, ver_harm,
                       ver_claimed, ver_frac, z)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
                    (acq_id, rail, p['idx'], p['track_id'], p['kind'], p['key'],
                     p['f0'], p['spacing'], p['energy'], p['energy_dec'],
                     p['share'], p['ver_harm'], p['ver_claimed'], p['ver_frac'],
                     p.get('z')))
            for t in tracks[rail]:
                if t.get('new'):
                    cur.execute(
                        '''INSERT INTO pattern_track (workspace_id, sensor_id,
                           rail, track_id, kind, key, first_seen, last_seen)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)''',
                        (wsid, sensor['sensor_id'], rail, t['track_id'],
                         t['kind'], t['key'], ts, ts))
                elif t.get('dirty'):
                    cur.execute(
                        'UPDATE pattern_track SET key=%s, last_seen=%s '
                        'WHERE sensor_id=%s AND rail=%s AND track_id=%s',
                        (t['key'], ts, sensor['sensor_id'], rail, t['track_id']))
            cur.execute(
                '''INSERT INTO codec_state (sensor_id, rail, state, last_ts)
                   VALUES (%s,%s,%s,%s)
                   ON CONFLICT (sensor_id, rail)
                   DO UPDATE SET state=EXCLUDED.state, last_ts=EXCLUDED.last_ts''',
                (sensor['sensor_id'], rail, res['codec_blobs'][rail], ts))
        cur.execute(
            '''INSERT INTO gate_state (sensor_id, state, n_seen)
               VALUES (%s,%s,1)
               ON CONFLICT (sensor_id)
               DO UPDATE SET state=EXCLUDED.state, n_seen=gate_state.n_seen+1''',
            (sensor['sensor_id'], res['gate_blob']))
        if res['band'] and not sensor.get('band_lo_hz'):
            cur.execute(
                'UPDATE sensor SET band_lo_hz=%s, band_hi_hz=%s WHERE sensor_id=%s',
                (res['band'][0], res['band'][1], sensor['sensor_id']))
        if store:
            blob, scale = compress_waveform(x)
            cur.execute(
                '''INSERT INTO waveform (acq_id, ts, sensor_id, encoding,
                   scale, fs, data) VALUES (%s,%s,%s,'int16-zstd',%s,%s,%s)''',
                (acq_id, ts, sensor['sensor_id'], scale, fs, blob))
        if res['gate']['decision'] == 'up_change':
            cur.execute(
                '''INSERT INTO event (workspace_id, event_id, sensor_id, ts,
                   type, payload) VALUES (%s,%s,%s,%s,'gate_change',%s)''',
                (wsid, uuid.uuid4().hex, sensor['sensor_id'], ts,
                 json.dumps(dict(score=res['gate']['score'],
                                 kind=res['gate']['kind']))))
        db.bump_version(cur, wsid)
        # NOTIFY rides the same transaction: subscribers hear about the
        # acquisition exactly when it becomes visible, never before
        cur.execute('SELECT pg_notify(%s, %s)', ('relspec_events', json.dumps(
            dict(ws=wsid, kind='acquisition', sensor_id=sensor['sensor_id'],
                 acq_id=acq_id, ts=ts.isoformat(), health=tier_h,
                 gate=res['gate']['decision'], stored=store))))
        if res['gate']['decision'] == 'up_change':
            cur.execute('SELECT pg_notify(%s, %s)',
                        ('relspec_events', json.dumps(
                            dict(ws=wsid, kind='gate_change',
                                 sensor_id=sensor['sensor_id'],
                                 ts=ts.isoformat(),
                                 score=res['gate']['score']))))
        conn.commit()
    return dict(acq_id=acq_id, duplicate=False, waveform_stored=store,
                sig_reason=reason, payload_bytes=pb, frame_kind=fk,
                fr_est=res['fr'], tier=res['tier'],
                features=dict(vel_rms=res['vel_rms'], acc_rms=res['acc_rms'],
                              env_rms=res['env_rms']),
                patterns=sum(len(res['patterns'][r]) for r in pipeline.RAILS),
                gate=res['gate'])

def _resolve_sensor(cur, wsid: str, body: dict) -> dict:
    cols = 'sensor_id, fr_nominal, band_lo_hz, band_hi_hz'
    if body.get('sensor_id'):
        row = cur.execute(
            f'SELECT {cols} FROM sensor WHERE workspace_id=%s AND sensor_id=%s',
            (wsid, body['sensor_id'])).fetchone()
    elif body.get('sensor_path'):
        try:
            plant, asset, comp, code = body['sensor_path'].split('/')
        except ValueError:
            raise IngestError(400,
                'sensor_path must be plant/asset/component/sensor')
        row = cur.execute(
            f'''SELECT s.sensor_id, s.fr_nominal, s.band_lo_hz, s.band_hi_hz
                FROM sensor s
                JOIN component c ON c.component_id=s.component_id
                JOIN asset a ON a.asset_id=c.asset_id
                JOIN plant p ON p.plant_id=a.plant_id
                WHERE s.workspace_id=%s AND p.name=%s AND a.tag=%s
                  AND c.name=%s AND s.code=%s''',
            (wsid, plant, asset, comp, code)).fetchone()
    else:
        raise IngestError(400, 'provide sensor_id or sensor_path')
    if not row:
        raise IngestError(404, 'sensor not found — create the hierarchy first')
    return dict(sensor_id=row[0], fr_nominal=row[1],
                band_lo_hz=row[2], band_hi_hz=row[3])
