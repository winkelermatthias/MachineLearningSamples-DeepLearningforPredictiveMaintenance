"""Server-side health model and per-pattern z-scores.

Everything here is computed at ingest, inside the per-sensor advisory
lock, and stored on the acquisition/pattern rows — clients only render.

Per-pattern z (design doc section 3): each (sensor, rail, track) keeps a
streaming baseline of E_dB = 10*log10(energy) as Welford running
mean/std — deliberately NOT forgetting, updated ONLY on frames the gate
held quiet and only while the track does not already look anomalous
(|z| < 3). An exponentially-weighted baseline was tried first and
rejected: it chases a slow fault ramp and re-normalizes it. z is
reported after 10 observations ("learning" before that), clamped to
±10. A track that had a baseline and then misses 3+ consecutive frames
is flagged vanished (a repair — or a broken sensor — both worth a
look).

Health tier (design doc section 2): tier = max(ISO-velocity floor, gate
state, pattern state), with one-step hysteresis — a tier only relaxes
after 5 consecutive frames whose raw tier sits below the displayed one.
The drivers list stores *why*, so every chip in the UI can explain
itself.
"""
from __future__ import annotations
import json
import struct
import numpy as np

TIERS = ('healthy', 'monitor', 'alert', 'critical')
BASELINE_MIN_N = 10
BASELINE_ETA = 0.1
MAD_FLOOR_DB = 0.25
HYSTERESIS_N = 5

# ISO 10816-ish velocity zones for a general class-II machine (mm/s RMS).
# Zone A/B -> floor 0, C -> 1, D -> 2. Machine-class tables can refine
# this per asset later; the shape of the model does not change.
VEL_ZONES = ((2.8, 0), (7.1, 1), (float('inf'), 2))

FP_ACC_BANDS, FP_ENV_BANDS = 16, 8
FP_LEN = FP_ACC_BANDS + FP_ENV_BANDS + 6


def fingerprint(res: dict) -> bytes:
    """~30 B change fingerprint carried by EVERY acquisition: banded
    means of the two binned dB spectra + gate score/drift + vel RMS.
    Trends interpolate between full spectral frames with these."""
    out = bytearray()
    for rail, nb in (('acc', FP_ACC_BANDS), ('env', FP_ENV_BANDS)):
        u8 = np.frombuffer(res['frames'][rail]['o'], dtype=np.uint8)
        bands = np.array_split(u8.astype(np.float64), nb)
        out += bytes(int(round(b.mean())) for b in bands)
    out += struct.pack('<HHH',
                       min(65535, int(res['gate']['score'] * 100)),
                       min(65535, int(abs(res['gate']['drift']) * 100)),
                       min(65535, int(res['vel_rms'] * 100)))
    return bytes(out)


def fp_decode(b: bytes | None) -> dict | None:
    if not b or len(b) != FP_LEN:
        return None
    acc = list(b[:FP_ACC_BANDS])
    env = list(b[FP_ACC_BANDS:FP_ACC_BANDS + FP_ENV_BANDS])
    score, drift, vel = struct.unpack('<HHH', b[-6:])
    return dict(acc_bands=acc, env_bands=env,
                score=score / 100, drift=drift / 100, vel_rms=vel / 100)


def _gate_quiet(gate: dict) -> bool:
    return gate['decision'] not in ('up_change',) and gate['score'] < 3.0


def update_z(cur, sensor_id: str, ts, res: dict) -> tuple[float | None, list]:
    """Compute z for every pattern in res (mutating each pattern dict),
    update baselines on quiet frames, track vanished baselined tracks.
    Returns (max z over tracks, vanished [(rail, track_id)])."""
    quiet = _gate_quiet(res['gate'])
    zmax = None
    seen: set[tuple[str, int]] = set()
    for rail, pats in res['patterns'].items():
        for p in pats:
            tid = p['track_id']
            seen.add((rail, tid))
            e_db = 10.0 * np.log10(max(p['energy'], 1e-12))
            row = cur.execute(
                'SELECT n, med, mad FROM pattern_baseline '
                'WHERE sensor_id=%s AND rail=%s AND track_id=%s',
                (sensor_id, rail, tid)).fetchone()
            z = None
            if row and row[0] >= BASELINE_MIN_N:
                sd = max(row[2], MAD_FLOOR_DB)
                z = float(np.clip((e_db - row[1]) / sd, -10, 10))
                zmax = z if zmax is None else max(zmax, z)
            p['z'] = z
            teach = quiet and (z is None or abs(z) < 3.0)
            if row is None:
                cur.execute(
                    '''INSERT INTO pattern_baseline (sensor_id, rail, track_id,
                       n, med, mad, window_start, last_seen, miss_streak)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,0)''',
                    (sensor_id, rail, tid, 1 if teach else 0,
                     e_db, 1.0, ts, ts))
            else:
                n, mean, sd0 = row
                if teach:
                    # Welford over accepted frames: med column holds the
                    # running mean, mad holds the running std
                    n += 1
                    var = sd0 * sd0
                    delta = e_db - mean
                    mean += delta / n
                    var = ((n - 1) * var + delta * (e_db - mean)) / n
                    sd0 = float(np.sqrt(max(var, 0.0)))
                cur.execute(
                    '''UPDATE pattern_baseline SET n=%s, med=%s, mad=%s,
                       last_seen=%s, miss_streak=0
                       WHERE sensor_id=%s AND rail=%s AND track_id=%s''',
                    (n, mean, sd0, ts, sensor_id, rail, tid))
    # baselined tracks absent from this frame
    vanished = []
    for rail, tid, streak in cur.execute(
            'SELECT rail, track_id, miss_streak FROM pattern_baseline '
            'WHERE sensor_id=%s AND n>=%s',
            (sensor_id, BASELINE_MIN_N)).fetchall():
        if (rail, tid) in seen:
            continue
        streak += 1
        cur.execute(
            'UPDATE pattern_baseline SET miss_streak=%s '
            'WHERE sensor_id=%s AND rail=%s AND track_id=%s',
            (streak, sensor_id, rail, tid))
        if streak >= 3:
            vanished.append((rail, tid))
            zmax = -10.0 if zmax is None else zmax
    return zmax, vanished


def _floor_state(vel_rms_mm_s: float) -> int:
    for hi, state in VEL_ZONES:
        if vel_rms_mm_s < hi:
            return state
    return 2


def _gate_state(gate: dict) -> int:
    if gate['decision'] == 'up_change':
        return 2
    if gate['score'] >= 3.0:
        return 1
    return 0


def _pattern_state(zmax: float | None) -> int:
    if zmax is None:
        return 0
    if zmax >= 6:
        return 3
    if zmax >= 4:
        return 2
    if zmax >= 2.5:
        return 1
    return 0


def compute(cur, sensor_id: str, res: dict,
            zmax: float | None, vanished: list) -> tuple[int, str]:
    """Returns (health, drivers_json) with one-step hysteresis against
    the previous acquisition's stored health."""
    floor = _floor_state(res['vel_rms'])
    gstate = _gate_state(res['gate'])
    pstate = _pattern_state(zmax)
    raw = max(floor, gstate, pstate)

    prev = cur.execute(
        'SELECT health, health_drivers FROM acquisition '
        'WHERE sensor_id=%s ORDER BY ts DESC LIMIT 1',
        (sensor_id,)).fetchone()
    prev_health = prev[0] if prev else None
    prev_streak = 0
    if prev and prev[1]:
        d = prev[1] if isinstance(prev[1], dict) else json.loads(prev[1])
        prev_streak = int(d.get('streak', 0))

    if prev_health is None or raw >= prev_health:
        health, streak = raw, 0
    else:
        streak = prev_streak + 1
        if streak >= HYSTERESIS_N:
            health, streak = prev_health - 1, 0
        else:
            health = prev_health

    drivers = [dict(signal='vel_rms', value=round(res['vel_rms'], 2),
                    state=floor),
               dict(signal='gate', decision=res['gate']['decision'],
                    score=round(res['gate']['score'], 2), state=gstate)]
    if zmax is not None:
        drivers.append(dict(signal='pattern_z', value=round(zmax, 2),
                            state=pstate))
    for rail, tid in vanished:
        drivers.append(dict(signal='vanished_track', rail=rail,
                            track_id=tid, state=1))
    return health, json.dumps(dict(raw=raw, streak=streak, drivers=drivers))
