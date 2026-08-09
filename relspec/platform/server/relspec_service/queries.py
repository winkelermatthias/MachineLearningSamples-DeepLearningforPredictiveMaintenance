"""Read-side assembly. Every function returns JSON-ready dicts; spectra go
out as base64 uint8 exactly in the explorer's frame format, so the explorer
bundle is a pure DB read with no signal processing on the path.
"""
from __future__ import annotations
import base64, datetime as dt
from . import db

B64 = lambda b: base64.b64encode(bytes(b)).decode()

def hierarchy(cur, wsid: str) -> dict:
    plants = {}
    rows = cur.execute(
        '''SELECT p.plant_id, p.name, a.asset_id, a.tag, a.machine_type,
                  c.component_id, c.name, c.kind,
                  s.sensor_id, s.code, s.units, s.fr_nominal
           FROM plant p
           LEFT JOIN asset a ON a.plant_id=p.plant_id
           LEFT JOIN component c ON c.asset_id=a.asset_id
           LEFT JOIN sensor s ON s.component_id=c.component_id
           WHERE p.workspace_id=%s
           ORDER BY p.name, a.tag, c.name, s.code''', (wsid,)).fetchall()
    for pid, pname, aid, atag, mt, cid, cname, ckind, sid, scode, su, sfr in rows:
        p = plants.setdefault(pid, dict(plant_id=pid, name=pname, assets={}))
        if aid is None: continue
        a = p['assets'].setdefault(aid, dict(asset_id=aid, tag=atag,
                                             machine_type=mt, components={}))
        if cid is None: continue
        c = a['components'].setdefault(cid, dict(component_id=cid, name=cname,
                                                 kind=ckind, sensors=[]))
        if sid is not None:
            c['sensors'].append(dict(sensor_id=sid, code=scode, units=su,
                                     fr_nominal=sfr))
    for p in plants.values():
        p['assets'] = list(p['assets'].values())
        for a in p['assets']:
            a['components'] = list(a['components'].values())
    return dict(plants=list(plants.values()))

def fleet_overview(cur, wsid: str) -> list[dict]:
    rows = cur.execute(
        '''SELECT sl.sensor_id, s.code, c.name, a.tag, p.name,
                  sl.last_ts, sl.vel_rms, sl.acc_rms, sl.env_rms,
                  sl.gate_score, sl.gate_decision, sl.fr_est
           FROM sensor_latest sl
           JOIN sensor s ON s.sensor_id=sl.sensor_id
           JOIN component c ON c.component_id=s.component_id
           JOIN asset a ON a.asset_id=c.asset_id
           JOIN plant p ON p.plant_id=a.plant_id
           WHERE sl.workspace_id=%s
           ORDER BY p.name, a.tag, c.name, s.code''', (wsid,)).fetchall()
    return [dict(sensor_id=r[0], code=r[1], component=r[2], asset=r[3],
                 plant=r[4], path=f'{r[4]}/{r[3]}/{r[2]}/{r[1]}',
                 last_ts=r[5].isoformat(), vel_rms=r[6], acc_rms=r[7],
                 env_rms=r[8], gate_score=r[9], gate_decision=r[10],
                 fr_est=r[11]) for r in rows]

def trend(cur, wsid: str, sensor_id: str, t0, t1) -> list[dict]:
    rows = cur.execute(
        '''SELECT bucket, n, vel_avg, vel_max, acc_avg, acc_max,
                  env_avg, env_max, gate_max, bytes
           FROM trend_daily
           WHERE workspace_id=%s AND sensor_id=%s AND bucket>=%s AND bucket<=%s
           ORDER BY bucket''', (wsid, sensor_id, t0, t1)).fetchall()
    return [dict(bucket=r[0].isoformat(), n=r[1], vel_avg=r[2], vel_max=r[3],
                 acc_avg=r[4], acc_max=r[5], env_avg=r[6], env_max=r[7],
                 gate_max=r[8], bytes=r[9]) for r in rows]

def frames(cur, wsid: str, sensor_id: str, t0, t1, limit: int) -> list[dict]:
    rows = cur.execute(
        '''SELECT acq_id, ts, fr_est, tier, vel_rms, acc_rms, env_rms,
                  gate_score, gate_drift, gate_decision, gate_kind,
                  frame_kind, payload_bytes, waveform_stored, sig_reason
           FROM acquisition
           WHERE workspace_id=%s AND sensor_id=%s AND ts>=%s AND ts<=%s
           ORDER BY ts DESC LIMIT %s''',
        (wsid, sensor_id, t0, t1, limit)).fetchall()
    return [dict(acq_id=r[0], ts=r[1].isoformat(), fr=r[2], tier=r[3],
                 vr=r[4], ar=r[5], er=r[6],
                 gate=dict(s=r[7], dr=r[8], dec=r[9], k=r[10]),
                 kind=r[11], bytes=r[12], waveform_stored=r[13],
                 sig_reason=r[14]) for r in rows]

def spectra(cur, wsid: str, acq_id: str) -> dict | None:
    own = cur.execute(
        'SELECT 1 FROM acquisition WHERE acq_id=%s AND workspace_id=%s',
        (acq_id, wsid)).fetchone()
    if not own: return None
    out = dict(acq_id=acq_id)
    for rail, kind, payload, o, dec, ownm, rs in cur.execute(
            'SELECT rail, kind, payload, o, dec, own, res_share '
            'FROM spectrum_frame WHERE acq_id=%s', (acq_id,)).fetchall():
        pats = [dict(idx=r[0], t=r[1], kind=r[2], key=r[3], f0=r[4], sp=r[5],
                     e=r[6], ed=r[7], share=r[8], nv=r[9], vc=r[10], vf=r[11])
                for r in cur.execute(
                    '''SELECT idx, track_id, kind, key, f0, spacing, energy,
                       energy_dec, share, ver_harm, ver_claimed, ver_frac
                       FROM pattern WHERE acq_id=%s AND rail=%s ORDER BY idx''',
                    (acq_id, rail)).fetchall()]
        out[rail] = dict(kind=kind, payload_bytes=len(payload),
                         o=B64(o), dec=B64(dec), own=B64(ownm),
                         res_share=rs, patterns=pats)
    return out

def explorer_bundle(cur, wsid: str, sensor_id: str, limit: int) -> dict | None:
    """A ready-to-render explorer sequence: same shape the offline exporter
    produces, assembled from stored rows, oldest of the window first."""
    meta = cur.execute(
        '''SELECT s.code, s.fr_nominal, c.name, a.tag, p.name, s.fs_nominal
           FROM sensor s JOIN component c ON c.component_id=s.component_id
           JOIN asset a ON a.asset_id=c.asset_id
           JOIN plant p ON p.plant_id=a.plant_id
           WHERE s.workspace_id=%s AND s.sensor_id=%s''',
        (wsid, sensor_id)).fetchone()
    if not meta: return None
    rows = cur.execute(
        '''SELECT acq_id, ts, fr_est, tier, vel_rms, acc_rms, env_rms,
                  gate_score, gate_drift, gate_decision, gate_kind,
                  frame_kind, payload_bytes, fs
           FROM acquisition WHERE workspace_id=%s AND sensor_id=%s
           ORDER BY ts DESC LIMIT %s''', (wsid, sensor_id, limit)).fetchall()
    rows = rows[::-1]
    frames_out, tracks = [], {'acc': {}, 'env': {}}
    for r in rows:
        acq_id = r[0]
        fr = dict(ts=r[1].isoformat(), fr=r[2], tier=r[3],
                  fr_true=None, vr=r[4], ar=r[5], er=r[6],
                  gate=dict(s=r[7] or 0, dr=r[8] or 0, dec=r[9] or 'init',
                            k=r[10] or 0),
                  kind=r[11], bytes=r[12], acq_id=acq_id)
        tot = {'acc': 0, 'env': 0}
        for rail, o, dec, ownm, rs, payload in cur.execute(
                'SELECT rail, o, dec, own, res_share, payload '
                'FROM spectrum_frame WHERE acq_id=%s', (acq_id,)).fetchall():
            pfx = 'a' if rail == 'acc' else 'e'
            fr[pfx+'_o'] = B64(o); fr[pfx+'_d'] = B64(dec)
            fr[rail+'_own'] = B64(ownm); fr[rail+'_res'] = rs
            tot[rail] = len(payload)
            pats = [dict(t=p[0], kind=p[1], key=p[2], f0=p[3], sp=p[4],
                         e=p[5], ed=p[6], share=p[7], nv=p[8], vc=p[9], vf=p[10])
                    for p in cur.execute(
                        '''SELECT track_id, kind, key, f0, spacing, energy,
                           energy_dec, share, ver_harm, ver_claimed, ver_frac
                           FROM pattern WHERE acq_id=%s AND rail=%s ORDER BY idx''',
                        (acq_id, rail)).fetchall()]
            fr[rail+'_pat'] = pats
            for p in pats: tracks[rail].setdefault(p['t'], dict(kind=p['kind'],
                                                                key=p['key']))
        fr['ba'], fr['be'] = tot['acc'], tot['env']
        frames_out.append(fr)
    # dense track list indexed by id (front end indexes tracks[rail][t])
    tr = {}
    for rail in ('acc', 'env'):
        mx = max(tracks[rail], default=-1)
        tr[rail] = [tracks[rail].get(i, dict(kind='other', key=0.0))
                    for i in range(mx+1)]
    code, frn, comp, atag, plant, fsn = meta
    return dict(name=sensor_id,
                title=f'{plant} / {atag} / {comp} / {code}',
                note='', cat=plant, cad='',
                fs=fsn or 12000.0, fr_nominal=frn,
                frames=frames_out, tracks=tr)

def events(cur, wsid: str, t0, t1, limit: int) -> list[dict]:
    rows = cur.execute(
        '''SELECT event_id, sensor_id, ts, type, payload FROM event
           WHERE workspace_id=%s AND ts>=%s AND ts<=%s
           ORDER BY ts DESC LIMIT %s''', (wsid, t0, t1, limit)).fetchall()
    return [dict(event_id=r[0], sensor_id=r[1], ts=r[2].isoformat(),
                 type=r[3], payload=r[4]) for r in rows]

def waveform(cur, wsid: str, acq_id: str) -> dict | None:
    row = cur.execute(
        '''SELECT w.encoding, w.scale, w.fs, w.data FROM waveform w
           JOIN acquisition a ON a.acq_id=w.acq_id
           WHERE w.acq_id=%s AND a.workspace_id=%s''',
        (acq_id, wsid)).fetchone()
    if not row: return None
    return dict(acq_id=acq_id, encoding=row[0], scale=row[1], fs=row[2],
                data_b64=B64(row[3]))
