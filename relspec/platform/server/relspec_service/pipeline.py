"""The relspec pipeline wrapped for service use: stateless process, stateful
sensor. Codec chains, gate baselines and the pinned demodulation band live
in the database; this module (de)serialises them and runs one acquisition
end to end exactly as the offline evaluation does — that equivalence is a
test, not an aspiration (tests/test_parity.py).
"""
from __future__ import annotations
import io, json
import numpy as np
from . import config  # noqa: F401  (sets sys.path for relspec import)
from relspec.pipeline import Gate, NBINS, ENV_BINS, to_amp, CENTERS, ENV_CENTERS
from relspec.codec2 import Codec2, Decoder2
from relspec import pipeline2
from relspec.pipeline2 import extract2
from relspec.patterns import extract_patterns

RAILS = ('acc', 'env')
NB = {'acc': NBINS, 'env': ENV_BINS}
CEN = {'acc': CENTERS, 'env': ENV_CENTERS}

def new_codec(rail: str) -> Codec2:
    return (Codec2(nbins=NBINS, n_bands=16, refs='best') if rail == 'acc'
            else Codec2(nbins=ENV_BINS, n_peaks=32, n_bands=12, refs='best'))

def new_decoder(rail: str) -> Decoder2:
    return (Decoder2(nbins=NBINS, n_bands=16) if rail == 'acc'
            else Decoder2(nbins=ENV_BINS, n_peaks=32, n_bands=12))

# -------------------------------------------------- codec state <-> bytes
def dump_codec(enc: Codec2, mad: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        anchor=(enc.anchor if enc.anchor is not None else np.array([])),
        prev=(enc.prev if enc.prev is not None else np.array([])),
        mad=mad, abytes=np.array([enc.abytes]))
    return buf.getvalue()

def load_codec(rail: str, blob: bytes | None):
    enc = new_codec(rail)
    mad = np.full(NB[rail], 4.0)
    if blob:
        z = np.load(io.BytesIO(blob))
        if z['anchor'].size: enc.anchor = z['anchor'].astype(np.int64)
        if z['prev'].size: enc.prev = z['prev'].astype(np.int64)
        mad = z['mad'].astype(np.float64)
        enc.abytes = int(z['abytes'][0])
    return enc, mad

# --------------------------------------------------- gate state <-> bytes
_GATE_ARRAYS = ('med', 'mad', 'smed', 'smad', 'cp', 'cn',
                'band_med', 'band_mad', 'bin_med', 'bin_mad')
_GATE_SCALARS = ('n', 'n_slow', 'slow_frozen', 'in_change', 'ring', 'burst',
                 'prev_score', 'step_latch', 'last_up', 'last_change',
                 'last_score')

def dump_gate(g: Gate) -> bytes:
    d = {k: (getattr(g, k).tolist() if getattr(g, k) is not None else None)
         for k in _GATE_ARRAYS}
    d.update({k: getattr(g, k) for k in _GATE_SCALARS})
    d['hist'] = list(g.hist)
    return json.dumps(d).encode()

def load_gate(blob: bytes | None) -> Gate:
    g = Gate()
    if not blob: return g
    d = json.loads(blob.decode())
    for k in _GATE_ARRAYS:
        v = d.get(k)
        if v is not None: setattr(g, k, np.array(v, dtype=np.float64))
    for k in _GATE_SCALARS:
        if k in d: setattr(g, k, d[k])
    g.hist = list(d.get('hist', []))
    return g

# ----------------------------------------------------------- track registry
def resolve_track(tracks: list[dict], kind: str, key: float) -> int:
    """tracks: [{track_id, kind, key, ...}] for one sensor+rail, mutated in
    place. Same 4% tolerance and slow key update as the offline registry."""
    best, err = None, 1e9
    for t in tracks:
        if t['kind'] != kind: continue
        e = abs(key-t['key'])/max(abs(t['key']), 1e-9)
        if e < 0.04 and e < err: best, err = t, e
    if best is None:
        tid = 1+max((t['track_id'] for t in tracks), default=-1)
        tracks.append(dict(track_id=tid, kind=kind, key=float(key), new=True))
        return tid
    best['key'] = 0.9*best['key']+0.1*float(key)
    best['dirty'] = True
    return best['track_id']

# ------------------------------------------------------------- one waveform
def process(x: np.ndarray, fs: float, sensor: dict, codec_blobs: dict,
            gate_blob: bytes | None, tracks: dict[str, list],
            t_epoch: int) -> dict:
    """sensor: row dict (sensor_id, fr_nominal, band_lo_hz, band_hi_hz).
    tracks: {'acc': [...], 'env': [...]} mutated in place (new/dirty flags).
    Returns everything ingest persists, plus refreshed state blobs."""
    key = sensor['sensor_id']
    if sensor.get('band_lo_hz') and sensor.get('band_hi_hz'):
        pipeline2._band_cache[key] = (sensor['band_lo_hz'], sensor['band_hi_hz'])
    e = extract2(x.astype(np.float64), fs, band_key=key,
                 fr_nominal=sensor.get('fr_nominal'))
    band = pipeline2._band_cache.get(key)

    out = dict(fr=float(e.fr), tier=int(e.tier), conf=float(e.conf),
               vel_rms=float(e.vel_rms), acc_rms=float(e.acc_rms),
               env_rms=float(e.env_rms), acc_kurt=float(e.acc_kurt),
               acc_crest=float(e.acc_crest),
               band=(float(band[0]), float(band[1])) if band else None,
               frames={}, patterns={}, codec_blobs={}, n=len(x))

    u8s = {'acc': e.acc_u8, 'env': e.env_u8}
    for rail in RAILS:
        enc, mad = load_codec(rail, codec_blobs.get(rail))
        dec = new_decoder(rail)
        # The decoder's state is definitionally the encoder's model of it.
        if enc.anchor is not None:
            dec.anchor = enc.anchor.copy()
            dec.prev = enc.prev.copy() if enc.prev is not None else enc.anchor.copy()
        u8 = u8s[rail]
        pl, kind = enc.encode(u8, mad)
        got, _ = dec.decode(pl)
        mad = np.maximum(0.95*mad+0.05*np.abs(u8.astype(float)-got.astype(float)), 1.0)

        amp_o = to_amp(u8)
        kw = dict(min_harmonics=3, f_hi=6.5) if rail == 'env' else {}
        _, _, acc = extract_patterns(amp_o, CEN[rail], **kw)
        owner = acc['owner']
        dec2 = to_amp(got).astype(np.float64)**2
        pats, omap = [], np.full(NB[rail], 255, dtype=np.uint8)
        for pi, p in enumerate(acc['patterns']):
            tid = resolve_track(tracks[rail], p['kind'], p['key'])
            ed = float(dec2[owner == p['pid']].sum())
            pats.append(dict(idx=pi, track_id=tid, kind=p['kind'],
                             key=float(p['key']), f0=float(p['f0']),
                             spacing=(float(p['spacing']) if p['spacing'] else None),
                             energy=float(p['energy']), energy_dec=ed,
                             share=float(p['share']),
                             ver_harm=p.get('ver_harm'),
                             ver_claimed=p.get('ver_claimed'),
                             ver_frac=p.get('ver_frac')))
            omap[owner == p['pid']] = min(pi, 254)
        out['frames'][rail] = dict(kind=kind, payload=pl,
                                   o=u8.tobytes(), dec=got.tobytes(),
                                   own=omap.tobytes(),
                                   res_share=float(acc['residual_share']))
        out['patterns'][rail] = pats
        out['codec_blobs'][rail] = dump_codec(enc, mad)

    gate = load_gate(gate_blob)
    if gate.n == 0:
        gate.init(e)
        out['gate'] = dict(score=0.0, drift=0.0, decision='init', kind=0)
    else:
        s = gate.decide(e, t_epoch)
        out['gate'] = dict(score=float(s['score']), drift=float(s['drift']),
                           decision=s['decision'], kind=int(s['kind']))
    out['gate_blob'] = dump_gate(gate)
    return out
