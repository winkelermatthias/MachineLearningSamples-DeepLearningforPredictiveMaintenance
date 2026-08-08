"""Codec v2: the entropy-coded successor to pipeline.Codec.

Three changes, each measured separately in eval_codecs.py:

1. ENTROPY STAGE. v1 spent whole bytes on symbols worth 2-4 bits: raw uint8
   anchors, int8 band gains, 3-byte peak records, varint residuals. v2 packs
   everything through adaptive Rice coding (LOCO-I style per-context parameter
   estimation). Rice, not arithmetic coding, deliberately: shift-and-mask on an
   MCU, no multiplies, no tables, and within a few percent of entropy on the
   geometric-ish residual distributions actually observed.

2. ZERO-BYTE PEAK TABLES. v1 transmitted 3 bytes per anchor peak. But the
   decoder holds a bit-exact copy of the reference spectrum, so it can run the
   same peak picker the encoder ran and get the same table. Determinism replaces
   transmission: 120 bytes per anchor become 0.

3. REFERENCE CHOICE. v1 residuals always referenced the GOP anchor. v2 encodes
   against both the anchor and the previous decoded frame and keeps the smaller,
   at the cost of one header bit. Chain references compound packet-loss blast
   radius, so the choice is a config: 'anchor' restores v1 semantics, 'best'
   buys the extra ratio where the link layer can afford it.

The distortion contract is UNCHANGED from v1: transmitted bins are exact to the
0.5 dB quantiser, suppressed bins are bounded by max(k*MAD, floor), matched
peaks within mr_tol. Anything the pattern layer could see through v1 it sees
through v2; the pattern-energy ledger is the regression test for that claim.
"""
from __future__ import annotations
import numpy as np

Q_DB = 0.5

# ------------------------------------------------------------------- bit I/O
class BitWriter:
    __slots__ = ('buf', 'acc', 'nb')
    def __init__(self):
        self.buf = bytearray(); self.acc = 0; self.nb = 0
    def bits(self, v: int, n: int):
        self.acc = (self.acc << n) | (v & ((1 << n)-1)); self.nb += n
        while self.nb >= 8:
            self.nb -= 8
            self.buf.append((self.acc >> self.nb) & 0xFF)
        self.acc &= (1 << self.nb)-1
    def unary(self, q: int):
        while q >= 8: self.bits(0xFF, 8); q -= 8
        self.bits(((1 << q)-1) << 1, q+1)      # q ones then a zero
    def finish(self) -> bytes:
        if self.nb: self.bits(0, 8-self.nb)
        return bytes(self.buf)

class BitReader:
    __slots__ = ('b', 'pos')
    def __init__(self, b: bytes, bitpos: int = 0):
        self.b = b; self.pos = bitpos
    def bits(self, n: int) -> int:
        v = 0
        for _ in range(n):
            byte = self.b[self.pos >> 3]
            v = (v << 1) | ((byte >> (7-(self.pos & 7))) & 1)
            self.pos += 1
        return v
    def unary(self) -> int:
        q = 0
        while self.bits(1): q += 1
        return q

# ------------------------------------------------------- adaptive Rice codes
ESC_Q = 20                       # unary quotient escape -> 16-bit raw payload

class RiceCtx:
    """LOCO-I parameter estimation: k is derived from the running mean of the
    coded magnitudes, halved periodically so it tracks. Encoder and decoder
    update identically from decoded values, so k never needs transmitting."""
    __slots__ = ('A', 'N')
    def __init__(self, k0: int = 2):
        self.N = 1; self.A = max(1, 1 << k0)
    def k(self) -> int:
        k = 0
        while (self.N << k) < self.A and k < 24: k += 1
        return k
    def update(self, u: int):
        self.A += u; self.N += 1
        if self.N >= 64: self.A >>= 1; self.N >>= 1

def _zz(n: int) -> int: return (n << 1) if n >= 0 else ((-n) << 1)-1
def _unzz(u: int) -> int: return (u >> 1) if not (u & 1) else -((u+1) >> 1)

def rice_write(w: BitWriter, ctx: RiceCtx, u: int):
    k = ctx.k(); q = u >> k
    if q < ESC_Q:
        w.unary(q); w.bits(u & ((1 << k)-1), k) if k else None
    else:
        w.unary(ESC_Q); w.bits(u, 16)
    ctx.update(u)

def rice_read(r: BitReader, ctx: RiceCtx) -> int:
    k = ctx.k(); q = r.unary()
    u = r.bits(16) if q >= ESC_Q else ((q << k) | (r.bits(k) if k else 0))
    ctx.update(u)
    return u

# ---------------------------------------------------------------- peak picker
def pick_peaks(u8, n=40):
    """Identical to pipeline.pick_peaks. Runs on the DECODED reference on both
    sides, which is what makes the table free."""
    v = u8.astype(np.int32)
    c = np.where((v[1:-1] >= v[:-2]) & (v[1:-1] >= v[2:]))[0]+1
    if not len(c): return np.array([], dtype=int)
    prom = v[c]-np.minimum(v[c-1], v[c+1])
    return np.sort(c[np.argsort(-(v[c]+2*prom))[:n]])

def _contexts(nbands):
    """Fresh per frame on both sides. Adapting across frames would couple a
    frame's decodability to every predecessor; within-frame adaptation costs a
    few bits of warm-up and keeps the blast radius of a lost packet at 1."""
    return dict(g=RiceCtx(2), cnt=RiceCtx(2), pid=RiceCtx(0), pkA=RiceCtx(2),
                pkF=RiceCtx(0), gap=RiceCtx(3),
                val=[RiceCtx(3) for _ in range(nbands)])

# ===================================================================== codec
class Codec2:
    """Drop-in for pipeline.Codec: encode(u8, mad) -> (payload, kind)."""
    # trig is 0.85 against v1's 0.60: v2 anchors are ~2.5x cheaper, so the
    # economics moved - re-anchoring "early" now costs more than it saves
    # almost everywhere. Measured on the fleet: 0.85 saves 3% over 0.60 and
    # is indistinguishable from 1.0, while keeping the drift guard.
    def __init__(self, nbins=768, n_peaks=40, n_bands=16, k_mad=2.5,
                 mad_floor_db=3.0, trig=0.85, mr_gate=0.35, mr_tol=3.0,
                 refs='best'):
        self.nb, self.np_, self.nbands = nbins, n_peaks, n_bands
        self.k, self.floor, self.trig = k_mad, mad_floor_db/Q_DB, trig
        self.mr_gate, self.mr_tol = mr_gate, mr_tol
        self.refs = refs                     # 'anchor' | 'best'
        self.band = np.minimum((np.arange(nbins)*n_bands)//nbins, n_bands-1)
        self.anchor = None                   # decoded anchor (== original: lossless)
        self.prev = None                     # decoded previous frame
        self.abytes = 0
        self.last_breakdown = None           # per-layer bit ledger, residuals only

    # ---- anchor: lossless first-difference + Rice --------------------------
    def encode_anchor(self, u8):
        w = BitWriter()
        w.bits(1, 2); w.bits(0, 6)           # frame type 1
        w.bits(int(u8[0]), 8)
        ctx = [RiceCtx(2) for _ in range(self.nbands)]
        prev = int(u8[0])
        for i in range(1, self.nb):
            cur = int(u8[i])
            rice_write(w, ctx[self.band[i]], _zz(cur-prev))
            prev = cur
        pl = w.finish()
        self.anchor = u8.astype(np.int32).copy()
        self.prev = self.anchor.copy()
        self.abytes = len(pl)
        return pl

    # ---- residual against one reference ------------------------------------
    def _residual_against(self, u8, mad, ref, ref_flag):
        a = ref; c = u8.astype(np.int32); d = c-a
        live = a > np.percentile(a, 25)
        if not live.any(): live = np.ones(self.nb, dtype=bool)
        g0 = int(np.clip(np.round(np.median(d[live])), -127, 127)); d0 = d-g0
        gb = np.array([int(np.clip(np.round(np.median(d0[self.band == b])), -127, 127))
                       for b in range(self.nbands)], dtype=np.int32)
        d1 = d0-gb[self.band]

        peaks = pick_peaks(np.clip(a, 0, 255).astype(np.uint8), self.np_)
        pk_rec = []; matched = 0
        pred = np.zeros(self.nb, dtype=np.int32)
        for pid, p in enumerate(peaks[:255]):
            lo, hi = max(0, p-2), min(self.nb, p+3)
            j = lo+int(np.argmax(c[lo:hi]))
            damp = int(np.clip(c[j]-(a[p]+g0+gb[self.band[p]]), -127, 127))
            dfr = int(j-p)
            if abs(damp)*Q_DB < self.mr_tol: matched += 1
            if abs(damp) < 3 and dfr == 0: continue
            pk_rec.append((pid, damp, dfr)); pred[j] += damp
        mr = matched/max(len(peaks), 1)

        r = d1-pred
        thr = np.maximum(self.k*mad, self.floor)
        act = np.where(np.abs(r) > thr)[0]

        w = BitWriter()
        w.bits(2, 2); w.bits(ref_flag, 1); w.bits(0, 5)   # frame type 2 + ref
        ctx = _contexts(self.nbands)
        rice_write(w, ctx['g'], _zz(g0))
        for b in range(self.nbands): rice_write(w, ctx['g'], _zz(int(gb[b])))
        b_gains = len(w.buf)*8+w.nb
        rice_write(w, ctx['cnt'], len(pk_rec))
        last = -1
        for pid, damp, dfr in pk_rec:
            rice_write(w, ctx['pid'], pid-last-1); last = pid
            rice_write(w, ctx['pkA'], _zz(damp))
            rice_write(w, ctx['pkF'], _zz(dfr))
        b_peaks = len(w.buf)*8+w.nb
        rice_write(w, ctx['cnt'], len(act))
        prev = -1
        for i in act:
            rice_write(w, ctx['gap'], int(i-prev-1)); prev = int(i)
            rice_write(w, ctx['val'][self.band[i]], _zz(int(r[i])))
        pl = w.finish()
        self.last_breakdown = dict(gains_bits=b_gains-8, peak_bits=b_peaks-b_gains,
                                   act_bits=len(pl)*8-b_peaks, n_peaks=len(pk_rec),
                                   n_act=len(act))

        # decoded frame, mirrored exactly (needed as the next chain reference)
        rec = a+g0+gb[self.band]
        for pid, damp, dfr in pk_rec:
            j = int(np.clip(peaks[pid]+dfr, 0, self.nb-1)); rec[j] += damp
        rec[act] += r[act]
        return pl, mr, np.clip(rec, 0, 255)

    def encode(self, u8, mad):
        if self.anchor is None:
            return self.encode_anchor(u8), 'anchor'
        pa, mra, ra = self._residual_against(u8, mad, self.anchor, 0)
        best, mr, rec, kind = pa, mra, ra, 'residual'
        if self.refs == 'best' and self.prev is not None:
            pp, mrp, rp = self._residual_against(u8, mad, self.prev, 1)
            if len(pp) < len(best): best, mr, rec, kind = pp, mrp, rp, 'residual_p'
        if len(best) > self.trig*self.abytes or mr < self.mr_gate:
            return self.encode_anchor(u8), 'anchor'
        self.prev = rec.astype(np.int32)
        return best, kind

class Decoder2:
    """Bit-exact mirror. Peak tables are recomputed, never parsed."""
    def __init__(self, nbins=768, n_peaks=40, n_bands=16):
        self.nb, self.np_, self.nbands = nbins, n_peaks, n_bands
        self.band = np.minimum((np.arange(nbins)*n_bands)//nbins, n_bands-1)
        self.anchor = None; self.prev = None

    def decode(self, p: bytes):
        r = BitReader(p)
        ftype = r.bits(2)
        if ftype == 1:
            r.bits(6)
            first = r.bits(8)
            ctx = [RiceCtx(2) for _ in range(self.nbands)]
            out = np.empty(self.nb, dtype=np.int32); out[0] = first
            cur = first
            for i in range(1, self.nb):
                cur += _unzz(rice_read(r, ctx[self.band[i]]))
                out[i] = cur
            self.anchor = out.copy(); self.prev = out.copy()
            return np.clip(out, 0, 255).astype(np.uint8), 'anchor'
        ref_flag = r.bits(1); r.bits(5)
        a = self.prev if ref_flag else self.anchor
        ctx = _contexts(self.nbands)
        g0 = _unzz(rice_read(r, ctx['g']))
        gb = np.array([_unzz(rice_read(r, ctx['g'])) for _ in range(self.nbands)],
                      dtype=np.int32)
        rec = a+g0+gb[self.band]
        peaks = pick_peaks(np.clip(a, 0, 255).astype(np.uint8), self.np_)
        npk = rice_read(r, ctx['cnt'])
        last = -1
        for _ in range(npk):
            pid = last+1+rice_read(r, ctx['pid']); last = pid
            damp = _unzz(rice_read(r, ctx['pkA']))
            dfr = _unzz(rice_read(r, ctx['pkF']))
            if pid < len(peaks):
                j = int(np.clip(peaks[pid]+dfr, 0, self.nb-1)); rec[j] += damp
        nact = rice_read(r, ctx['cnt'])
        prev = -1
        for _ in range(nact):
            prev += 1+rice_read(r, ctx['gap'])
            v = _unzz(rice_read(r, ctx['val'][self.band[prev]]))
            if 0 <= prev < self.nb: rec[prev] += v
        self.prev = np.clip(rec, 0, 255).astype(np.int32)
        return np.clip(rec, 0, 255).astype(np.uint8), \
            ('residual_p' if ref_flag else 'residual')
