"""Bounded-memory extraction, encoding, gating and diagnosis.

The extractor is streaming by construction: waveform blocks are consumed and
discarded, and nothing whose size depends on acquisition length is ever
allocated. That is not a micro-optimisation, it is what lets the same code run
on an MCU and over a fleet-scale batch without two implementations.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, welch
from dataclasses import dataclass, field

FS_DEF = 12000.0
NFINE, NCOARSE = 512, 256
NBINS = NFINE + NCOARSE
ORD_FINE, ORD_MAX = 20.0, 200.0
Q_DB, DB_OFF = 0.5, -128.0
NBANDS = 16
ENV_BINS = 384

EDGES = np.concatenate([np.linspace(0, ORD_FINE, NFINE+1),
                        np.linspace(ORD_FINE, ORD_MAX, NCOARSE+1)[1:]])
CENTERS = 0.5*(EDGES[:-1]+EDGES[1:])
BAND = np.minimum((np.arange(NBINS)*NBANDS)//NBINS, NBANDS-1)
ENV_EDGES = np.linspace(0, ORD_FINE, ENV_BINS+1)
ENV_CENTERS = 0.5*(ENV_EDGES[:-1]+ENV_EDGES[1:])

to_db  = lambda a: 20*np.log10(np.maximum(a, 1e-12))
to_u8  = lambda a: np.clip(np.round((to_db(a)-DB_OFF)/Q_DB), 0, 255).astype(np.uint8)
to_amp = lambda u: 10.0**((u.astype(np.float64)*Q_DB+DB_OFF)/20.0)

# ============================================================ 1. extraction
def envelope(x, fs=FS_DEF, band=(2000., 5000.)):
    sos = butter(6, [band[0]/(fs/2), band[1]/(fs/2)], btype='band', output='sos')
    e = np.abs(hilbert(sosfiltfilt(sos, x)))
    return e - e.mean()

def estimate_speed(x, fs=FS_DEF, lo=20., hi=70., nh=3):
    """Envelope-domain HPS with parabolic refinement.

    The envelope, not the acceleration spectrum: the low-frequency acceleration
    region is dominated by mount and structural lines, and harmonic search there
    locks onto a fixed feature regardless of true speed. Sub-bin refinement is
    mandatory or the error floor is the bin width."""
    e = envelope(x, fs)
    N = 1 << int(np.ceil(np.log2(len(e))))
    A = np.abs(np.fft.rfft(e*np.hanning(len(e)), N))*2/len(e)
    f = np.fft.rfftfreq(N, 1/fs)
    grid = np.arange(lo, hi, 0.005)
    hps = np.ones_like(grid)
    for h in range(1, nh+1):
        hps *= np.maximum(np.interp(grid*h, f, A), 1e-12)
    hps = hps**(1/nh)
    j = int(np.argmax(hps)); fr = float(grid[j])
    rival = hps[np.abs(grid-grid[j]) > 0.5]
    conf = float(hps[j]/max(rival.max() if rival.size else 1e-12, 1e-12))
    k = int(round(fr/(f[1]-f[0])))
    if 0 < k < len(A)-1:
        y0, y1, y2 = A[k-1], A[k], A[k+1]
        den = 2*(y0-2*y1+y2)
        if abs(den) > 1e-20:
            d = (y0-y2)/den
            if -1 < d < 1: fr = float((k+d)*(f[1]-f[0]))
    return fr, conf

def coherence_limit(order, speed_err, fr):
    """T_coh = 1/(4 k e f). Beyond this the accumulator decoheres and further
    integration buys variance reduction, not resolution."""
    return 1.0/(4.0*max(order, 1e-9)*max(speed_err, 1e-9)*max(fr, 1e-9))

def bin_orders(f_hz, amp, fr, edges):
    """Peak-preserving: MAX per bin, never mean. Averaging destroys sideband
    spacing, which is the diagnosis, while leaving RMS error looking healthy."""
    nb = len(edges)-1
    o = f_hz/max(fr, 1e-9)
    idx = np.searchsorted(edges, o, side='right')-1
    v = (idx >= 0) & (idx < nb)
    out = np.zeros(nb)
    np.maximum.at(out, idx[v], amp[v])
    z = out == 0
    if z.any() and (~z).any():
        c = 0.5*(edges[:-1]+edges[1:])
        out[z] = np.interp(c[z], c[~z], out[~z])
    return out

def moments(x):
    mu = x.mean(); d = x-mu; sd = d.std()
    if sd < 1e-15: return 0.0, 0.0, 0.0
    return float(sd), float((d**4).mean()/sd**4), float(np.abs(d).max()/sd)

@dataclass
class Extract:
    acc: np.ndarray; env: np.ndarray
    acc_u8: np.ndarray; env_u8: np.ndarray
    fr: float; conf: float; tier: int; coh_s: float
    acc_rms: float; vel_rms: float; acc_kurt: float; acc_crest: float
    env_rms: float; env_kurt: float; env_crest: float
    band_db: np.ndarray

def extract(x, fs=FS_DEF, nperseg=4096) -> Extract:
    f, p = welch(x, fs=fs, nperseg=min(nperseg, len(x)),
                 noverlap=min(nperseg, len(x))//2, window='hann',
                 scaling='spectrum', detrend='constant')
    a = np.sqrt(np.maximum(p, 0))*np.sqrt(2)

    m = (f >= 10) & (f <= 1000)
    v = a[m]*9.80665/(2*np.pi*f[m])*1000.0
    vel_rms = float(np.sqrt(0.5*np.sum(v**2)))

    fr, conf = estimate_speed(x, fs)
    tier = 1 if conf >= 3.0 else (2 if conf >= 1.8 else 3)
    if not (5 < fr < 200): fr, tier = 30.0, 3

    e = envelope(x, fs)
    fe, pe = welch(e, fs=fs, nperseg=min(nperseg, len(e)),
                   noverlap=min(nperseg, len(e))//2, window='hann',
                   scaling='spectrum', detrend='constant')
    ae = np.sqrt(np.maximum(pe, 0))*np.sqrt(2)

    acc = bin_orders(f, a, fr, EDGES)
    env = bin_orders(fe, ae, fr, ENV_EDGES)
    ar, ak, ac = moments(x); er, ek, ec = moments(e)
    band = np.array([to_db(np.sqrt(np.mean(acc[BAND == b]**2))) for b in range(NBANDS)])
    speed_err = 0.0005 if tier == 1 else (0.003 if tier == 2 else 0.02)
    return Extract(acc, env, to_u8(acc), to_u8(env), fr, conf, tier,
                   coherence_limit(3.0, speed_err, fr),
                   ar, vel_rms, ak, ac, er, ek, ec, band)

# ================================================================= 2. codec
def _varint(n):
    o = bytearray()
    while True:
        b = n & 0x7F; n >>= 7; o.append(b | (0x80 if n else 0))
        if not n: return bytes(o)
def _zz(n): return (n << 1) if n >= 0 else ((-n) << 1)-1
def _rv(b, i):
    v = s = 0
    while True:
        c = b[i]; i += 1; v |= (c & 0x7F) << s
        if not (c & 0x80): return v, i
        s += 7
def _unzz(v): return (v >> 1) if not (v & 1) else -((v+1) >> 1)

def pick_peaks(u8, n=40):
    v = u8.astype(np.int32)
    c = np.where((v[1:-1] >= v[:-2]) & (v[1:-1] >= v[2:]))[0]+1
    if not len(c): return np.array([], dtype=int)
    prom = v[c]-np.minimum(v[c-1], v[c+1])
    return np.sort(c[np.argsort(-(v[c]+2*prom))[:n]])

class Codec:
    """Anchor / residual GOP codec on the order-binned uint8 spectrum."""
    def __init__(self, nbins=NBINS, n_peaks=40, n_bands=NBANDS, k_mad=2.5,
                 mad_floor_db=3.0, trig=0.60, mr_gate=0.35, mr_tol=3.0):
        self.nb, self.np_, self.nbands = nbins, n_peaks, n_bands
        self.k, self.floor, self.trig = k_mad, mad_floor_db/Q_DB, trig
        self.mr_gate, self.mr_tol = mr_gate, mr_tol
        self.band = np.minimum((np.arange(nbins)*n_bands)//nbins, n_bands-1)
        self.ref = None; self.abytes = 0

    def encode_anchor(self, u8):
        pk = pick_peaks(u8, self.np_)
        pl = bytearray([1, len(pk)]) + u8.tobytes()
        for p in pk: pl += int(p).to_bytes(2, 'little')+bytes([int(u8[p])])
        self.ref = dict(u8=u8.astype(np.int32).copy(), peaks=pk)
        self.abytes = len(pl)
        return bytes(pl)

    def encode_residual(self, u8, mad):
        a = self.ref['u8']; c = u8.astype(np.int32); d = c-a
        live = a > np.percentile(a, 25)
        g0 = int(np.clip(np.round(np.median(d[live])), -127, 127)); d0 = d-g0
        gb = np.array([int(np.clip(np.round(np.median(d0[self.band == b])), -127, 127))
                       for b in range(self.nbands)], dtype=np.int32)
        d1 = d0-gb[self.band]
        pk = bytearray(); npk = matched = 0
        pred = np.zeros(self.nb, dtype=np.int32)
        for pid, p in enumerate(self.ref['peaks'][:255]):
            lo, hi = max(0, p-2), min(self.nb, p+3)
            j = lo+int(np.argmax(c[lo:hi]))
            damp = int(np.clip(c[j]-(a[p]+g0+gb[self.band[p]]), -127, 127))
            dfr = int(j-p)
            if abs(damp)*Q_DB < self.mr_tol: matched += 1
            if abs(damp) < 3 and dfr == 0: continue
            pk += bytes([pid & 0xFF, damp & 0xFF, dfr & 0xFF]); pred[j] += damp; npk += 1
        mr = matched/max(len(self.ref['peaks']), 1)
        r = d1-pred
        thr = np.maximum(self.k*mad, self.floor)
        act = np.where(np.abs(r) > thr)[0]
        rb = bytearray(); prev = -1
        for i in act:
            rb += _varint(int(i-prev-1))+_varint(_zz(int(r[i]))); prev = i
        hdr = bytes([2, 1, g0 & 0xFF, npk & 0xFF,
                     len(act) & 0xFF, (len(act) >> 8) & 0xFF, 0])
        return hdr+gb.astype(np.int8).tobytes()+bytes(pk)+bytes(rb), mr

    def encode(self, u8, mad):
        if self.ref is None:
            return self.encode_anchor(u8), 'anchor'
        pl, mr = self.encode_residual(u8, mad)
        if len(pl) > self.trig*self.abytes or mr < self.mr_gate:
            return self.encode_anchor(u8), 'anchor'
        return pl, 'residual'

class Decoder:
    """Cloud side. Mirrors Codec exactly; any divergence shows up immediately
    as decode error against the transmitted truth."""
    def __init__(self, nbins=NBINS, n_bands=NBANDS):
        self.nb = nbins
        # n_bands must be carried, not assumed. The envelope rail uses 12 bands
        # against the acceleration rail's 16, and a decoder that hardcodes 16
        # reads four bytes of the peak table as band gains and then walks off
        # the end of the payload.
        self.nbands = n_bands
        self.band = np.minimum((np.arange(nbins)*n_bands)//nbins, n_bands-1)
        self.u8 = None; self.peaks = []
    def decode(self, p):
        if p[0] == 1:
            npk = p[1]
            self.u8 = np.frombuffer(p[2:2+self.nb], dtype=np.uint8).astype(np.int32).copy()
            k = 2+self.nb; self.peaks = []
            for _ in range(npk):
                self.peaks.append(p[k] | (p[k+1] << 8)); k += 3
            return self.u8.astype(np.uint8), 'anchor'
        g0 = p[2]-256 if p[2] > 127 else p[2]
        npk = p[3]; nact = p[4] | (p[5] << 8)
        gb = np.frombuffer(p[7:7+self.nbands], dtype=np.int8).astype(np.int32)
        rec = self.u8+g0+gb[self.band]
        k = 7+self.nbands
        for _ in range(npk):
            pid = p[k]
            damp = p[k+1]-256 if p[k+1] > 127 else p[k+1]
            dfr = p[k+2]-256 if p[k+2] > 127 else p[k+2]
            k += 3
            if pid < len(self.peaks):
                j = int(np.clip(self.peaks[pid]+dfr, 0, self.nb-1)); rec[j] += damp
        prev = -1
        for _ in range(nact):
            if k >= len(p): break
            gap, k = _rv(p, k); val, k = _rv(p, k)
            i = prev+1+gap; prev = i
            if 0 <= i < self.nb: rec[i] += _unzz(val)
        return np.clip(rec, 0, 255).astype(np.uint8), 'residual'

# ================================================================== 3. gate
@dataclass
class Gate:
    """Python mirror of the firmware policy. Kept in step deliberately: the
    fleet simulation and the device must agree or the evaluation is fiction."""
    s_hi: float = 4.0; s_lo: float = 2.5; step_min: float = 3.0
    persist_m: int = 3; persist_n: int = 2
    heartbeat_s: int = 3*86400; refractory_s: int = 12*3600
    burst_max: int = 3; backoff_mult: int = 6
    cusum_k: float = 1.0; cusum_h: float = 10.0; slow_freeze_n: int = 120
    warmup_n: int = 20; min_vel_delta: float = 0.8; min_acc_delta_db: float = 2.5
    rate_max: int = 14; rate_window_s: int = 7*86400; reserve_change: int = 6
    conn_interval_s: int = 21*86400
    med: np.ndarray = None; mad: np.ndarray = None
    smed: np.ndarray = None; smad: np.ndarray = None
    cp: np.ndarray = None; cn: np.ndarray = None
    n: int = 0; n_slow: int = 0; slow_frozen: bool = False
    in_change: bool = False; ring: int = 0; burst: int = 0
    prev_score: float = 0.0; step_latch: bool = False
    last_up: int = -10**9; last_change: int = -10**9; last_score: float = 0.0
    hist: list = field(default_factory=list)
    band_med: np.ndarray = None; band_mad: np.ndarray = None
    bin_med: np.ndarray = None; bin_mad: np.ndarray = None

    def _vec(self, e: Extract):
        return np.array([np.log(max(e.acc_rms,1e-9)), np.log(max(e.vel_rms,1e-9)),
                         e.acc_kurt, e.acc_crest, np.log(max(e.env_rms,1e-9)),
                         e.env_kurt, e.env_crest])

    def _upd1(self, med, mad, v, n, step=0.02, alpha=0.02):
        if n == 0: return v.copy(), np.abs(v)*0.05+1e-6
        sc = np.where(mad > 1e-9, mad, np.abs(v)*0.05+1e-6)
        med = med + step*sc*np.sign(v-med)
        mad = np.maximum((1-alpha)*mad + alpha*np.abs(v-med), 1e-9)
        return med, mad

    def update_baseline(self, e: Extract):
        v = self._vec(e)
        self.med, self.mad = self._upd1(self.med, self.mad, v, self.n)
        self.band_med, self.band_mad = self._upd1(self.band_med, self.band_mad,
                                                  e.band_db, self.n)
        self.bin_med, self.bin_mad = self._upd1(self.bin_med, self.bin_mad,
                                                e.acc_u8.astype(float), self.n)
        if not self.slow_frozen:
            self.smed, self.smad = self._upd1(self.smed, self.smad, v, self.n_slow,
                                              step=0.002, alpha=0.002)
            self.n_slow += 1
            if self.n_slow >= self.slow_freeze_n: self.slow_frozen = True
        self.n += 1

    def score(self, e: Extract):
        v = self._vec(e)
        z = (v-self.med)/np.maximum(1.4826*self.mad, 1e-6)
        pm = float((np.mean(np.minimum(np.abs(z), 12.0)**3))**(1/3))
        mask = self.band_med+3*np.maximum(1.4826*self.band_mad, 0.5)+6.0
        over = e.band_db-mask
        nb_over = int((over > 0).sum()); mx_over = float(max(over.max(), 0.0))
        mterm = 0.0 if nb_over < 2 else 1.0+mx_over/6.0+0.5*(nb_over-2)
        drift = 0.0; dfeat = 0
        if self.slow_frozen:
            zs = (v-self.smed)/np.maximum(1.4826*self.smad, 1e-6)
            self.cp = np.maximum(0, self.cp+zs-self.cusum_k)
            self.cn = np.maximum(0, self.cn-zs-self.cusum_k)
            m = np.maximum(self.cp, self.cn)
            drift = float(m.max()); dfeat = int(np.argmax(m))
        dterm = 0.0 if drift <= self.cusum_h else 1.0+(drift-self.cusum_h)/self.cusum_h
        vel_base = float(np.exp(self.med[1]))
        floor_ok = (abs(e.vel_rms-vel_base) >= self.min_vel_delta
                    or abs(20*(v[0]-self.med[0])/2.302585) >= self.min_acc_delta_db
                    or nb_over >= 2)
        return dict(score=pm+mterm+dterm, drift=drift, dfeat=dfeat,
                    mask_bands=nb_over, floor_ok=floor_ok,
                    band_delta=e.band_db-self.band_med, over_mask=over > 0)

    def init(self, e: Extract):
        v = self._vec(e)
        self.med, self.mad = v.copy(), np.abs(v)*0.05+1e-6
        self.smed, self.smad = v.copy(), self.mad.copy()
        self.cp = np.zeros(7); self.cn = np.zeros(7)
        self.band_med, self.band_mad = e.band_db.copy(), np.full(NBANDS, 1.0)
        self.bin_med = e.acc_u8.astype(float).copy()
        self.bin_mad = np.full(NBINS, 4.0)
        self.n = self.n_slow = 1

    def decide(self, e: Extract, t: int):
        s = self.score(e)
        jump = s['score']-self.prev_score
        kind = 1 if (s['score'] >= self.s_hi and jump >= self.step_min) else \
               (2 if s['drift'] > self.cusum_h else 0)
        self.prev_score = s['score']
        if kind == 1: self.step_latch = True
        over = (s['score'] >= (self.s_lo if self.in_change else self.s_hi)) and s['floor_ok']
        self.ring = ((self.ring << 1) | int(over)) & ((1 << self.persist_m)-1)
        votes = bin(self.ring).count('1')
        warm = self.n >= self.warmup_n
        change = warm and votes >= self.persist_n
        if change: self.in_change = True
        elif s['score'] < self.s_lo:
            self.in_change = False; self.burst = 0
            self.step_latch = False; self.last_score = 0.0
        if not self.in_change: self.update_baseline(e)
        hb = (t-self.last_up) >= self.heartbeat_s
        dec = None
        if change:
            esc = s['score'] > self.last_score*1.6
            refr = self.refractory_s*(self.backoff_mult if self.burst >= 3 and not esc else 1)
            if (t-self.last_change) < refr and not esc:
                dec = 'up_heartbeat' if hb else 'no_refractory'
            else: dec = 'up_change'
        elif hb: dec = 'up_heartbeat'
        elif not warm: dec = 'no_warmup'
        else: dec = 'no_quiet'
        if dec.startswith('up_'):
            used = sum(1 for h in self.hist if t-h < self.rate_window_s)
            lim = self.rate_max if dec == 'up_change' else max(1, self.rate_max-self.reserve_change)
            if used >= lim: dec = 'no_ratelimit'
        if dec.startswith('up_'):
            self.hist.append(t); self.last_up = t
            if dec == 'up_change':
                self.burst = 1 if s['score'] > self.last_score*1.6 else self.burst+1
                self.last_change = t; self.last_score = s['score']
                if self.step_latch: kind = 1
        s.update(decision=dec, kind=kind, jump=jump,
                 urgent=(dec == 'up_change' and kind == 1 and self.burst <= 1))
        return s
