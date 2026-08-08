"""s10_cloud.py - cloud side.

1. Decode the raw payloads the firmware transmitted (anchor / residual GOP codec).
2. Score reconstruction against the quantised truth the device saw.
3. Detect harmonic families and sideband families on the reconstructed spectra,
   and check that the same families are found on the originals.

Harmonic and sideband identification is deliberately cloud-side: it needs a
search over candidate fundamentals and spacings that the MCU has no budget for,
and it needs no extra bytes because the reconstruction is already exact enough.
"""
import numpy as np, struct, os

NBINS, NFINE, ORD_FINE, ORD_MAX, NBANDS = 768, 512, 20.0, 200.0, 16
Q_DB, DB_OFF = 0.5, -128.0
BEARING = dict(BPFO=3.5848, BPFI=5.4152, BSF=4.7135, FTF=0.3983)

def edges():
    return np.concatenate([np.linspace(0, ORD_FINE, NFINE+1),
                           np.linspace(ORD_FINE, ORD_MAX, NBINS-NFINE+1)[1:]])
EDG = edges(); CEN = 0.5*(EDG[:-1]+EDG[1:])
BAND = np.minimum((np.arange(NBINS)*NBANDS)//NBINS, NBANDS-1)

def u8_to_amp(u): return 10.0**((u.astype(np.float64)*Q_DB + DB_OFF)/20.0)

# ------------------------------------------------------------------ decoder
def read_varint(b, i):
    v, s = 0, 0
    while True:
        c = b[i]; i += 1
        v |= (c & 0x7F) << s
        if not (c & 0x80): return v, i
        s += 7
def unzz(v): return (v >> 1) if not (v & 1) else -((v+1) >> 1)

class Decoder:
    """Mirrors eg_enc_anchor / eg_enc_residual in edge_gate.c."""
    def __init__(self):
        self.u8 = None; self.peaks = []
    def decode(self, p):
        if p[0] == 0x01:                                  # anchor
            npk = p[1]
            self.u8 = np.frombuffer(p[2:2+NBINS], dtype=np.uint8).astype(np.int32).copy()
            self.peaks = []
            k = 2 + NBINS
            for _ in range(npk):
                idx = p[k] | (p[k+1] << 8); self.peaks.append(idx); k += 3
            return self.u8.astype(np.uint8), 'anchor'
        if p[0] != 0x02: raise ValueError('bad frame type')
        tier = p[1]
        g0 = p[2] - 256 if p[2] > 127 else p[2]
        npk = p[3]; nact = p[4] | (p[5] << 8)
        gb = np.frombuffer(p[7:7+NBANDS], dtype=np.int8).astype(np.int32)
        rec = self.u8 + g0 + gb[BAND]
        k = 7 + NBANDS
        step = 2 if tier == 3 else 3
        for _ in range(npk):
            pid = p[k]
            damp = p[k+1] - 256 if p[k+1] > 127 else p[k+1]
            dfr = 0 if step == 2 else (p[k+2] - 256 if p[k+2] > 127 else p[k+2])
            k += step
            if pid < len(self.peaks):
                j = int(np.clip(self.peaks[pid] + dfr, 0, NBINS-1)); rec[j] += damp
        prev = -1
        for _ in range(nact):
            if k >= len(p): break
            gap, k = read_varint(p, k); val, k = read_varint(p, k)
            i = prev + 1 + gap; prev = i
            if 0 <= i < NBINS: rec[i] += unzz(val)
        return np.clip(rec, 0, 255).astype(np.uint8), 'residual'

def load_payloads(path):
    out = []
    with open(path, 'rb') as f:
        while True:
            hdr = f.read(16)
            if len(hdr) < 16: break
            i, t, fr, L = struct.unpack('<IIfI', hdr)
            pl = f.read(L); truth = np.frombuffer(f.read(NBINS), dtype=np.uint8)
            out.append(dict(i=i, t=t, fr=fr, payload=pl, truth=truth))
    return out

# ------------------------------------- harmonic family detection (cloud only)
def _at(spec, order, tol=0.12):
    m = np.abs(CEN - order) < tol
    return spec[m].max() if m.any() else 0.0

def harmonic_families(spec, f_lo=0.7, f_hi=12.0, nh=8, min_score=2.2):
    """Search for combs. Score each candidate fundamental by the geometric mean
    of its harmonic amplitudes divided by the local median floor, so a family is
    only reported if the whole comb stands above the noise, not one loud line."""
    floor = np.median(spec[spec > 0]) + 1e-12
    grid = np.arange(f_lo, f_hi, 0.01)
    score = np.zeros_like(grid); cnt = np.zeros_like(grid)
    for gi, f in enumerate(grid):
        vals, n = [], 0
        for h in range(1, nh+1):
            o = f*h
            if o > ORD_MAX: break
            v = _at(spec, o, tol=max(0.06, 0.01*o))
            if v > 0: vals.append(v); n += 1 if v > 3*floor else 0
        if len(vals) >= 3:
            score[gi] = np.exp(np.mean(np.log(np.maximum(vals, 1e-12))))/floor
            cnt[gi] = n
    def sc_at(f):
        j = int(round((f - f_lo)/0.01))
        return score[j] if 0 <= j < len(score) else 0.0

    fam, used = [], np.zeros_like(grid, dtype=bool)
    for gi in np.argsort(-score):
        if score[gi] < min_score: break
        if used[gi]: continue
        f = float(grid[gi])
        # Reduce to the lowest plausible fundamental. A comb at 12x also scores
        # at 6x, 4x, 3x and 1x; reporting the highest is diagnostically useless
        # because every real family is named by its fundamental.
        for k in (6, 5, 4, 3, 2):
            g = f/k
            if g >= f_lo and sc_at(g) > 0.55*score[gi]:
                f = g; break
        # suppression must be wide enough to swallow the plateau around a peak
        for k in range(1, 13):
            used |= np.abs(grid - f*k) < max(0.15, 0.02*f*k)
        if any(abs(f - x['f0']) < 0.15 for x in fam): continue
        fam.append(dict(f0=round(f, 3), score=round(float(sc_at(f) or score[gi]), 2),
                        n_harm=int(cnt[gi])))
        if len(fam) >= 5: break
    return fam

def sideband_family(spec, carrier, span=3.2, dmin=0.35, dmax=2.5):
    """Find the modulation spacing around a carrier by autocorrelating the local
    log spectrum. Returns spacing in orders and the sideband-to-carrier ratio."""
    m = (CEN > carrier-span) & (CEN < carrier+span)
    if m.sum() < 16: return None
    x = np.log10(np.maximum(spec[m], 1e-12)); x = x - x.mean()
    o = CEN[m]
    du = np.median(np.diff(o))
    ac = np.correlate(x, x, 'full')[len(x)-1:]
    ac = ac/(ac[0]+1e-12)
    lo, hi = int(dmin/du), min(int(dmax/du), len(ac)-1)
    if hi <= lo+1: return None
    j = lo + int(np.argmax(ac[lo:hi]))
    # Autocorrelation of a symmetric carrier-plus-sidebands pattern peaks at the
    # spacing but also at its sub-multiples. Prefer the largest lag whose score
    # is within 15% of the maximum, which is the true spacing.
    best = ac[j]
    cand = [q for q in range(lo, hi) if ac[q] >= 0.85*best]
    if cand: j = max(cand)
    if j <= 0 or j+1 >= len(ac): return None
    y0,y1,y2 = ac[j-1],ac[j],ac[j+1]
    den = 2*(y0-2*y1+y2)
    d = j + ((y0-y2)/den if abs(den) > 1e-12 else 0.0)
    spacing = d*du
    c = _at(spec, carrier)
    sb = 0.5*(_at(spec, carrier-spacing) + _at(spec, carrier+spacing))
    if c <= 0: return None
    return dict(carrier=round(float(carrier),3), spacing=round(float(spacing),3),
                peak_ac=round(float(ac[j]),3),
                scr_db=round(float(20*np.log10(max(sb,1e-12)/c)),2))

def label_family(f0):
    for k, v in BEARING.items():
        if abs(f0 - v) < 0.10: return k
    if abs(f0 - round(f0)) < 0.05 and f0 >= 0.9: return f'{int(round(f0))}x shaft'
    return None

# ------------------------------------------------------------------ main
if __name__ == '__main__':
    import pandas as pd, json
    res = {}
    for scn in ['S4_duty_gradual', 'S3_duty_step_fault', 'S7_incipient_blend']:
        pf = f'scen/{scn}.pl'
        if not os.path.exists(pf): continue
        recs = load_payloads(pf); dec = Decoder()
        rows = []
        for r in recs:
            rec, kind = dec.decode(r['payload'])
            err = np.abs(rec.astype(int) - r['truth'].astype(int))*Q_DB
            o_amp = u8_to_amp(r['truth']); r_amp = u8_to_amp(rec)
            rows.append(dict(i=r['i'], day=r['t']/86400, kind=kind, bytes=len(r['payload']),
                             med=float(np.median(err)), p95=float(np.percentile(err,95)),
                             p99=float(np.percentile(err,99)), mx=float(err.max()),
                             exact=float((err==0).mean()),
                             o=o_amp, rc=r_amp))
        df = pd.DataFrame([{k:v for k,v in r.items() if k not in ('o','rc')} for r in rows])
        print(f"\n=== {scn} : {len(rows)} transmitted frames decoded ===")
        print(f"  anchors {sum(1 for r in rows if r['kind']=='anchor')}  "
              f"residuals {sum(1 for r in rows if r['kind']=='residual')}  "
              f"mean {df.bytes.mean():.0f} B")
        print(f"  reconstruction error dB: median {df.med.median():.2f}  "
              f"p95 {df.p95.median():.2f}  p99 {df.p99.median():.2f}  worst {df.mx.max():.2f}")
        print(f"  bins reconstructed bit-exact: {100*df.exact.mean():.1f}%")
        rr = [r for r in rows if r['kind']=='residual']
        if rr:
            print(f"  residual frames only: median {np.median([r['med'] for r in rr]):.2f} dB, "
                  f"p99 {np.median([r['p99'] for r in rr]):.2f} dB, "
                  f"{np.mean([r['bytes'] for r in rr]):.0f} B avg")

        # harmonic + sideband agreement, original vs reconstructed
        agree, tot, sb_o, sb_r = 0, 0, [], []
        late = rows[len(rows)//2:]
        for r in late[:12]:
            fo = harmonic_families(r['o']); fr_ = harmonic_families(r['rc'])
            so = {round(x['f0'],1) for x in fo}; sr = {round(x['f0'],1) for x in fr_}
            agree += len(so & sr); tot += len(so)
            a = sideband_family(r['o'], BEARING['BPFI'])
            b = sideband_family(r['rc'], BEARING['BPFI'])
            if a and b: sb_o.append(a); sb_r.append(b)
        print(f"  harmonic families recovered on reconstruction: {agree}/{tot}")
        if sb_o:
            ds = np.array([a['spacing'] for a in sb_o]); dr = np.array([b['spacing'] for b in sb_r])
            cs = np.array([a['scr_db'] for a in sb_o]); cr = np.array([b['scr_db'] for b in sb_r])
            print(f"  BPFI sideband spacing: original {ds.mean():.3f} ord, "
                  f"reconstructed {dr.mean():.3f} ord, error {100*np.abs(dr-ds).mean()/ds.mean():.2f}%")
            print(f"  sideband-to-carrier ratio error: {np.abs(cr-cs).mean():.2f} dB")
        last = rows[-1]
        fam = harmonic_families(last['rc'])
        print("  families on the last reconstructed frame:")
        for f in fam:
            lb = label_family(f['f0'])
            print(f"    fundamental {f['f0']:6.3f} orders  score {f['score']:6.2f}  "
                  f"{f['n_harm']} harmonics above floor" + (f"   <- {lb}" if lb else ""))
        res[scn] = dict(rows=[{k:v for k,v in r.items() if k not in ('o','rc')} for r in rows],
                        fam=fam)
    json.dump(res, open('scen/cloud.json','w'), default=float)
    print("\nwrote scen/cloud.json")
