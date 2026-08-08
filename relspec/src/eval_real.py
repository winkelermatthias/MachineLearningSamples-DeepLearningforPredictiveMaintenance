"""Codec v1 vs v2 across three public rigs.

Per machine state (group), windows are consumed in recorded order, exactly as
a device would see them, and each codec keeps its own GOP state and its own
MAD ledger. Reported per dataset:

  bytes/frame for: raw Welch f32, uint8 order bins, zlib-9 on those bins,
                   codec v1, codec v2 (anchor refs), codec v2 (best refs)
  named-line amplitude error through v2 (the distortion side of the ratio)
  pattern-energy recovery orig vs decoded (CWRU, both codecs)

zlib is the honesty baseline: a general-purpose compressor with none of the
codec's structure. If the codec cannot beat it soundly, the codec is
ceremony. v1 could not run at all on SEU (fixed band above Nyquist); v2's
numbers there come from the kurtogram band.
"""
import sys, os, json, time, zlib, argparse
sys.path.insert(0, os.path.dirname(__file__) or '.')
import numpy as np
from relspec.pipeline import (Codec, Decoder, CENTERS, ENV_CENTERS, to_amp,
                              Q_DB, NBINS, ENV_BINS)
from relspec.codec2 import Codec2, Decoder2
from relspec.pipeline2 import extract2
from relspec.patterns import extract_patterns
from relspec import datasets as DS

def line(spec, centers, order, tol=0.10):
    m = np.abs(centers-order) < max(tol, 0.012*order)
    return float(spec[m].max()) if m.any() else 0.0

class Rail:
    """One codec instance pair (both rails) plus its MAD state."""
    def __init__(self, kind):
        self.kind = kind
        if kind == 'v1':
            self.ca, self.da = Codec(), Decoder()
            self.ce = Codec(nbins=ENV_BINS, n_peaks=32, n_bands=12)
            self.de = Decoder(nbins=ENV_BINS, n_bands=12)
        else:
            refs = 'best' if kind == 'v2best' else 'anchor'
            self.ca = Codec2(nbins=NBINS, n_bands=16, refs=refs)
            self.da = Decoder2(nbins=NBINS, n_bands=16)
            self.ce = Codec2(nbins=ENV_BINS, n_peaks=32, n_bands=12, refs=refs)
            self.de = Decoder2(nbins=ENV_BINS, n_peaks=32, n_bands=12)
        self.mad_a = np.full(NBINS, 4.0); self.mad_e = np.full(ENV_BINS, 4.0)

    def step(self, e):
        pa, ka = self.ca.encode(e.acc_u8, self.mad_a)
        pe, ke = self.ce.encode(e.env_u8, self.mad_e)
        ra, _ = self.da.decode(pa); re_, _ = self.de.decode(pe)
        self.mad_a = np.maximum(0.95*self.mad_a +
                                0.05*np.abs(e.acc_u8.astype(float)-ra.astype(float)), 1.0)
        self.mad_e = np.maximum(0.95*self.mad_e +
                                0.05*np.abs(e.env_u8.astype(float)-re_.astype(float)), 1.0)
        return len(pa)+len(pe), f'{ka}/{ke}', ra, re_

def pattern_loss(o_amp, r_amp, centers, **kw):
    _, _, bo = extract_patterns(o_amp, centers, **kw)
    _, _, br = extract_patterns(r_amp, centers, **kw)
    rows = []
    for p in bo['patterns']:
        if p['energy'] <= 0: continue
        m = None
        for q in br['patterns']:
            if q['kind'] == p['kind'] and \
               abs(q['key']-p['key']) < 0.05*max(abs(p['key']), 1.0):
                m = q; break
        rows.append(dict(kind=p['kind'], share=p['share'],
                         recovered=m is not None,
                         err_db=(10*np.log10(max(m['energy'], 1e-30) /
                                             max(p['energy'], 1e-30)) if m else None)))
    return rows

def run_dataset(recs, win_s, n_win, do_patterns, tag):
    frames = []; lines_rows = []; loss = {'v1': [], 'v2best': []}
    t0 = time.time()
    for rec in recs:
        fs, x = rec['fs'], rec['x']
        n = int(win_s*fs)
        nw = min(n_win, len(x)//n)
        if nw == 0: continue
        rails = {k: Rail(k) for k in ('v1', 'v2anchor', 'v2best')}
        nps = DS.nperseg_for(fs)
        raw_b = 2*(nps//2+1)*4
        for w in range(nw):
            seg = x[w*n:(w+1)*n]
            e = extract2(seg, fs, band_key=rec['group'],
                         fr_nominal=rec['fr_nominal'])
            u8cat = np.concatenate([e.acc_u8, e.env_u8]).tobytes()
            row = dict(dataset=rec['dataset'], state=rec['state'],
                       group=rec['group'], w=w, fr=e.fr, tier=e.tier,
                       raw=raw_b, u8=len(u8cat),
                       gz=len(zlib.compress(u8cat, 9)))
            dec = {}
            for k, rail in rails.items():
                b, kind, ra, re_ = rail.step(e)
                row[k] = b; row[k+'_kind'] = kind
                dec[k] = (ra, re_)
            frames.append(row)
            ra, re_ = dec['v2best']
            ao, eo = to_amp(e.acc_u8), to_amp(e.env_u8)
            ar, er = to_amp(ra), to_amp(re_)
            for nm, o in {**rec['lines'], '1x': 1.0, '2x': 2.0, '3x': 3.0}.items():
                for railnm, O, R, cen in (('acc', ao, ar, CENTERS),
                                          ('env', eo, er, ENV_CENTERS)):
                    if o > cen[-1]*0.95: continue
                    a1 = line(O, cen, o)
                    if a1 <= 0: continue
                    lines_rows.append(dict(dataset=rec['dataset'],
                        state=rec['state'], name=nm, rail=railnm,
                        err_db=float(20*np.log10(max(line(R, cen, o), 1e-12)/a1))))
            if do_patterns:
                r1a, r1e = dec['v1']
                loss['v1'] += pattern_loss(ao, to_amp(r1a), CENTERS)
                loss['v1'] += pattern_loss(eo, to_amp(r1e), ENV_CENTERS,
                                           min_harmonics=3, f_hi=6.5)
                loss['v2best'] += pattern_loss(ao, ar, CENTERS)
                loss['v2best'] += pattern_loss(eo, er, ENV_CENTERS,
                                               min_harmonics=3, f_hi=6.5)
    print(f'[{tag}] {len(frames)} frames in {time.time()-t0:.0f}s', flush=True)
    return frames, lines_rows, loss

def summarise(frames, lines_rows, loss, tag):
    import pandas as pd
    F = pd.DataFrame(frames)
    if F.empty: return {}
    out = {'frames': len(F)}
    print(f'\n===== {tag}: {len(F)} frames, {F.group.nunique()} machine states =====')
    print(f'{"method":<10}{"B/frame":>9}{"vs raw":>9}{"vs u8":>8}')
    for k in ('raw', 'u8', 'gz', 'v1', 'v2anchor', 'v2best'):
        b = F[k].mean()
        out[k] = float(b)
        print(f'{k:<10}{b:>9.0f}{F.raw.mean()/b:>8.1f}x{F.u8.mean()/b:>7.1f}x')
    if lines_rows:
        L = pd.DataFrame(lines_rows)
        g = L.groupby(['rail', 'name']).err_db.agg(
            n='size', med=lambda s: np.median(np.abs(s)),
            p95=lambda s: np.percentile(np.abs(s), 95))
        print('\nnamed-line |err| dB through v2best:')
        print(g.round(3).to_string())
        out['lines'] = {f'{r}_{n}': dict(med=float(v['med']), p95=float(v['p95']))
                        for (r, n), v in g.iterrows()}
    for ck, rows in loss.items():
        if not rows: continue
        R = pd.DataFrame(rows)
        wr = R[R.recovered].share.sum()/max(R.share.sum(), 1e-12)*100
        errs = R.err_db.dropna().abs()
        print(f'pattern energy via {ck}: {wr:.2f}% recovered '
              f'({100*R.recovered.mean():.1f}% of count), '
              f'med |err| {errs.median() if len(errs) else float("nan"):.3f} dB')
        out[f'loss_{ck}'] = dict(energy_recovered_pct=float(wr),
                                 med_err_db=float(errs.median()) if len(errs) else None)
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                    default=['cwru', 'mfpt', 'seu_bearing', 'seu_gear'])
    ap.add_argument('--win', type=float, default=2.0)
    ap.add_argument('--nw', type=int, default=8)
    ap.add_argument('--out', default='../eval_real.json')
    args = ap.parse_args()
    all_out = {}
    for ds in args.datasets:
        if ds == 'cwru': recs = DS.cwru_records()
        elif ds == 'mfpt': recs = DS.mfpt_records()
        elif ds == 'seu_bearing': recs = DS.seu_records('bearingset')
        elif ds == 'seu_gear': recs = DS.seu_records('gearset')
        else: raise SystemExit(f'unknown dataset {ds}')
        fr, ln, lo = run_dataset(recs, args.win, args.nw,
                                 do_patterns=(ds == 'cwru'), tag=ds)
        all_out[ds] = summarise(fr, ln, lo, ds)
        all_out[ds+'_frames'] = fr
    json.dump(all_out, open(args.out, 'w'), default=float)
    print(f'\nwrote {args.out}')
