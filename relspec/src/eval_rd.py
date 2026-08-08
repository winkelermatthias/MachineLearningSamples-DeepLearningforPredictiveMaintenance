"""Rate-distortion for codec v2 on real CWRU data.

Two sweeps:

  1. The dead-zone knobs (k_mad, floor_db) at fixed 2 s acquisitions. Rate is
     bytes/frame; distortion is the two numbers an analyst cares about:
     energy-weighted pattern recovery and p95 named-line error.
  2. Acquisition length {1, 2, 4} s at default knobs, because v1 found length
     dominates codec tuning and that claim deserves a v2 re-test with the
     entropy stage in place.

A representative 8-file subset (2 normal + 6 fault states across loads),
8 windows each, both rails, best-refs chaining.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__) or '.')
import numpy as np
from relspec.pipeline import CENTERS, ENV_CENTERS, to_amp, NBINS, ENV_BINS
from relspec.codec2 import Codec2, Decoder2
from relspec.pipeline2 import extract2
from relspec.patterns import extract_patterns
from relspec import datasets as DS
from eval_real import line, pattern_loss

SUBSET = {'normal': (0, 2), 'IR007': (0,), 'OR007_6': (1,), 'B014': (2,),
          'IR021': (3,), 'OR021_6': (0,), 'B007': (1,)}

def pick_records():
    recs = []
    for r in DS.cwru_records():
        want = SUBSET.get(r['state'])
        if want is None: continue
        load = int(r['group'].rsplit('L', 1)[1])
        if load in want: recs.append(r)
    return recs

def run_config(recs, win_s, k_mad, floor_db, do_patterns=True):
    frames = 0; total_b = 0
    line_errs = []; loss_rows = []
    for rec in recs:
        fs, x = rec['fs'], rec['x']
        n = int(win_s*fs)
        nw = min(8, len(x)//n)
        ca = Codec2(nbins=NBINS, n_bands=16, refs='best',
                    k_mad=k_mad, mad_floor_db=floor_db)
        ce = Codec2(nbins=ENV_BINS, n_peaks=32, n_bands=12, refs='best',
                    k_mad=k_mad, mad_floor_db=floor_db)
        da = Decoder2(nbins=NBINS, n_bands=16)
        de = Decoder2(nbins=ENV_BINS, n_peaks=32, n_bands=12)
        mad_a = np.full(NBINS, 4.0); mad_e = np.full(ENV_BINS, 4.0)
        for w in range(nw):
            seg = x[w*n:(w+1)*n]
            e = extract2(seg, fs, band_key=rec['group'],
                         fr_nominal=rec['fr_nominal'])
            pa, _ = ca.encode(e.acc_u8, mad_a)
            pe, _ = ce.encode(e.env_u8, mad_e)
            ra, _ = da.decode(pa); re_, _ = de.decode(pe)
            mad_a = np.maximum(0.95*mad_a+0.05*np.abs(e.acc_u8.astype(float)-ra.astype(float)), 1.0)
            mad_e = np.maximum(0.95*mad_e+0.05*np.abs(e.env_u8.astype(float)-re_.astype(float)), 1.0)
            frames += 1; total_b += len(pa)+len(pe)
            ao, eo = to_amp(e.acc_u8), to_amp(e.env_u8)
            ar, er = to_amp(ra), to_amp(re_)
            for nm, o in {**rec['lines'], '1x': 1.0, '2x': 2.0, '3x': 3.0}.items():
                a1 = line(ao, CENTERS, o)
                if a1 > 0:
                    line_errs.append(abs(20*np.log10(max(line(ar, CENTERS, o), 1e-12)/a1)))
            if do_patterns:
                loss_rows += pattern_loss(ao, ar, CENTERS)
                loss_rows += pattern_loss(eo, er, ENV_CENTERS,
                                          min_harmonics=3, f_hi=6.5)
    res = dict(bytes_frame=total_b/max(frames, 1), frames=frames,
               line_p95=float(np.percentile(line_errs, 95)) if line_errs else None,
               line_med=float(np.median(line_errs)) if line_errs else None)
    if loss_rows:
        share = np.array([r['share'] for r in loss_rows])
        rec_ = np.array([r['recovered'] for r in loss_rows])
        res['energy_rec_pct'] = float(100*share[rec_].sum()/max(share.sum(), 1e-12))
    return res

if __name__ == '__main__':
    recs = pick_records()
    print(f'{len(recs)} files in subset')
    out = dict(knobs=[], length=[])
    for k_mad in (1.5, 2.5, 4.0, 6.0):
        for floor_db in (2.0, 3.0, 5.0):
            t0 = time.time()
            r = run_config(recs, 2.0, k_mad, floor_db)
            r.update(k_mad=k_mad, floor_db=floor_db)
            out['knobs'].append(r)
            print(f'k={k_mad:<4} floor={floor_db:<4} {r["bytes_frame"]:7.1f} B/f  '
                  f'line p95 {r["line_p95"]:.2f} dB  '
                  f'energy rec {r.get("energy_rec_pct", float("nan")):.2f}%  '
                  f'({time.time()-t0:.0f}s)', flush=True)
    for win in (1.0, 2.0, 4.0):
        r = run_config(recs, win, 2.5, 3.0, do_patterns=False)
        r.update(win_s=win)
        out['length'].append(r)
        print(f'win={win}s  {r["bytes_frame"]:7.1f} B/frame  '
              f'line p95 {r["line_p95"]:.2f} dB', flush=True)
    json.dump(out, open('../eval_rd.json', 'w'), default=float)
    print('wrote ../eval_rd.json')
