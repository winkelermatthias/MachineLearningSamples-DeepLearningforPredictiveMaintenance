"""Where the bytes go: per-layer ledger for v2 residual frames on CWRU.

The layers are gains (global + band), tracked-peak deltas, and the active-bin
residual list. The split determines where the next factor comes from: if
actives dominate, better prediction pays; if peaks dominate, the peak matcher
is mis-tracking; if gains dominate, something upstream (load normalisation)
is being paid for repeatedly.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__) or '.')
import numpy as np
from relspec.pipeline import NBINS, ENV_BINS
from relspec.codec2 import Codec2, Decoder2
from relspec.pipeline2 import extract2
from relspec import datasets as DS
from eval_rd import pick_records

if __name__ == '__main__':
    recs = pick_records()
    agg = {'acc': [], 'env': []}
    anchors = {'acc': [], 'env': []}
    for rec in recs:
        fs, x = rec['fs'], rec['x']
        n = int(2.0*fs)
        nw = min(8, len(x)//n)
        ca = Codec2(nbins=NBINS, n_bands=16, refs='best')
        ce = Codec2(nbins=ENV_BINS, n_peaks=32, n_bands=12, refs='best')
        da = Decoder2(nbins=NBINS, n_bands=16)
        de = Decoder2(nbins=ENV_BINS, n_peaks=32, n_bands=12)
        mad_a = np.full(NBINS, 4.0); mad_e = np.full(ENV_BINS, 4.0)
        for w in range(nw):
            e = extract2(x[w*n:(w+1)*n], fs, band_key=rec['group'],
                         fr_nominal=rec['fr_nominal'])
            for tag, enc, dec, u8, mad in (('acc', ca, da, e.acc_u8, mad_a),
                                           ('env', ce, de, e.env_u8, mad_e)):
                pl, kind = enc.encode(u8, mad)
                got, _ = dec.decode(pl)
                mad[:] = np.maximum(0.95*mad+0.05*np.abs(u8.astype(float)-got.astype(float)), 1.0)
                if kind == 'anchor': anchors[tag].append(len(pl))
                elif enc.last_breakdown:
                    d = dict(enc.last_breakdown); d['total'] = len(pl)
                    agg[tag].append(d)
    out = {}
    for tag in ('acc', 'env'):
        A = agg[tag]
        if not A: continue
        gains = np.mean([a['gains_bits'] for a in A])/8
        peaks = np.mean([a['peak_bits'] for a in A])/8
        act = np.mean([a['act_bits'] for a in A])/8
        tot = np.mean([a['total'] for a in A])
        print(f'{tag}: residual frames {len(A)}, mean {tot:.0f} B  '
              f'gains {gains:.0f} B ({100*gains/tot:.0f}%)  '
              f'peaks {peaks:.0f} B ({100*peaks/tot:.0f}%)  '
              f'actives {act:.0f} B ({100*act/tot:.0f}%)  '
              f'[{np.mean([a["n_peaks"] for a in A]):.0f} pk, '
              f'{np.mean([a["n_act"] for a in A]):.0f} act]')
        print(f'{tag}: anchors {len(anchors[tag])}, mean '
              f'{np.mean(anchors[tag]):.0f} B (v1 anchor: '
              f'{"890" if tag == "acc" else "482"} B raw+table)')
        out[tag] = dict(res_frames=len(A), res_mean=float(tot),
                        gains_B=float(gains), peaks_B=float(peaks),
                        act_B=float(act),
                        anchor_mean=float(np.mean(anchors[tag])) if anchors[tag] else None)
    json.dump(out, open('../eval_breakdown.json', 'w'))
    print('wrote ../eval_breakdown.json')
