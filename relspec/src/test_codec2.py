"""Fuzz the v2 codec against its decoder.

Three invariants, checked over randomised spectrum sequences that include the
nasty cases (level jumps, peak births, all-zero frames, saturated frames):

1. STATE LOCKSTEP. The encoder's model of the decoded frame must equal what
   Decoder2 actually produces, byte for byte, over long chains with mixed
   anchor / residual / chain-residual frames. Any divergence compounds, so one
   frame of drift anywhere in 200 is a hard fail.
2. ANCHOR LOSSLESSNESS. Decoded anchors equal the input exactly.
3. DEAD-ZONE BOUND. On residual frames, every bin's decode error is bounded by
   the suppression threshold max(k*MAD, floor) plus the peak-layer tolerance
   on the few bins the peak matcher owns.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__) or '.')
import numpy as np
from relspec.codec2 import Codec2, Decoder2

def walk_sequence(rng, nb, n_frames):
    """Random spectrum sequence: smooth base + persistent peaks + noise, with
    occasional regime jumps and a slow drift, in uint8 dB counts."""
    base = 120 + 40*np.sin(np.linspace(0, 3*np.pi, nb)) \
               + rng.normal(0, 6, nb).cumsum()*0.05
    peaks = rng.choice(nb, 25, replace=False)
    amps = rng.uniform(20, 80, 25)
    for t in range(n_frames):
        cur = base + rng.normal(0, 1.5, nb)
        if t and t % 37 == 0: base += rng.uniform(-25, 25)          # regime jump
        base += rng.normal(0, 0.15, nb)                             # slow drift
        cur[peaks] += amps * rng.uniform(0.9, 1.1, len(peaks))
        if t and t % 53 == 0:                                       # peak birth
            j = rng.integers(2, nb-2); cur[j] += 60
        if t % 71 == 70: cur[:] = 0                                 # dropout
        if t % 89 == 88: cur[:] = 255                               # saturation
        yield np.clip(cur, 0, 255).astype(np.uint8)

def run_case(seed, nb, nbands, refs, n_frames=200):
    rng = np.random.default_rng(seed)
    enc = Codec2(nbins=nb, n_bands=nbands, refs=refs)
    dec = Decoder2(nbins=nb, n_bands=nbands)
    mad = np.full(nb, 4.0)
    n_anchor = n_res = n_chain = 0
    for t, u8 in enumerate(walk_sequence(rng, nb, n_frames)):
        pl, kind = enc.encode(u8, mad)
        got, kind_d = dec.decode(pl)
        assert kind == kind_d, f'kind mismatch at t={t}: {kind} vs {kind_d}'
        if kind == 'anchor':
            n_anchor += 1
            assert np.array_equal(got, u8), f'anchor not lossless at t={t}'
        else:
            n_res += kind == 'residual'; n_chain += kind == 'residual_p'
            err = np.abs(got.astype(int)-u8.astype(int))
            thr = np.maximum(enc.k*mad, enc.floor)
            # peak layer may skip |damp|<3 on ~n_peaks bins; residual layer
            # bounds the rest at the suppression threshold (+0 quantiser error:
            # transmitted values are exact integers)
            bad = err > np.maximum(thr, 3.0)+1e-9
            assert not bad.any(), \
                f't={t}: {bad.sum()} bins exceed bound, max err {err.max()}'
        # lockstep: encoder state vs decoder state
        assert np.array_equal(np.asarray(enc.prev), np.asarray(dec.prev)), \
            f'encoder/decoder chain state diverged at t={t}'
        mad = np.maximum(0.95*mad+0.05*np.abs(u8.astype(float)-got.astype(float)), 1.0)
    return n_anchor, n_res, n_chain

if __name__ == '__main__':
    total = np.zeros(3, dtype=int)
    for seed in range(6):
        for nb, nbands in ((768, 16), (384, 12), (96, 4)):
            for refs in ('anchor', 'best'):
                total += run_case(seed, nb, nbands, refs)
    print(f'PASS  frames: {total[0]} anchor, {total[1]} residual, '
          f'{total[2]} chain-residual, zero divergence')
