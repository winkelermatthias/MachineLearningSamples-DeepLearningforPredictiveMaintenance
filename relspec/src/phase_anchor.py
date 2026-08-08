"""Anchor-mode normalisation with phase.

The premise under test: we do not need shaft speed. We need a reference mode
that is dominant, consistently present, and holds a FIXED RATIO to shaft speed.
Normalise every spectrum by that mode and patterns land on the same axis
position every time, which is all pattern tracking actually requires. True
orders are recoverable later by one scalar per asset, or never, if only relative
tracking matters.

Two things decide whether this works.

1. ORDER-LOCKED vs FREQUENCY-LOCKED. The anchor must move with the shaft. A
   structural resonance or a line-frequency component sits at fixed Hz no matter
   what the machine does; normalising by it would ACTIVELY MISALIGN patterns,
   turning a stable comb into one that drifts. Distinguishing the two needs
   observations at different speeds, and it is the single most important test
   here because getting it wrong is worse than not doing it at all.

2. PRECISION OF THE ANCHOR FREQUENCY. Alignment is only as good as the estimate
   of f_a. Bin-limited estimation on a 0.039 order/bin axis caps alignment at
   about 4% at order 1. Phase does far better for free.

PHASE: the two-frame estimator. Take two FFT frames offset by hop H. For a
component in bin k the phase must advance by 2*pi*k*H/N if it sits exactly at
the bin centre; the wrapped excess gives its true sub-bin position.

    dphi = wrap( angle(X2[k]) - angle(X1[k]) - 2*pi*k*H/N )
    f    = (k + dphi*N/(2*pi*H)) * fs/N

This is the phase-vocoder estimator. It costs one extra FFT of a frame we
already have, needs no extra bins, and resolves far below bin width. Phase is
never transmitted: it is consumed on the device to produce one precise number.
"""
from __future__ import annotations
import numpy as np, scipy.io as sio, os, json, time
import sys; sys.path.insert(0, os.path.dirname(__file__) or '.')
from relspec.pipeline import estimate_speed, envelope

DATA = os.environ.get('CWRU', '/home/claude/data')
FS = 12000.0
NOMINAL = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1730.0}
FILES = [('97','normal',0),('98','normal',1),('99','normal',2),('100','normal',3),
         ('105','IR_007',0),('106','IR_007',1),('107','IR_007',2),('108','IR_007',3),
         ('169','IR_014',0),('170','IR_014',1),('171','IR_014',2),('172','IR_014',3),
         ('209','IR_021',0),('210','IR_021',1),('211','IR_021',2),('212','IR_021',3)]

def load(fid):
    m = sio.loadmat(f'{DATA}/{fid}.mat')
    de = [k for k in m if k.endswith('_DE_time')][0]
    rpm = [k for k in m if k.endswith('RPM')]
    return (m[de].ravel().astype(np.float64),
            float(m[rpm[0]].ravel()[0]) if rpm else np.nan)

# ---------------------------------------------------------------- phase core
def phase_refine(seg, N=4096, hop=None, fs=FS, fmin=20.0, fmax=5500.0, topk=40):
    """Phase-vocoder frequency refinement. Returns (freqs, amps) for the topk
    strongest bins, each frequency corrected to sub-bin precision."""
    hop = hop or N//4
    if len(seg) < N+hop: return np.array([]), np.array([])
    w = np.hanning(N)
    X1 = np.fft.rfft(seg[:N]*w)
    X2 = np.fft.rfft(seg[hop:hop+N]*w)
    mag = np.abs(X1)
    k = np.arange(len(mag))
    expected = 2*np.pi*k*hop/N
    d = np.angle(X2) - np.angle(X1) - expected
    d = (d + np.pi) % (2*np.pi) - np.pi          # wrap to +/- pi
    f = (k + d*N/(2*np.pi*hop)) * fs/N
    band = (f > fmin) & (f < fmax)
    idx = np.where(band)[0]
    if idx.size == 0: return np.array([]), np.array([])
    sel = idx[np.argsort(-mag[idx])[:topk]]
    return f[sel], mag[sel]

def bin_freq(seg, N=4096, fs=FS, fmin=20.0, fmax=5500.0, topk=40):
    """Bin-limited baseline: same peaks, frequency taken as the bin centre."""
    w = np.hanning(N)
    mag = np.abs(np.fft.rfft(seg[:N]*w))
    k = np.arange(len(mag)); f = k*fs/N
    band = (f > fmin) & (f < fmax)
    idx = np.where(band)[0]
    sel = idx[np.argsort(-mag[idx])[:topk]]
    return f[sel], mag[sel]

def dominant_mode(freqs, amps, n_cluster=1):
    """Strongest spectral component, with its amplitude."""
    if freqs.size == 0: return np.nan, 0.0
    j = int(np.argmax(amps))
    return float(freqs[j]), float(amps[j])

# ------------------------------------------- order-locked vs frequency-locked
def lock_test(obs):
    """obs: list of (fr_true, f_mode). A mode that tracks the shaft has constant
    f_mode/fr_true; a structural mode has constant f_mode. Compare the relative
    spread of each hypothesis: whichever is tighter is what the mode is."""
    fr = np.array([o[0] for o in obs]); fm = np.array([o[1] for o in obs])
    ok = np.isfinite(fr) & np.isfinite(fm) & (fr > 0) & (fm > 0)
    fr, fm = fr[ok], fm[ok]
    if fr.size < 4: return None
    ratio = fm/fr
    cv_order = float(np.std(ratio)/np.mean(ratio))     # order-locked hypothesis
    cv_hz = float(np.std(fm)/np.mean(fm))              # frequency-locked
    return dict(n=int(fr.size), mean_order=float(np.mean(ratio)),
                cv_order=cv_order, cv_hz=cv_hz,
                verdict=('order-locked' if cv_order < cv_hz else 'frequency-locked'),
                margin=float(max(cv_hz, 1e-9)/max(cv_order, 1e-9)))

if __name__ == '__main__':
    WIN = 2.0; n = int(WIN*FS)
    recs = []
    t0 = time.time()
    for fid, state, load_hp in FILES:
        x, rpm = load(fid)
        fr_true = (rpm if np.isfinite(rpm) else NOMINAL[load_hp])/60.0
        for w in range(min(6, max(1, len(x)//n))):
            s = w*n
            if s+n > len(x): break
            seg = x[s:s+n]
            fp, ap = phase_refine(seg)
            fb, ab = bin_freq(seg)
            f_ph, a_ph = dominant_mode(fp, ap)
            f_bin, _ = dominant_mode(fb, ab)
            fr_env, conf = estimate_speed(seg)
            recs.append(dict(fid=fid, state=state, load=load_hp, w=w,
                             fr_true=fr_true, fr_env=fr_env, conf=conf,
                             f_phase=f_ph, f_bin=f_bin,
                             fp=fp.tolist(), ap=ap.tolist()))
    print(f'{len(recs)} acquisitions in {time.time()-t0:.0f}s\n')

    # ---- 1. is the dominant mode order-locked? ----
    print('='*88)
    print('1. IS THE DOMINANT SPECTRAL MODE ORDER-LOCKED OR FREQUENCY-LOCKED?')
    print('='*88)
    print('   CWRU load steps move the shaft 1797 -> 1730 rpm, a 3.9% span, which')
    print('   is what makes the two hypotheses separable at all.\n')
    for state in ('normal','IR_007','IR_014','IR_021'):
        sub = [r for r in recs if r['state'] == state]
        for label, key in (('phase-refined','f_phase'), ('bin-limited','f_bin')):
            t = lock_test([(r['fr_true'], r[key]) for r in sub])
            if t:
                print(f"  {state:<8} {label:<14} mean order {t['mean_order']:7.3f}  "
                      f"cv_order {t['cv_order']:.4f}  cv_hz {t['cv_hz']:.4f}  "
                      f"-> {t['verdict']} ({t['margin']:.2f}x)")
        print()

    # ---- 2. anchor candidates: which mode is most consistently order-locked ----
    print('='*88)
    print('2. ANCHOR SELECTION: rank candidate modes by order stability')
    print('='*88)
    print('   For each candidate order band, how tightly does the strongest line')
    print('   inside it hold a constant ratio to shaft speed across all 4 loads?\n')
    bands = [(0.8,1.2,'1x'),(1.8,2.2,'2x'),(2.8,3.2,'3x'),(3.4,3.8,'BPFO'),
             (5.2,5.7,'BPFI'),(9.5,13.5,'~12x family'),(30,60,'30-60x'),
             (60,120,'60-120x'),(120,200,'120-200x')]
    print(f"{'band':<14}{'n':>5}{'mean order':>12}{'cv_order':>11}{'cv_hz':>10}{'verdict':>18}{'margin':>8}")
    anchors = []
    for lo, hi, name in bands:
        obs = []
        for r in recs:
            fp = np.array(r['fp']); ap = np.array(r['ap'])
            if fp.size == 0: continue
            o = fp/r['fr_true']
            m = (o >= lo) & (o <= hi)
            if not m.any(): continue
            j = np.argmax(ap[m])
            obs.append((r['fr_true'], float(fp[m][j])))
        t = lock_test(obs)
        if not t: continue
        anchors.append((name, t))
        print(f"{name:<14}{t['n']:>5}{t['mean_order']:>12.3f}{t['cv_order']:>11.4f}"
              f"{t['cv_hz']:>10.4f}{t['verdict']:>18}{t['margin']:>8.2f}")

    # ---- 3. alignment stability under three normalisations ----
    print('\n'+'='*88)
    print('3. ALIGNMENT: how tightly do patterns land on the same axis position?')
    print('='*88)
    print('   Take the strongest lines of each acquisition, normalise by each')
    print('   candidate reference, and measure the spread of the resulting axis')
    print('   positions within a health state. Tighter is better: it is exactly')
    print('   the smearing that destroys a pattern time series.\n')
    def spread(state, mode):
        pos = []
        for r in recs:
            if r['state'] != state: continue
            fp = np.array(r['fp']); ap = np.array(r['ap'])
            if fp.size == 0: continue
            if mode == 'true':   ref = r['fr_true']
            elif mode == 'env':  ref = r['fr_env']
            elif mode == 'phase':ref = r['f_phase']
            elif mode == 'bin':  ref = r['f_bin']
            if not np.isfinite(ref) or ref <= 0: continue
            o = fp/ref
            sel = np.argsort(-ap)[:8]
            pos.append(np.sort(o[sel]))
        if len(pos) < 3: return np.nan
        L = min(len(p) for p in pos)
        P = np.stack([p[:L] for p in pos])
        # relative spread of each ranked position, averaged
        return float(np.mean(np.std(P, 0)/np.maximum(np.mean(P, 0), 1e-9)))
    print(f"{'state':<10}{'true speed':>13}{'envelope est':>14}{'phase anchor':>14}{'bin anchor':>13}")
    for state in ('normal','IR_007','IR_014','IR_021'):
        vals = [spread(state, m) for m in ('true','env','phase','bin')]
        print(f"{state:<10}" + ''.join(f'{v:>13.4f} ' if np.isfinite(v) else f"{'—':>13} "
                                       for v in vals))

    # ---- 4. what phase costs and buys ----
    print('\n'+'='*88)
    print('4. PRECISION: phase-refined against bin-limited frequency')
    print('='*88)
    df = FS/4096
    err = []
    for r in recs:
        if np.isfinite(r['f_phase']) and np.isfinite(r['f_bin']):
            err.append(abs(r['f_phase']-r['f_bin']))
    err = np.array(err)
    print(f'  FFT bin width           : {df:.4f} Hz')
    print(f'  phase vs bin difference : median {np.median(err):.4f} Hz, '
          f'p95 {np.percentile(err,95):.4f} Hz')
    print(f'  as a fraction of a bin  : median {np.median(err)/df:.3f}, '
          f'p95 {np.percentile(err,95)/df:.3f}')
    print(f'\n  payload cost: f_anchor as 24-bit fixed point = 3 B/frame')
    print(f'                anchor quality byte             = 1 B/frame')
    print(f'                per-peak sub-bin offset, 4 bits = 20 B for 40 peaks')
    print(f'                total 24 B on a ~611 B frame    = 3.9% overhead')
    json.dump(dict(recs=[{k: v for k, v in r.items() if k not in ('fp','ap')}
                         for r in recs],
                   anchors=[(n, t) for n, t in anchors]),
              open('../phase_eval.json','w'), default=float)
    print('\nwrote phase_eval.json')

# ============================================================================
def followup():
    """The selected anchor, not the globally dominant one.

    Test 1 showed the strongest line in the spectrum is a different mode in each
    health state, so 'take the loudest' is not an anchor. Test 2 found the mode
    that IS consistent. This re-runs alignment using it."""
    import numpy as np
    WIN = 2.0; n = int(WIN*FS); recs = []
    for fid, state, load_hp in FILES:
        x, rpm = load(fid)
        fr_true = (rpm if np.isfinite(rpm) else NOMINAL[load_hp])/60.0
        for w in range(min(6, max(1, len(x)//n))):
            s = w*n
            if s+n > len(x): break
            seg = x[s:s+n]
            fp, ap = phase_refine(seg); fb, ab = bin_freq(seg)
            fr_env, _ = estimate_speed(seg)
            recs.append(dict(state=state, fr_true=fr_true, fr_env=fr_env,
                             fp=fp, ap=ap, fb=fb, ab=ab))

    LO, HI = 9.5, 13.5           # the ~12x family, cv_order 0.0054
    def anchor_of(r, which):
        f = r['fp'] if which == 'phase' else r['fb']
        a = r['ap'] if which == 'phase' else r['ab']
        if f.size == 0: return np.nan
        # bracket by a coarse speed proxy, then take the strongest line in band
        o = f/max(r['fr_env'], 1e-9)
        m = (o >= LO) & (o <= HI)
        if not m.any(): return np.nan
        return float(f[m][np.argmax(a[m])])

    print('\n'+'='*88)
    print('5. ALIGNMENT USING THE SELECTED ANCHOR (the ~12x order-locked family)')
    print('='*88)
    def spread(state, mode):
        pos = []
        for r in recs:
            if r['state'] != state: continue
            f, a = r['fp'], r['ap']
            if f.size == 0: continue
            ref = {'true': r['fr_true'], 'env': r['fr_env'],
                   'anchor_phase': anchor_of(r, 'phase'),
                   'anchor_bin': anchor_of(r, 'bin')}[mode]
            if not np.isfinite(ref) or ref <= 0: continue
            o = f/ref
            sel = np.argsort(-a)[:8]
            pos.append(np.sort(o[sel]))
        if len(pos) < 3: return np.nan
        L = min(len(p) for p in pos)
        P = np.stack([p[:L] for p in pos])
        return float(np.mean(np.std(P, 0)/np.maximum(np.mean(P, 0), 1e-9)))
    print(f"{'state':<10}{'true speed':>13}{'envelope':>12}{'anchor+phase':>15}{'anchor+bin':>13}")
    agg = {m: [] for m in ('true','env','anchor_phase','anchor_bin')}
    for state in ('normal','IR_007','IR_014','IR_021'):
        row = {m: spread(state, m) for m in agg}
        for m in agg:
            if np.isfinite(row[m]): agg[m].append(row[m])
        print(f"{state:<10}" + ''.join(f"{row[m]:>13.4f} " for m in
              ('true','env','anchor_phase','anchor_bin')))
    print(f"{'MEAN':<10}" + ''.join(f"{np.mean(agg[m]):>13.4f} " for m in
          ('true','env','anchor_phase','anchor_bin')))

    print('\n'+'='*88)
    print('6. DOES PHASE TIGHTEN THE ANCHOR ITSELF?')
    print('='*88)
    for which in ('phase','bin'):
        obs = [(r['fr_true'], anchor_of(r, which)) for r in recs]
        obs = [(a,b) for a,b in obs if np.isfinite(b)]
        t = lock_test(obs)
        print(f"  anchor via {which:<6}: n={t['n']:>3}  ratio to shaft {t['mean_order']:.4f}  "
              f"cv_order {t['cv_order']:.5f}  cv_hz {t['cv_hz']:.5f}  {t['verdict']}")
    ph = [anchor_of(r,'phase') for r in recs]; bn = [anchor_of(r,'bin') for r in recs]
    d = np.array([abs(a-b) for a,b in zip(ph,bn) if np.isfinite(a) and np.isfinite(b)])
    print(f"  phase correction applied : median {np.median(d):.4f} Hz "
          f"({np.median(d)/(FS/4096):.3f} of a bin), p95 {np.percentile(d,95):.4f} Hz")
    return recs

if os.environ.get('FOLLOWUP'): followup()
