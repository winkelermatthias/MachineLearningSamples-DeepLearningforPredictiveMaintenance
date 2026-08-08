"""Full CWRU evaluation and shaft-speed recovery from the stored spectrum.

Two questions.

1. LOSS. Across the whole dataset, does the pattern layer see the same patterns
   carrying the same energy after the codec? Measured per pattern kind and on
   the specific spectral lines a reliability engineer would actually read.

2. THE MAIN MODE. The payload stores magnitude only, quantised to 0.5 dB. Phase
   is discarded entirely. Can shaft speed still be recovered?

   Yes, and the reason is structural rather than lucky: shaft speed is encoded
   in the SPACING of the harmonic comb, not in phase. Three phase-free
   estimators all read that spacing:

     cepstrum        C(q) = |IFFT{ log|X(f)| }|^2, peak at q = 1/f_shaft
     harmonic prod   P(f) = prod_k |X(kf)|^(1/K)
     comb score      geometric mean of harmonic amplitudes over the local floor

   Phase would be needed to track speed WITHIN an acquisition (that is the
   coherent order-tracking problem, T_coh = 1/(4 k e f)). It is not needed to
   estimate the mean speed OF a stored spectrum.

   There is a sharper consequence. The order axis is locked to the speed
   estimated at encode time, so by construction the dominant comb must sit at
   order 1.000. Its measured deviation from 1.000 is therefore a direct,
   self-contained readout of the speed estimator's error, recoverable from the
   compressed payload alone with no tacho and no reference.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__) or '.')
import numpy as np, scipy.io as sio
from relspec.pipeline import (extract, Codec, Decoder, CENTERS, ENV_CENTERS,
                              to_amp, Q_DB, NBINS, estimate_speed)
from relspec.patterns import extract_patterns, find_combs

DATA = os.environ.get('CWRU', '/home/claude/data')
FS = 12000.0; WIN = 2.0
NOMINAL = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1730.0}
FILES = [('97','normal',0),('98','normal',1),('99','normal',2),('100','normal',3),
         ('105','IR_007',0),('106','IR_007',1),('107','IR_007',2),('108','IR_007',3),
         ('169','IR_014',0),('170','IR_014',1),('171','IR_014',2),('172','IR_014',3),
         ('209','IR_021',0),('210','IR_021',1),('211','IR_021',2),('212','IR_021',3)]
BEAR = dict(BPFO=3.5848, BPFI=5.4152, BSF=4.7135, FTF=0.3983)

def load(fid):
    m = sio.loadmat(f'{DATA}/{fid}.mat')
    de = [k for k in m if k.endswith('_DE_time')][0]
    rpm = [k for k in m if k.endswith('RPM')]
    return (m[de].ravel().astype(np.float64),
            float(m[rpm[0]].ravel()[0]) if rpm else np.nan)

# ------------------------------------------------- phase-free speed readout
def cepstrum_order(spec_u8, centers, nfine=512, qlo=0.35, qhi=3.0):
    """Real cepstrum of the stored log-magnitude spectrum, over the uniformly
    spaced fine region only. The uint8 payload IS log magnitude in 0.5 dB steps,
    so no log is needed: the codec already stores the cepstrum's input.
    Returns the dominant harmonic spacing in orders."""
    v = spec_u8[:nfine].astype(np.float64)*Q_DB
    v = v - v.mean()
    v *= np.hanning(len(v))
    c = np.abs(np.fft.rfft(v))**2
    du = float(centers[1]-centers[0])            # orders per bin, fine region
    # quefrency axis: lag index j <-> spacing = 1/(j/(nfine*du)) orders
    j = np.arange(len(c))
    with np.errstate(divide='ignore'):
        spacing = np.where(j > 0, nfine*du/np.maximum(j, 1), np.inf)
    m = (spacing >= qlo) & (spacing <= qhi)
    if not m.any(): return np.nan, 0.0
    idx = np.where(m)[0]
    # The cepstrum of a comb of spacing S peaks at quefrency 1/S and at every
    # multiple of it, which in spacing units is S, S/2, S/3. Taking the global
    # maximum therefore lands on a rahmonic about half the time. The
    # fundamental is the LARGEST spacing (smallest lag) whose peak is within a
    # factor of the strongest.
    kmax = idx[int(np.argmax(c[idx]))]
    thr = 0.45*c[kmax]
    strong = [j for j in idx if c[j] >= thr and 0 < j < len(c)-1
              and c[j] >= c[j-1] and c[j] >= c[j+1]]
    k = min(strong) if strong else kmax
    # parabolic refinement in lag, then convert
    if 0 < k < len(c)-1:
        y0, y1, y2 = c[k-1], c[k], c[k+1]
        den = 2*(y0-2*y1+y2)
        d = (y0-y2)/den if abs(den) > 1e-20 else 0.0
        kk = k + (d if -1 < d < 1 else 0.0)
    else:
        kk = k
    sp = nfine*du/max(kk, 1e-9)
    rival = np.delete(c[idx], np.argmax(c[idx]))
    snr = float(c[k]/max(np.median(rival), 1e-30))
    return float(sp), snr

def line(spec, centers, order, tol=0.10):
    m = np.abs(centers-order) < max(tol, 0.012*order)
    return float(spec[m].max()) if m.any() else 0.0

def main():
    ca, cd = Codec(), Decoder()
    ce = Codec(nbins=len(ENV_CENTERS), n_peaks=32, n_bands=12)
    de = Decoder(nbins=len(ENV_CENTERS), n_bands=12)
    mad_a = np.full(NBINS, 4.0); mad_e = np.full(len(ENV_CENTERS), 4.0)
    rows, loss, speed, lines_rows = [], [], [], []
    raw_b = coded_b = 0
    rng = np.random.default_rng(7)
    t_all = time.time()

    for fid, state, load_hp in FILES:
        x, rpm = load(fid)
        fr_true = (rpm if np.isfinite(rpm) else NOMINAL[load_hp])/60.0
        n = int(WIN*FS)
        nwin = min(8, max(1, len(x)//n))
        for w in range(nwin):
            s = w*n
            if s+n > len(x): break
            seg = x[s:s+n]
            e = extract(seg)
            pa, ka = ca.encode(e.acc_u8, mad_a)
            pe, ke = ce.encode(e.env_u8, mad_e)
            ra, _ = cd.decode(pa); re_, _ = de.decode(pe)
            mad_a = np.maximum(0.95*mad_a+0.05*np.abs(e.acc_u8.astype(float)-ra.astype(float)), 1.0)
            mad_e = np.maximum(0.95*mad_e+0.05*np.abs(e.env_u8.astype(float)-re_.astype(float)), 1.0)
            raw_b += 2*(4096//2+1)*4; coded_b += len(pa)+len(pe)
            ao, ar = to_amp(e.acc_u8), to_amp(ra)
            eo, er = to_amp(e.env_u8), to_amp(re_)

            # ---- speed: three phase-free readouts on the STORED spectrum ----
            cep_o, cep_snr = cepstrum_order(e.acc_u8, CENTERS)
            cep_r, _ = cepstrum_order(ra, CENTERS)
            cb_o = find_combs(ao, CENTERS)
            cb_r = find_combs(ar, CENTERS)
            f0_o = cb_o[0]['f0'] if cb_o else np.nan
            f0_r = cb_r[0]['f0'] if cb_r else np.nan
            speed.append(dict(fid=fid, state=state, load=load_hp, w=w,
                              fr_true=fr_true, fr_est=e.fr, conf=e.conf,
                              est_err_pct=100*(e.fr-fr_true)/fr_true,
                              cep_o=cep_o, cep_r=cep_r, cep_snr=cep_snr,
                              f0_o=f0_o, f0_r=f0_r))

            # ---- pattern loss ----
            _, _, bo = extract_patterns(ao, CENTERS)
            _, _, br = extract_patterns(ar, CENTERS)
            _, _, beo = extract_patterns(eo, ENV_CENTERS, min_harmonics=3, f_hi=6.5)
            _, _, ber = extract_patterns(er, ENV_CENTERS, min_harmonics=3, f_hi=6.5)
            for tag, O, R in (('acc', bo, br), ('env', beo, ber)):
                for p in O['patterns']:
                    if p['energy'] <= 0: continue
                    m = None
                    for qq in R['patterns']:
                        if qq['kind'] != p['kind']: continue
                        if abs(qq['key']-p['key']) < 0.05*max(abs(p['key']), 1.0):
                            m = qq; break
                    loss.append(dict(rail=tag, kind=p['kind'], state=state,
                        share=p['share'], recovered=m is not None,
                        err_db=(10*np.log10(max(m['energy'],1e-30)/max(p['energy'],1e-30))
                                if m else None),
                        key_err_pct=(100*abs(m['key']-p['key'])/max(abs(p['key']),1e-9)
                                     if m else None)))
            rows.append(dict(fid=fid, state=state, load=load_hp,
                             resid_acc=bo['residual_share'], resid_env=beo['residual_share'],
                             bytes=len(pa)+len(pe), kind=ka+'/'+ke,
                             med_err=float(np.median(np.abs(ra.astype(int)-e.acc_u8.astype(int))*Q_DB))))
            # ---- named lines a reliability engineer would read ----
            for nm, o in list(BEAR.items())+[('1x',1.0),('2x',2.0),('3x',3.0)]:
                a1, a2 = line(ao, CENTERS, o), line(ar, CENTERS, o)
                if a1 <= 0: continue
                lines_rows.append(dict(state=state, name=nm,
                    err_db=float(20*np.log10(max(a2,1e-12)/a1))))
    out = dict(rows=rows, loss=loss, speed=speed, lines=lines_rows,
               raw_b=raw_b, coded_b=coded_b, secs=time.time()-t_all)
    return out

if __name__ == '__main__':
    import pandas as pd
    d = main()
    R = pd.DataFrame(d['rows']); L = pd.DataFrame(d['loss'])
    S = pd.DataFrame(d['speed']); LN = pd.DataFrame(d['lines'])
    print(f"acquisitions {len(R)}  ratio {d['raw_b']/d['coded_b']:.1f}x  "
          f"mean frame {R.bytes.mean():.0f} B  {d['secs']:.0f}s")
    print('\n== LOSS by rail and kind ==')
    g = L.groupby(['rail','kind']).agg(n=('recovered','size'),
        rec=('recovered','sum'),
        med_err=('err_db', lambda s: np.nanmedian(np.abs(s.dropna())) if s.notna().any() else np.nan),
        p95=('err_db', lambda s: np.nanpercentile(np.abs(s.dropna()),95) if s.notna().any() else np.nan),
        key_err=('key_err_pct', lambda s: np.nanmedian(s.dropna()) if s.notna().any() else np.nan),
        share=('share','mean'))
    g['rate'] = (100*g.rec/g.n).round(1)
    print(g.round(3).to_string())
    print('\n== energy-weighted recovery (does the loss land on patterns that matter?) ==')
    L['w'] = L.share
    for rail in ('acc','env'):
        sub = L[L.rail==rail]
        wr = (sub[sub.recovered].w.sum()/sub.w.sum())*100
        print(f'  {rail}: {wr:.2f}% of pattern energy recovered, '
              f'{100*sub.recovered.mean():.1f}% of pattern count')
    print('\n== named line amplitude error (dB) ==')
    print(LN.groupby('name').err_db.agg(n='size',
        med=lambda s: np.median(np.abs(s)), p95=lambda s: np.percentile(np.abs(s),95),
        mx=lambda s: np.abs(s).max()).round(3).to_string())
    print('\n== residual share ==')
    print(f"  acc mean {R.resid_acc.mean()*100:.1f}%  median {R.resid_acc.median()*100:.1f}%")
    print(f"  env mean {R.resid_env.mean()*100:.1f}%  median {R.resid_env.median()*100:.1f}%")
    print('\n== SHAFT SPEED: phase-free recovery ==')
    S['cep_err'] = 100*(S.cep_o-1.0)
    S['cep_err_r'] = 100*(S.cep_r-1.0)
    S['f0_err_o'] = 100*(S.f0_o-1.0)
    S['f0_err_r'] = 100*(S.f0_r-1.0)
    print(f"  envelope HPS on waveform : median |err| {np.median(np.abs(S.est_err_pct)):.3f}%  "
          f"p95 {np.percentile(np.abs(S.est_err_pct),95):.3f}%")
    print(f"  cepstrum on stored spec  : median |dev from 1.000| {np.median(np.abs(S.cep_err)):.3f}%  "
          f"p95 {np.percentile(np.abs(S.cep_err),95):.3f}%")
    print(f"  cepstrum after codec     : median {np.median(np.abs(S.cep_err_r)):.3f}%  "
          f"p95 {np.percentile(np.abs(S.cep_err_r),95):.3f}%")
    print(f"  comb f0 on stored spec   : median {np.nanmedian(np.abs(S.f0_err_o)):.3f}%  "
          f"p95 {np.nanpercentile(np.abs(S.f0_err_o),95):.3f}%")
    print(f"  comb f0 after codec      : median {np.nanmedian(np.abs(S.f0_err_r)):.3f}%")
    agree = np.abs(S.cep_o-S.cep_r)
    print(f"  cepstrum orig vs decoded : median {np.median(agree)*100:.4f}% of an order, "
          f"identical in {100*np.mean(agree<1e-9):.0f}% of frames")
    json.dump(dict(rows=d['rows'], loss=d['loss'], speed=d['speed'], lines=d['lines'],
                   raw_b=d['raw_b'], coded_b=d['coded_b']),
              open('../cwru_eval.json','w'), default=float)
    print('\nwrote cwru_eval.json')
