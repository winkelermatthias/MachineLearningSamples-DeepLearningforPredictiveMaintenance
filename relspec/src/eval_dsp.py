"""Three DSP studies, each cashing out in the currency it claims.

A. BAND SELECTION (currency: envelope line SNR). Fixed 2-5 kHz vs kurtogram
   vs prewhitening+kurtogram, judged by the fault line's SNR in the envelope
   order spectrum on real fault recordings from all three rigs.

B. VITERBI SPEED TRACKING (currency: bytes). A 120-acquisition sequence from
   one noisy asset. Per-frame argmax teleports when the fundamental has a
   weak frame; the tracker cannot. Both speed series drive the same
   extraction and the same v2 codec; the difference is pure alignment cost.

C. ORDER-DOMAIN RESAMPLING (currency: both). Same machine, four speed
   conditions from steady to ramp. Welch-then-bin vs angular resampling,
   judged by line SNR at the shaft harmonics and by codec bytes across a
   40-frame sequence.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__) or '.')
import numpy as np
from scipy.signal import welch
from relspec.pipeline import EDGES, ENV_EDGES, bin_orders, to_u8, NBINS
from relspec.pipeline2 import extract2, _band_cache
from relspec.dsp2 import (kurtogram_band, envelope_banded, prewhiten,
                          hps_grid, viterbi_track, order_welch, line_snr_db)
from relspec.codec2 import Codec2
from relspec.synth2 import FaultState2, machine_catalog2, generate2
from relspec import datasets as DS

def env_order_spectrum(x, fs, band, fr):
    e = envelope_banded(x, fs, band)
    nps = DS.nperseg_for(fs)
    f, p = welch(e, fs=fs, nperseg=min(nps, len(e)),
                 noverlap=min(nps, len(e))//2, window='hann',
                 scaling='spectrum')
    return f/max(fr, 1e-9), np.sqrt(np.maximum(p, 0))

# =============================================================== A. bands
def study_bands():
    print('== A. demodulation band: fixed vs kurtogram vs prewhiten+kurtogram ==')
    rows = []
    cases = []
    for r in DS.cwru_records():
        st = r['state']
        if st.startswith('IR007'): key = 'BPFI'
        elif st.startswith('OR007_6'): key = 'BPFO'
        elif st.startswith('B007'): key = 'BSF'
        else: continue
        cases.append((r, key))
    for r in DS.mfpt_records():
        if r['state'] in ('IR', 'OR'):
            cases.append((r, 'BPFI' if r['state'] == 'IR' else 'BPFO'))
    for rec, key in cases:
        fs, fr = rec['fs'], rec['fr_nominal']
        n = int(2.0*fs)
        if len(rec['x']) < n: continue
        x = rec['x'][:n]
        target = rec['lines'][key]
        fixed = (2000., 5000.) if fs/2 > 5200 else (0.3*fs/2, 0.8*fs/2)
        kb, kurt = kurtogram_band(x, fs)
        xp = prewhiten(x)
        kbp, _ = kurtogram_band(xp, fs)
        for tag, sig, band in (('fixed', x, fixed), ('kurtogram', x, kb),
                               ('prewhiten+kurt', xp, kbp)):
            o, a = env_order_spectrum(sig, fs, band, fr)
            snr = line_snr_db(o, a, target)
            rows.append(dict(dataset=rec['dataset'], state=rec['state'],
                             file=rec['name'], method=tag, line=key,
                             snr_db=snr, band=list(band)))
    import pandas as pd
    R = pd.DataFrame(rows)
    piv = R.pivot_table(index=['dataset', 'state', 'file'], columns='method',
                        values='snr_db')
    g = piv.groupby(level='dataset').median().round(1)
    print(g.to_string())
    print('median gain over fixed: '
          f'kurtogram {(piv["kurtogram"]-piv["fixed"]).median():+.1f} dB, '
          f'prewhiten+kurt {(piv["prewhiten+kurt"]-piv["fixed"]).median():+.1f} dB')
    return rows

# ============================================================= B. viterbi
def study_viterbi(n_frames=120):
    print('\n== B. speed: per-frame argmax vs Viterbi, cashed out in bytes ==')
    cat = machine_catalog2()
    spec = cat['induction_motor']
    rng = np.random.default_rng(3)
    fr0 = 24.8
    # a hostile asset: strong mount line region, weak 1x, real wander
    truths, sigs = [], []
    fr = fr0
    for t in range(n_frames):
        fr = np.clip(fr*(1+rng.normal(0, 0.004)), fr0*0.94, fr0*1.06)
        # inner-race defect: load-zone passage modulates at 1x, so the
        # envelope carries a genuine shaft comb - the estimator has signal,
        # it just also has a rival (the fixed mount line) to teleport to
        f = FaultState2(inner_race=0.12, imbalance=0.2, lubrication=0.10)
        x = generate2(spec, f, 2.0, fr, load=0.5, speed_wander_pct=1.2,
                      rng=rng).astype(np.float64)
        x += 0.06*np.sin(2*np.pi*47.6*np.arange(len(x))/spec.fs)   # mount line
        truths.append(fr); sigs.append(x)
    grid = None; ems = []
    for x in sigs:
        g, h = hps_grid(x, spec.fs, lo=fr0*0.85, hi=fr0*1.18, step=0.01)
        grid = g; ems.append(h)
    argmax = np.array([grid[int(np.argmax(e))] for e in ems])
    vit = viterbi_track(ems, grid, max_jump_pct=1.5)
    truths = np.array(truths)
    e_a = 100*np.abs(argmax-truths)/truths
    e_v = 100*np.abs(vit-truths)/truths
    tele = int(np.sum(np.abs(np.diff(argmax))/argmax[:-1] > 0.02))
    print(f'  argmax : median {np.median(e_a):.3f}%  p95 {np.percentile(e_a,95):.3f}%  '
          f'max {e_a.max():.2f}%  jumps>2%: {tele}')
    print(f'  viterbi: median {np.median(e_v):.3f}%  p95 {np.percentile(e_v,95):.3f}%  '
          f'max {e_v.max():.2f}%  jumps>2%: {int(np.sum(np.abs(np.diff(vit))/vit[:-1] > 0.02))}')
    out = dict(err_argmax_med=float(np.median(e_a)), err_vit_med=float(np.median(e_v)),
               err_argmax_p95=float(np.percentile(e_a, 95)),
               err_vit_p95=float(np.percentile(e_v, 95)), teleports_argmax=tele)
    for tag, speeds in (('argmax', argmax), ('viterbi', vit)):
        enc = Codec2(refs='best')
        mad = np.full(NBINS, 4.0)
        tot = 0
        for x, fr_hat in zip(sigs, speeds):
            e = extract2(x, spec.fs, band_key='vit_study', fr_nominal=fr0,
                         fr_override=fr_hat)
            pl, kind = enc.encode(e.acc_u8, mad)
            got = enc.prev if 'residual' in kind else enc.anchor
            mad = np.maximum(0.95*mad+0.05*np.abs(e.acc_u8.astype(float) -
                                                  np.asarray(got, dtype=float)), 1.0)
            tot += len(pl)
        print(f'  codec v2best, acc rail, speed={tag:<7}: {tot/len(sigs):7.1f} B/frame')
        out[f'bytes_{tag}'] = tot/len(sigs)
    return out

# ======================================================== C. order tracking
def study_order(n_frames=40):
    print('\n== C. Welch-bin vs angular resampling, by speed condition ==')
    cat = machine_catalog2()
    spec = cat['gearbox']
    out = []
    for cond, wander, profile in (('steady', 0.1, 'steady'),
                                  ('wander1', 1.0, 'steady'),
                                  ('wander3', 3.0, 'steady'),
                                  ('ramp', 0.3, 'ramp')):
        rng = np.random.default_rng(17)
        fr0 = 16.2
        snr_w, snr_o = [], []
        byt = {'welch': 0, 'order': 0}
        enc = {k: Codec2(refs='best') for k in byt}
        mad = {k: np.full(NBINS, 4.0) for k in byt}
        for t in range(n_frames):
            fr = fr0*(1+rng.normal(0, 0.003))
            x = generate2(spec, FaultState2(gear_wear=0.3), 4.0, fr,
                          speed_wander_pct=wander, profile=profile,
                          rng=rng).astype(np.float64)
            nps = 8192
            f, p = welch(x, fs=spec.fs, nperseg=nps, noverlap=nps//2,
                         window='hann', scaling='spectrum')
            aw = np.sqrt(np.maximum(p, 0))*np.sqrt(2)
            ow = f/fr
            oo, ao = order_welch(x, spec.fs, fr)
            for k in (1, 2, 3, 23):        # shaft comb + gear mesh at 23x
                s1 = line_snr_db(ow, aw, float(k))
                s2 = line_snr_db(oo, ao, float(k))
                if np.isfinite(s1) and np.isfinite(s2):
                    snr_w.append(s1); snr_o.append(s2)
            uw = to_u8(bin_orders(f, aw, fr, EDGES))
            uo = to_u8(bin_orders(oo, ao, 1.0, EDGES))
            for k, u in (('welch', uw), ('order', uo)):
                pl, kind = enc[k].encode(u, mad[k])
                got = enc[k].prev if 'residual' in kind else enc[k].anchor
                mad[k] = np.maximum(0.95*mad[k] +
                                    0.05*np.abs(u.astype(float) -
                                                np.asarray(got, dtype=float)), 1.0)
                byt[k] += len(pl)
        r = dict(cond=cond,
                 snr_welch=float(np.median(snr_w)), snr_order=float(np.median(snr_o)),
                 bytes_welch=byt['welch']/n_frames, bytes_order=byt['order']/n_frames)
        out.append(r)
        print(f'  {cond:<8} line SNR: welch {r["snr_welch"]:5.1f} dB, '
              f'order {r["snr_order"]:5.1f} dB | codec: welch {r["bytes_welch"]:6.1f}, '
              f'order {r["bytes_order"]:6.1f} B/frame')
    return out

if __name__ == '__main__':
    t0 = time.time()
    res = dict(bands=study_bands(), viterbi=study_viterbi(),
               order=study_order())
    json.dump(res, open('../eval_dsp.json', 'w'), default=float)
    print(f'\nwrote ../eval_dsp.json  ({time.time()-t0:.0f}s)')
