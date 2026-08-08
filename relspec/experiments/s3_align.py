"""Stage 3: envelope-domain tacholess speed estimation.
The ACC spectrum's low-frequency region is dominated by mount resonances, so HPS
on ACC locks onto a fixed structural line. The envelope demodulates the impact
train and exposes shaft 1x plus BPFI, which is where the speed evidence lives."""
import numpy as np, pickle, scipy.signal as sg

FS = 12000.0

def env_hires(seg, fs=FS, band=(2000., 5000.)):
    """High-resolution envelope spectrum. No Welch averaging: speed estimation
    wants frequency resolution, not variance reduction."""
    sos = sg.butter(6, [band[0]/(fs/2), band[1]/(fs/2)], btype='band', output='sos')
    e = np.abs(sg.hilbert(sg.sosfiltfilt(sos, seg)))
    e = (e - e.mean()) * np.hanning(len(e))
    n = 1 << int(np.ceil(np.log2(len(e))))
    A = np.abs(np.fft.rfft(e, n)) * 2.0 / len(e)
    f = np.fft.rfftfreq(n, 1/fs)
    return f, A

def parabolic(f, a, j):
    if j <= 0 or j >= len(a)-1: return f[j]
    y0, y1, y2 = a[j-1], a[j], a[j+1]
    d = (y0 - y2) / (2*(y0 - 2*y1 + y2) + 1e-30)
    return f[j] + d * (f[1]-f[0])

def estimate_speed_env(seg, lo=26.0, hi=32.0, harmonics=(1,2,3)):
    """HPS over the envelope spectrum with sub-bin parabolic refinement.
    Confidence = winning peak vs the best rival at least 0.5 Hz away."""
    f, A = env_hires(seg)
    grid = np.arange(lo, hi, 0.002)
    hps = np.ones_like(grid)
    for h in harmonics:
        hps *= np.maximum(np.interp(grid*h, f, A), 1e-12)
    hps = hps ** (1.0/len(harmonics))
    j = int(np.argmax(hps))
    fr0 = float(grid[j])
    # refine on the fundamental itself
    band = (f > fr0-0.6) & (f < fr0+0.6)
    if band.sum() > 3:
        idx = np.where(band)[0]
        k = idx[int(np.argmax(A[idx]))]
        fr0 = float(parabolic(f, A, k))
    rival = hps[np.abs(grid-grid[j]) > 0.5]
    conf = float(hps[j] / max(rival.max() if rival.size else 1e-12, 1e-12))
    return fr0, conf

if __name__ == '__main__':
    D = pickle.load(open('frames.pkl','rb')); F = D['frames']
    NOM = {0:1797.,1:1772.,2:1750.,3:1730.}
    print(f"{'k':>3} {'state':>8} {'true':>7} {'est':>7} {'err%':>7} {'conf':>7}")
    errs, confs = [], []
    for fr in F:
        true = (fr['rpm'] if np.isfinite(fr['rpm']) else NOM[fr['load']])/60.0
        est, conf = estimate_speed_env(fr['seg'])
        e = 100*(est-true)/true
        errs.append(e); confs.append(conf)
        fr['fr_true'], fr['fr_env'], fr['conf_env'] = true, est, conf
        print(f"{fr['idx']:>3} {fr['state']:>8} {true:>7.3f} {est:>7.3f} {e:>7.2f} {conf:>7.2f}")
    errs = np.array(errs); confs = np.array(confs)
    print(f"\nmedian |err| {np.median(np.abs(errs)):.3f}%   p95 {np.percentile(np.abs(errs),95):.3f}%"
          f"   max {np.abs(errs).max():.3f}%")
    print(f"conf range {confs.min():.2f} - {confs.max():.2f}   median {np.median(confs):.2f}")
    for g in [1.5, 2.0, 3.0, 5.0]:
        m = confs >= g
        if m.any():
            print(f"  gate conf>={g}: {m.sum():2d}/{len(F)} frames, "
                  f"median |err| {np.median(np.abs(errs[m])):.3f}%, max {np.abs(errs[m]).max():.3f}%")
    pickle.dump(D, open('frames.pkl','wb'))
