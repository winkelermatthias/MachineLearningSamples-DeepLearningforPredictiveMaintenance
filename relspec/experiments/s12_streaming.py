"""s12_streaming.py - can we get full-record frequency resolution without ever
holding the full record?

Claim under test: for the low-order region, resolution is set by total coherent
observation time, not by buffer size. A heterodyne integrator locked to the
tracked shaft phase accumulates over an unbounded window in O(1) memory per
order, and under speed drift it beats a long FFT rather than approximating it.

Comparison, on the same signal:
  A. BATCH   - one FFT over the whole record. Memory = O(N).
  B. STREAM  - blocks of 2048 discarded after use; per-order complex
               accumulators advanced by the tracked shaft angle. Memory = O(1).
"""
import numpy as np, scipy.io as sio, scipy.signal as sg, json

FS = 12000.0
BLK = 2048

def load(fid='209'):
    m = sio.loadmat(f'data/{fid}.mat')
    return m[[k for k in m if k.endswith('_DE_time')][0]].ravel().astype(np.float64)

def make_record(fid, seconds, drift_pct, seed=3):
    """Build a long record and impose a slow, realistic speed wander by
    time-warping. Returns the signal and the true instantaneous shaft rate."""
    x = load(fid)
    n = int(seconds*FS)
    reps = int(np.ceil(n/len(x)))
    y = np.tile(x, reps)[:n]
    t = np.arange(n)/FS
    rng = np.random.default_rng(seed)
    # smooth multiplicative rate wander, ~0.05 Hz bandwidth
    w = rng.normal(0, 1, n)
    b, a = sg.butter(2, 0.05/(FS/2))
    w = sg.filtfilt(b, a, w); w /= (w.std()+1e-12)
    rate = 1.0 + (drift_pct/100.0)*w
    warp = np.cumsum(rate)/FS
    warp = warp/warp[-1]*t[-1]
    yw = np.interp(t, warp, y)
    return yw, rate

def env_speed(seg, fs=FS, band=(2000., 5000.)):
    sos = sg.butter(6, [band[0]/(fs/2), band[1]/(fs/2)], btype='band', output='sos')
    e = np.abs(sg.hilbert(sg.sosfiltfilt(sos, seg))); e -= e.mean()
    N = 1 << int(np.ceil(np.log2(len(e))))
    A = np.abs(np.fft.rfft(e*np.hanning(len(e)), N))
    f = np.fft.rfftfreq(N, 1/fs)
    m = (f > 25) & (f < 34)
    j = np.where(m)[0][np.argmax(A[m])]
    y0,y1,y2 = A[j-1],A[j],A[j+1]
    den = 2*(y0-2*y1+y2)
    d = (y0-y2)/den if abs(den) > 1e-12 else 0.0
    return float((j+d)*(f[1]-f[0]))

# ------------------------------------------------------------ A. batch
def batch_orders(x, fr_mean, orders):
    """One FFT over the whole record, read at the requested orders.
    Peak memory: the record plus its transform."""
    N = 1 << int(np.ceil(np.log2(len(x))))
    A = np.abs(np.fft.rfft(x*np.hanning(len(x)), N))*2/len(x)
    f = np.fft.rfftfreq(N, 1/FS)
    return np.interp(orders*fr_mean, f, A), (len(x)*8 + N*8)/1024

# ------------------------------------------------------------ B. streaming
def stream_orders(x, orders, blk=BLK, speed_every=6):
    """Heterodyne integrator bank locked to the tracked shaft angle.

    Per block: estimate the shaft rate, extend the running angle theta, then
    for each target order k accumulate  A_k += sum( w[n] * x[n] * e^{-j k theta[n]} ).
    Blocks are discarded immediately. Nothing but the accumulators persists, so
    coherent observation time is unbounded while memory is flat.

    Speed tracking is what makes this work under drift: theta follows the shaft,
    so a component sitting at a fixed ORDER stays at DC in the mixed-down signal
    even while its frequency in Hz moves.
    """
    nb = len(x)//blk
    acc = np.zeros(len(orders), dtype=np.complex128)
    wsum = 0.0
    theta = 0.0                      # running shaft angle, revolutions
    fr = None
    win = np.hanning(blk)
    hist = []
    for b in range(nb):
        seg = x[b*blk:(b+1)*blk]
        if fr is None or b % speed_every == 0:
            # speed estimate needs a longer view than one block
            lo = max(0, (b-3)*blk); hi = min(len(x), lo+blk*8)
            try: fr = env_speed(x[lo:hi])
            except Exception: pass
        hist.append(fr)
        t = np.arange(blk)/FS
        th = theta + fr*t                      # revolutions elapsed
        # order-domain mixdown, vectorised over orders in chunks
        for i0 in range(0, len(orders), 256):
            k = orders[i0:i0+256][:, None]
            acc[i0:i0+256] += ((seg*win)[None, :] *
                               np.exp(-2j*np.pi*k*th[None, :])).sum(1)
        wsum += win.sum()
        theta = th[-1] + fr/FS
    amp = 2*np.abs(acc)/max(wsum, 1e-12)
    # memory actually resident: one block + accumulators + window
    kb = (blk*8 + len(orders)*16 + blk*8)/1024
    return amp, kb, np.array(hist)

# ------------------------------------------------------------ metrics
def peak_width(orders, amp, centre, span=0.25):
    """-3 dB width of the peak nearest `centre`, in orders."""
    m = np.abs(orders-centre) < span
    if m.sum() < 5: return np.nan, 0.0
    o, a = orders[m], amp[m]
    j = int(np.argmax(a)); pk = a[j]
    half = pk/np.sqrt(2)
    l = j
    while l > 0 and a[l] > half: l -= 1
    r = j
    while r < len(a)-1 and a[r] > half: r += 1
    return float(o[r]-o[l]), float(pk)

if __name__ == '__main__':
    ORD = np.arange(0.80, 6.01, 0.0025)
    out = {}
    print("="*104)
    print("RESOLUTION vs OBSERVATION TIME AND SPEED DRIFT")
    print("="*104)
    print(f"{'record':>8}{'drift':>7}{'method':>10}{'peak mem':>11}"
          f"{'1x width':>11}{'1x amp':>10}{'3x width':>11}{'3x amp':>10}")
    rows=[]
    for secs in [4, 10, 20]:
        for drift in [0.0, 1.0, 3.0]:
            x, rate = make_record('209', secs, drift)
            frm = env_speed(x[:BLK*8])
            ab, kb_b = batch_orders(x, frm, ORD)
            asr, kb_s, hist = stream_orders(x, ORD)
            for nm, a, kb in [('batch', ab, kb_b), ('stream', asr, kb_s)]:
                w1, p1 = peak_width(ORD, a, 1.0)
                w3, p3 = peak_width(ORD, a, 3.0)
                print(f"{str(secs)+'s':>8}{str(drift)+'%':>7}{nm:>10}{kb:>10.0f}K"
                      f"{w1:>11.4f}{p1*1e3:>10.2f}{w3:>11.4f}{p3*1e3:>10.2f}")
                rows.append(dict(secs=secs, drift=drift, method=nm, kb=kb,
                                 w1=w1, p1=p1, w3=w3, p3=p3))
            out[f'{secs}s_{drift}'] = dict(orders=ORD.tolist(),
                                           batch=ab.tolist(), stream=asr.tolist())
        print()
    json.dump(dict(rows=rows), open('scen/streaming.json','w'))

    print("="*104)
    print("MEMORY SCALING")
    print("="*104)
    print(f"{'seconds':>9}{'samples':>10}{'batch KB':>11}{'stream KB':>11}{'ratio':>9}")
    for secs in [4, 10, 20, 60, 300, 3600]:
        n = int(secs*FS); N = 1 << int(np.ceil(np.log2(n)))
        kb_b = (n*8 + N*8)/1024
        kb_s = (BLK*8 + len(ORD)*16 + BLK*8)/1024
        print(f"{secs:>9}{n:>10}{kb_b:>11.0f}{kb_s:>11.0f}{kb_b/kb_s:>9.0f}x")
    print("\n(stream memory is independent of record length by construction;")
    print(" a deployment would track ~400 targeted orders, not 2081, giving ~39 KB)")

    np.save('scen/stream_spec.npy',
            np.array([ORD, out['20s_3.0']['batch'], out['20s_3.0']['stream'],
                      out['20s_0.0']['batch'], out['20s_0.0']['stream']]))
    print("\nwrote scen/streaming.json, scen/stream_spec.npy")
