"""Stage 1: build a 28-frame GOP from real CWRU records, compute ACC + ENV spectra."""
import numpy as np, scipy.io as sio, scipy.signal as sg, pickle, json

FS = 12000.0
WIN_S = 4.0                      # 4 s per frame
NPERSEG = 16384                  # 0.7324 Hz resolution
BEARING = dict(BPFI=5.4152, BPFO=3.5848, BSF=4.7135, FTF=0.3983)  # SKF 6205-2RS JEM

# CWRU 12k Drive End: (file, health_state, load_hp)
CATALOG = [
    ('97', 'normal', 0), ('98', 'normal', 1), ('99', 'normal', 2), ('100', 'normal', 3),
    ('105', 'IR_007', 0), ('106', 'IR_007', 1), ('107', 'IR_007', 2), ('108', 'IR_007', 3),
    ('169', 'IR_014', 0), ('170', 'IR_014', 1), ('171', 'IR_014', 2), ('172', 'IR_014', 3),
    ('209', 'IR_021', 0), ('210', 'IR_021', 1), ('211', 'IR_021', 2), ('212', 'IR_021', 3),
]

# 28-frame GOP = 14 days x 2/day. Anchor is frame 0 (healthy).
# Health progresses; load wanders like a real process, giving genuine speed variation.
STAGE = ['normal']*8 + ['IR_007']*6 + ['IR_014']*8 + ['IR_021']*6
LOADS = [0,0,1,1,2,1,0,1, 1,2,2,3,2,1, 1,2,3,3,2,2,1,2, 2,3,3,2,3,3]
assert len(STAGE) == 28 and len(LOADS) == 28

def load_rec(fid):
    m = sio.loadmat(f'data/{fid}.mat')
    de = [k for k in m if k.endswith('_DE_time')][0]
    rpm = [k for k in m if k.endswith('RPM')]
    x = m[de].ravel().astype(np.float64)
    r = float(m[rpm[0]].ravel()[0]) if rpm else np.nan
    return x, r

def spectrum(x, fs=FS, nperseg=NPERSEG):
    """Linear-amplitude spectrum (g, peak-referenced) via Welch."""
    f, p = sg.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg//2,
                    window='hann', scaling='spectrum', detrend='constant')
    return f, np.sqrt(np.maximum(p, 0.0)) * np.sqrt(2.0)   # -> amplitude

def envelope_spectrum(x, fs=FS, band=(2000., 5000.), nperseg=NPERSEG):
    sos = sg.butter(6, [band[0]/(fs/2), band[1]/(fs/2)], btype='band', output='sos')
    xb = sg.sosfiltfilt(sos, x)
    env = np.abs(sg.hilbert(xb))
    env = env - env.mean()
    f, p = sg.welch(env, fs=fs, nperseg=nperseg, noverlap=nperseg//2,
                    window='hann', scaling='spectrum', detrend='constant')
    return f, np.sqrt(np.maximum(p, 0.0)) * np.sqrt(2.0)

# ---- build frames -----------------------------------------------------------
rec_cache, frames = {}, []
cursor = {}   # per-record read cursor so repeated use takes fresh data
n = int(WIN_S * FS)

for k, (st, ld) in enumerate(zip(STAGE, LOADS)):
    fid = [c[0] for c in CATALOG if c[1] == st and c[2] == ld][0]
    if fid not in rec_cache:
        rec_cache[fid] = load_rec(fid)
        cursor[fid] = 0
    x, rpm = rec_cache[fid]
    s = cursor[fid]
    if s + n > len(x):
        s = 0
    seg = x[s:s+n]
    cursor[fid] = s + n // 2
    f_acc, a_acc = spectrum(seg)
    f_env, a_env = envelope_spectrum(seg)
    frames.append(dict(idx=k, day=k/2.0, fid=fid, state=st, load=ld,
                       rpm=rpm, fr_hz=rpm/60.0, seg=seg,
                       f_acc=f_acc, a_acc=a_acc, f_env=f_env, a_env=a_env))
    print(f"frame {k:2d} day {k/2:5.1f}  {st:8s} load{ld}  file {fid:>4s}  "
          f"rpm {rpm:7.1f}  fr {rpm/60:6.3f} Hz  rms {seg.std():.4f} g")

rpms = np.array([fr['rpm'] for fr in frames])
print(f"\nspeed range {rpms.min():.0f}-{rpms.max():.0f} rpm  "
      f"spread {100*(rpms.max()-rpms.min())/rpms.mean():.2f}%")
print(f"acc bins {len(frames[0]['f_acc'])}  df {frames[0]['f_acc'][1]:.4f} Hz  "
      f"fmax {frames[0]['f_acc'][-1]:.0f} Hz")

with open('frames.pkl', 'wb') as fh:
    pickle.dump(dict(frames=frames, bearing=BEARING, fs=FS), fh)
print("\nwrote frames.pkl")
