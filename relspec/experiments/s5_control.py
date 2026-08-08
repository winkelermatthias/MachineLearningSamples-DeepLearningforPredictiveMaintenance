"""Stage 5: control experiment on genuinely continuous data.
Exp B (s4) jumps between different CWRU recordings every frame, which is far more
volatile than 2-samples/day telemetry from one asset. This measures the codec on
28 sequential non-overlapping windows from a single continuous recording."""
import numpy as np, pickle, scipy.io as sio, scipy.signal as sg, zstandard as zstd
from s4_run import (binify, to_u8, to_amp, Codec, run_gop, zsize, ACC_EDGES,
                    ENV_EDGES, Q_DB)
from s3_align import estimate_speed_env

FS=12000.; WIN=1.0; NPS=8192

def spec(x, nps=NPS):
    f,p = sg.welch(x, fs=FS, nperseg=min(nps,len(x)), noverlap=min(nps,len(x))//2,
                   window='hann', scaling='spectrum', detrend='constant')
    return f, np.sqrt(np.maximum(p,0))*np.sqrt(2)

def espec(x, nps=NPS, band=(2000.,5000.)):
    sos=sg.butter(6,[band[0]/(FS/2),band[1]/(FS/2)],btype='band',output='sos')
    e=np.abs(sg.hilbert(sg.sosfiltfilt(sos,x))); e-=e.mean()
    return spec(e,nps)

def build_seq(fid, n_frames=28, win_s=WIN):
    m=sio.loadmat(f'data/{fid}.mat')
    de=[k for k in m if k.endswith('_DE_time')][0]
    x=m[de].ravel().astype(np.float64)
    n=int(win_s*FS); avail=len(x)//n
    print(f"file {fid}: {len(x):,} samples = {len(x)/FS:.1f} s -> {avail} windows of {win_s}s")
    idx=np.linspace(0, avail-1, min(n_frames,avail)).astype(int)
    out=[]
    for k,i in enumerate(idx):
        seg=x[i*n:(i+1)*n]
        fa,aa=spec(seg); fe,ae=espec(seg)
        fr,conf=estimate_speed_env(seg)
        out.append(dict(idx=k,seg=seg,f_acc=fa,a_acc=aa,f_env=fe,a_env=ae,
                        fr_hat=fr,conf=conf,tier='B' if conf>=3.0 else ('C' if conf>=1.8 else 'D')))
    return out

def evaluate(frames, label, trig=0.60, mr_gate=0.50):
    for fr in frames:
        if fr['tier']=='D': fr['fr_hat']=frames[0]['fr_hat']
    ACC=np.stack([binify(f['f_acc'],f['a_acc'],f['fr_hat'],ACC_EDGES) for f in frames])
    ENV=np.stack([binify(f['f_env'],f['a_env'],f['fr_hat'],ENV_EDGES) for f in frames])
    UA=np.stack([to_u8(x) for x in ACC]); UE=np.stack([to_u8(x) for x in ENV])
    ma=np.maximum(np.median(np.abs(UA[:8].astype(int)-np.median(UA[:8],0)),0),1.)
    me=np.maximum(np.median(np.abs(UE[:8].astype(int)-np.median(UE[:8],0)),0),1.)
    tiers=[f['tier'] for f in frames]
    ca=Codec(768,trig=trig); ce=Codec(len(ENV_EDGES)-1,n_peaks=32,n_bands=12,trig=trig)
    ca.mr_gate=ce.mr_gate=mr_gate
    RA=run_gop(UA,ca,ma,tiers); RE=run_gop(UE,ce,me,tiers)
    ZA,ZE=zsize(RA),zsize(RE)
    RECA=np.stack([r['rec'] for r in RA]); RECE=np.stack([r['rec'] for r in RE])
    raw=sum(len(f['a_acc'])*4+len(f['a_env'])*4 for f in frames)
    zt=sum(ZA)+sum(ZE)
    na=sum(1 for r in RA if r['kind']=='anchor')
    resz=[z for z,r in zip(ZA,RA) if r['kind']=='residual']+[z for z,r in zip(ZE,RE) if r['kind']=='residual']
    eA=np.abs(RECA.astype(int)-UA.astype(int)).ravel()*Q_DB
    print(f"\n--- {label} ---")
    print(f"speed: median {np.median([f['fr_hat'] for f in frames]):.3f} Hz  "
          f"spread {100*(max(f['fr_hat'] for f in frames)-min(f['fr_hat'] for f in frames))/np.median([f['fr_hat'] for f in frames]):.2f}%  "
          f"tiers {dict((t,sum(1 for f in frames if f['tier']==t)) for t in 'BCD')}")
    print(f"anchors {na}/{len(frames)} (ACC)  {sum(1 for r in RE if r['kind']=='anchor')}/{len(frames)} (ENV)")
    print(f"residual bytes: median {np.median(resz):.0f}  mean {np.mean(resz):.0f}  p95 {np.percentile(resz,95):.0f}")
    print(f"raw {raw:,} B -> coded {zt:,} B  = {raw/zt:.1f}x")
    print(f"recon err ACC: median {np.median(eA):.2f} dB  p95 {np.percentile(eA,95):.2f}  p99 {np.percentile(eA,99):.2f}")
    print(f"telemetry: {2*zt/len(frames)/1024:.2f} KB/day/sensor")
    return dict(label=label,frames=frames,ACC=ACC,ENV=ENV,UA=UA,UE=UE,RECA=RECA,RECE=RECE,
                RA=RA,RE=RE,ZA=ZA,ZE=ZE,raw=raw,zt=zt,ratio=raw/zt,na=na,
                ACCr=np.stack([to_amp(u) for u in RECA]),ENVr=np.stack([to_amp(u) for u in RECE]))

if __name__=='__main__':
    print("="*70); print("CONTROL EXPERIMENTS: continuous single-asset sequences"); print("="*70)
    out={}
    for fid,lab in [('99','Exp A1: healthy, continuous (file 99, 2 HP)'),
                    ('209','Exp A2: IR 0.021in, continuous (file 209, 0 HP)'),
                    ('100','Exp A3: healthy, continuous (file 100, 3 HP)')]:
        fs_=build_seq(fid)
        out[fid]=evaluate(fs_, lab)
    pickle.dump(out, open('control.pkl','wb'))

    print("\n"+"="*70); print("TRIGGER SENSITIVITY on the volatile progression sequence"); print("="*70)
    D=pickle.load(open('frames.pkl','rb')); F=D['frames']
    for fr in F:
        fr['fr_hat'],fr['conf']=fr['fr_env'],fr['conf_env']
        fr['tier']='B' if fr['conf']>=3.0 else ('C' if fr['conf']>=1.8 else 'D')
    print(f"{'anchor trigger':>16}{'anchors':>10}{'ratio':>10}{'medErr':>9}")
    for trig in [0.60, 0.85, 1.20, 2.00]:
        r=evaluate([dict(f) for f in F], f"  trig={trig}", trig=trig)
        print(f"{trig:>16.2f}{r['na']:>10}{r['ratio']:>9.1f}x")
