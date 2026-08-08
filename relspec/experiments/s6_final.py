"""Stage 6: final run with the match-tolerance sized to Welch estimator variance."""
import numpy as np, pickle, scipy.io as sio, scipy.signal as sg
from s4_run import (binify, to_u8, to_amp, Codec, run_gop, zsize,
                    ACC_EDGES, ENV_EDGES, Q_DB, BEARING)
from s3_align import estimate_speed_env

FS=12000.

def spec(x,nps):
    f,p=sg.welch(x,fs=FS,nperseg=min(nps,len(x)),noverlap=min(nps,len(x))//2,
                 window='hann',scaling='spectrum',detrend='constant')
    return f,np.sqrt(np.maximum(p,0))*np.sqrt(2)
def espec(x,nps,band=(2000.,5000.)):
    sos=sg.butter(6,[band[0]/(FS/2),band[1]/(FS/2)],btype='band',output='sos')
    e=np.abs(sg.hilbert(sg.sosfiltfilt(sos,x))); e-=e.mean(); return spec(e,nps)

def seq_from_file(fid,n=28,win_s=2.0,nps=8192):
    m=sio.loadmat(f'data/{fid}.mat')
    x=m[[k for k in m if k.endswith('_DE_time')][0]].ravel().astype(np.float64)
    w=int(win_s*FS); avail=len(x)//w
    idx=np.arange(min(n,avail))
    out=[]
    for k,i in enumerate(idx):
        s=x[i*w:(i+1)*w]; fa,aa=spec(s,nps); fe,ae=espec(s,nps)
        fr,cf=estimate_speed_env(s)
        out.append(dict(idx=k,day=k/2.,seg=s,f_acc=fa,a_acc=aa,f_env=fe,a_env=ae,
                        fr_hat=fr,conf=cf,state='cont',
                        tier='B' if cf>=3. else ('C' if cf>=1.8 else 'D')))
    return out

def evaluate(frames,label,**kw):
    for f in frames:
        if f['tier']=='D': f['fr_hat']=frames[0]['fr_hat']
    ACC=np.stack([binify(f['f_acc'],f['a_acc'],f['fr_hat'],ACC_EDGES) for f in frames])
    ENV=np.stack([binify(f['f_env'],f['a_env'],f['fr_hat'],ENV_EDGES) for f in frames])
    UA=np.stack([to_u8(x) for x in ACC]); UE=np.stack([to_u8(x) for x in ENV])
    ma=np.maximum(np.median(np.abs(UA[:8].astype(int)-np.median(UA[:8],0)),0),1.)
    me=np.maximum(np.median(np.abs(UE[:8].astype(int)-np.median(UE[:8],0)),0),1.)
    t=[f['tier'] for f in frames]
    ca=Codec(768,**kw); ce=Codec(len(ENV_EDGES)-1,n_peaks=32,n_bands=12,**kw)
    RA=run_gop(UA,ca,ma,t); RE=run_gop(UE,ce,me,t)
    ZA,ZE=zsize(RA),zsize(RE)
    RECA=np.stack([r['rec'] for r in RA]); RECE=np.stack([r['rec'] for r in RE])
    raw=sum(len(f['a_acc'])*4+len(f['a_env'])*4 for f in frames); zt=sum(ZA)+sum(ZE)
    na=sum(1 for r in RA if r['kind']=='anchor')+sum(1 for r in RE if r['kind']=='anchor')
    rz=[z for z,r in zip(ZA,RA) if r['kind']=='residual']+[z for z,r in zip(ZE,RE) if r['kind']=='residual']
    e=np.abs(RECA.astype(int)-UA.astype(int)).ravel()*Q_DB
    d=dict(label=label,frames=frames,ACC=ACC,ENV=ENV,UA=UA,UE=UE,RECA=RECA,RECE=RECE,
           RA=RA,RE=RE,ZA=ZA,ZE=ZE,raw=raw,zt=zt,ratio=raw/zt,na=na,
           ACCr=np.stack([to_amp(u) for u in RECA]),ENVr=np.stack([to_amp(u) for u in RECE]),
           medres=np.median(rz) if rz else np.nan,p95res=np.percentile(rz,95) if rz else np.nan,
           err=e,nres=len(rz))
    print(f"{label:<46}{na:>3}/{2*len(frames):<4}{d['ratio']:>8.1f}x"
          f"{(np.median(rz) if rz else 0):>8.0f}{np.median(e):>8.2f}{np.percentile(e,95):>8.2f}"
          f"{np.percentile(e,99):>8.2f}{2*zt/len(frames)/1024:>9.2f}")
    return d

if __name__=='__main__':
    print("="*118)
    print(f"{'experiment':<46}{'anchors':>7}{'ratio':>9}{'resB':>8}{'med':>8}{'p95':>8}{'p99':>8}{'KB/day':>9}")
    print("="*118)

    D=pickle.load(open('frames.pkl','rb')); F=D['frames']
    for f in F:
        f['fr_hat'],f['conf']=f['fr_env'],f['conf_env']
        f['tier']='B' if f['conf']>=3. else ('C' if f['conf']>=1.8 else 'D')

    res={}
    res['B'] = evaluate([dict(x) for x in F], 'Exp B: fault progression (volatile, 4s frames)')
    for fid,lab,ws in [('99','Exp A1: healthy continuous, 2s frames',2.0),
                       ('100','Exp A3: healthy continuous, 2s frames',2.0),
                       ('99','Exp A2: healthy continuous, 4s frames',4.0)]:
        res[lab[:7]+str(ws)] = evaluate(seq_from_file(fid,28,ws,8192 if ws<3 else 16384), f'{lab} [{fid}]')

    print("\n"+"="*118)
    print("MATCH-TOLERANCE / GATE SWEEP  (tolerance must exceed Welch estimator scatter)")
    print("="*118)
    print(f"{'config':<46}{'anchors':>7}{'ratio':>9}{'resB':>8}{'med':>8}{'p95':>8}{'p99':>8}{'KB/day':>9}")
    cont=seq_from_file('99',28,2.0,8192)
    for tol in [1.5,3.0,4.5,6.0]:
        for gate in [0.35,0.20]:
            evaluate([dict(x) for x in cont], f'  continuous: mr_tol={tol} dB, gate={gate}',
                     mr_tol=tol, mr_gate=gate)
    print()
    for tol in [1.5,3.0,4.5,6.0]:
        evaluate([dict(x) for x in F], f'  progression: mr_tol={tol} dB, gate=0.35', mr_tol=tol)

    print("\n"+"="*118)
    print("ACQUISITION LENGTH  (more Welch averages -> less estimator scatter -> cheaper residuals)")
    print("="*118)
    print(f"{'config':<46}{'anchors':>7}{'ratio':>9}{'resB':>8}{'med':>8}{'p95':>8}{'p99':>8}{'KB/day':>9}")
    for ws,nps in [(1.0,8192),(2.0,8192),(4.0,8192),(4.0,16384),(8.0,8192)]:
        try:
            s_=seq_from_file('99',28,ws,nps)
            if len(s_)>=10: evaluate(s_, f'  win={ws}s nperseg={nps} ({len(s_)} frames)')
        except Exception as ex: print('  skip',ws,nps,ex)

    pickle.dump(res, open('final.pkl','wb'))
    print("\nwrote final.pkl")
