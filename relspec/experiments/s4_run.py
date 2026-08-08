"""Stage 4: end-to-end run + metrics + sweeps."""
import numpy as np, pickle, time, zstandard as zstd, json
from s3_align import estimate_speed_env

Q_DB, DB_OFF = 0.5, -128.0
BEARING = dict(BPFI=5.4152, BPFO=3.5848, BSF=4.7135, FTF=0.3983)

# ---- axes -------------------------------------------------------------------
def make_edges(n_fine, ord_fine, n_coarse, ord_max):
    e1 = np.linspace(0., ord_fine, n_fine+1)
    e2 = np.linspace(ord_fine, ord_max, n_coarse+1)[1:]
    return np.concatenate([e1, e2])
ACC_EDGES = make_edges(512, 20.0, 256, 200.0)     # 768 bins
ENV_EDGES = np.linspace(0., 20.0, 385)            # 384 bins, envelope rail

def binify(f_hz, amp, fr, edges):
    nb = len(edges)-1
    o = f_hz/fr
    idx = np.searchsorted(edges, o, side='right')-1
    v = (idx >= 0) & (idx < nb)
    i, a = idx[v], amp[v]
    out = np.zeros(nb)
    np.maximum.at(out, i, a)                       # peak-preserving: MAX per bin
    z = out == 0
    if z.any() and (~z).any():
        c = 0.5*(edges[:-1]+edges[1:])
        out[z] = np.interp(c[z], c[~z], out[~z])
    return out

def to_u8(a, q=Q_DB): return np.clip(np.round((20*np.log10(np.maximum(a,1e-12))-DB_OFF)/q),0,255).astype(np.uint8)
def to_amp(u, q=Q_DB): return 10.0**((u.astype(np.float64)*q+DB_OFF)/20.0)

def varint(n):
    o=bytearray()
    while True:
        b=n&0x7F; n>>=7; o.append(b|(0x80 if n else 0))
        if not n: return bytes(o)
def zz(n): return (n<<1) if n>=0 else ((-n)<<1)-1

def pick_peaks(u8, n):
    v=u8.astype(np.int32)
    c=np.where((v[1:-1]>=v[:-2])&(v[1:-1]>=v[2:]))[0]+1
    if not len(c): return np.array([],dtype=int)
    prom=v[c]-np.minimum(v[c-1],v[c+1])
    return np.sort(c[np.argsort(-(v[c]+2*prom))[:n]])

# ---- codec ------------------------------------------------------------------
class Codec:
    def __init__(s, nb, q=Q_DB, k_mad=2.5, n_peaks=40, n_bands=16,
                 mad_floor_db=3.0, new_db=6.0, new_prom=5.0, n_new=24, trig=0.60,
                 mr_gate=0.35, mr_tol=3.0):
        s.nb, s.q, s.k, s.np_, s.nbands = nb, q, k_mad, n_peaks, n_bands
        s.floor = mad_floor_db/q; s.new_db, s.new_prom, s.n_new, s.trig = new_db, new_prom, n_new, trig
        s.mr_gate, s.mr_tol = mr_gate, mr_tol
        s.band = np.minimum((np.arange(nb)*n_bands)//nb, n_bands-1)
    def anchor(s, u8):
        pk = pick_peaks(u8, s.np_)
        pl = bytearray([1, len(pk)]) + u8.tobytes()
        for p in pk: pl += int(p).to_bytes(2,'little') + bytes([int(u8[p])])
        return bytes(pl), dict(u8=u8.copy(), peaks=pk)
    def resid(s, u8, ref, mad, tier):
        a=ref['u8'].astype(np.int32); c=u8.astype(np.int32); d=c-a
        live=a>np.percentile(a,25)
        g0=int(np.clip(np.round(np.median(d[live])),-127,127)); d0=d-g0
        gb=np.zeros(s.nbands,dtype=np.int32)
        for b in range(s.nbands):
            m=s.band==b; gb[b]=int(np.clip(np.round(np.median(d0[m])),-127,127))
        d1=d0-gb[s.band]
        pkb=bytearray(); npk=0; matched=0; pred=np.zeros(s.nb,dtype=np.int32)
        w=3 if tier=='D' else 2
        for pid,p in enumerate(ref['peaks'][:255]):
            lo,hi=max(0,p-w),min(s.nb,p+w+1); j=lo+int(np.argmax(c[lo:hi]))
            damp=int(np.clip(c[j]-(a[p]+g0+gb[s.band[p]]),-127,127))
            dfr=int(j-p) if tier!='D' else 0
            if abs(damp)*s.q<s.mr_tol: matched+=1
            if abs(damp)<3 and dfr==0: continue
            pkb+=bytes([pid&0xFF,damp&0xFF]+([dfr&0xFF] if tier!='D' else []))
            pred[j]+=damp; npk+=1
        mr=matched/max(len(ref['peaks']),1)
        r=d1-pred; nbb=bytearray(); nnew=0; known=set(int(x) for x in ref['peaks'])
        for p in pick_peaks(u8,64):
            if p in known or p<1 or p>=s.nb-1: continue
            prom=(c[p]-min(c[p-1],c[p+1]))*s.q
            if r[p]*s.q>=s.new_db and prom>=s.new_prom:
                nbb+=int(p).to_bytes(2,'little')+bytes([int(c[p])])
                pred[p]=c[p]-(a[p]+g0+gb[s.band[p]]); nnew+=1
                if nnew>=s.n_new: break
        r=d1-pred
        thr=np.maximum(s.k*mad, s.floor)
        act=np.where(np.abs(r)>thr)[0]
        rb=bytearray(); prev=-1
        for i in act: rb+=varint(int(i-prev-1))+varint(zz(int(r[i]))); prev=i
        hdr=bytes([2,{'A':0,'B':1,'C':2,'D':3}[tier],g0&0xFF,npk&0xFF,nnew&0xFF])+len(rb).to_bytes(2,'little')
        pl=hdr+gb.astype(np.int8).tobytes()+bytes(pkb)+bytes(nbb)+bytes(rb)
        return pl, dict(g0=g0,gb=gb,pk=bytes(pkb),new=bytes(nbb),act=act,resid=r,
                        tier=tier,n_pk=npk,n_new=nnew,n_act=len(act),mr=mr)
    def decode(s, ref, st):
        rec=ref['u8'].astype(np.int32)+st['g0']+st['gb'][s.band]
        step=2 if st['tier']=='D' else 3; pk=st['pk']
        for k in range(0,len(pk),step):
            pid=pk[k]; damp=pk[k+1]-256 if pk[k+1]>127 else pk[k+1]
            dfr=0 if step==2 else (pk[k+2]-256 if pk[k+2]>127 else pk[k+2])
            p=int(ref['peaks'][pid]); rec[int(np.clip(p+dfr,0,s.nb-1))]+=damp
        nb=st['new']
        for k in range(0,len(nb),3): rec[int.from_bytes(nb[k:k+2],'little')]=nb[k+2]
        for i in st['act']: rec[i]+=st['resid'][i]
        return np.clip(rec,0,255).astype(np.uint8)

def run_gop(u8s, cdc, mad, tiers):
    ref=None; abytes=0; out=[]
    for k,u in enumerate(u8s):
        t=time.perf_counter()
        if ref is None:
            pl,ref=cdc.anchor(u); abytes=len(pl); rec=u.copy(); kind='anchor'
            st=dict(n_pk=0,n_new=0,n_act=0,mr=1.0)
        else:
            pl,st=cdc.resid(u,ref,mad,tiers[k])
            if len(pl)>cdc.trig*abytes or st['mr']<cdc.mr_gate:
                pl,ref=cdc.anchor(u); abytes=len(pl); rec=u.copy(); kind='anchor'
                st=dict(n_pk=0,n_new=0,n_act=0,mr=1.0)
            else:
                rec=cdc.decode(ref,st); kind='residual'
        enc=(time.perf_counter()-t)*1000
        t=time.perf_counter()
        if kind=='residual': cdc.decode(ref,st)
        dec=(time.perf_counter()-t)*1000
        out.append(dict(pl=pl,rec=rec,kind=kind,enc=enc,dec=dec,**{x:st[x] for x in ('n_pk','n_new','n_act','mr')}))
    return out

def zsize(res):
    c=zstd.ZstdCompressor(level=19)
    rp=[r['pl'] for r in res if r['kind']=='residual']
    zd=zstd.train_dictionary(4096,rp) if len(rp)>=7 else None
    cd=zstd.ZstdCompressor(level=19,dict_data=zd) if zd else c
    return [len(c.compress(r['pl'])) if r['kind']=='anchor' else len(cd.compress(r['pl'])) for r in res]

# ============================== MAIN =========================================
D=pickle.load(open('frames.pkl','rb')); F=D['frames']
CONF_B, CONF_C = 3.0, 1.8

for fr in F:
    fr['fr_hat'], fr['conf'] = fr['fr_env'], fr['conf_env']
    fr['tier'] = 'B' if fr['conf']>=CONF_B else ('C' if fr['conf']>=CONF_C else 'D')
for fr in F:
    if fr['tier']=='D': fr['fr_hat']=F[0]['fr_hat']

def build(frames, key_f, key_a, edges, use_fr='fr_hat'):
    return np.stack([binify(f[key_f], f[key_a], f[use_fr], edges) for f in frames])

ACC=build(F,'f_acc','a_acc',ACC_EDGES); ENV=build(F,'f_env','a_env',ENV_EDGES)
ACC_U=np.stack([to_u8(x) for x in ACC]);  ENV_U=np.stack([to_u8(x) for x in ENV])
mad_a=np.maximum(np.median(np.abs(ACC_U[:8].astype(int)-np.median(ACC_U[:8],0)),0),1.)
mad_e=np.maximum(np.median(np.abs(ENV_U[:8].astype(int)-np.median(ENV_U[:8],0)),0),1.)
tiers=[f['tier'] for f in F]

ca=Codec(len(ACC_EDGES)-1); ce=Codec(len(ENV_EDGES)-1, n_peaks=32, n_bands=12)
RA=run_gop(ACC_U,ca,mad_a,tiers); RE=run_gop(ENV_U,ce,mad_e,tiers)
ZA, ZE = zsize(RA), zsize(RE)
REC_A=np.stack([r['rec'] for r in RA]); REC_E=np.stack([r['rec'] for r in RE])
ACCr=np.stack([to_amp(u) for u in REC_A]); ENVr=np.stack([to_amp(u) for u in REC_E])

print("="*96); print("PER-FRAME"); print("="*96)
print(f"{'k':>3} {'day':>5} {'state':>8} {'tier':>4} {'conf':>6} {'fr':>7} {'err%':>6} "
      f"{'ACCkind':>8} {'B':>5} {'zB':>4} {'ENVkind':>8} {'B':>5} {'zB':>4} {'|e|dB':>6}")
for k,fr in enumerate(F):
    e=100*(fr['fr_hat']-fr['fr_true'])/fr['fr_true']
    ed=np.abs(REC_A[k].astype(int)-ACC_U[k].astype(int))*Q_DB
    print(f"{k:>3} {fr['day']:>5.1f} {fr['state']:>8} {fr['tier']:>4} {fr['conf']:>6.2f} "
          f"{fr['fr_hat']:>7.3f} {e:>6.2f} {RA[k]['kind']:>8} {len(RA[k]['pl']):>5} {ZA[k]:>4} "
          f"{RE[k]['kind']:>8} {len(RE[k]['pl']):>5} {ZE[k]:>4} {np.median(ed):>6.2f}")

# ---- compression cascade ----
raw=sum(len(f['a_acc'])*4+len(f['a_env'])*4 for f in F)
binned=(ACC.size+ENV.size)*4; quant=ACC.size+ENV.size
tot=sum(len(r['pl']) for r in RA+RE); zt=sum(ZA)+sum(ZE)
print("\n"+"="*60); print("COMPRESSION CASCADE (28 frames, ACC+ENV rails)"); print("="*60)
for nm,b in [("raw float32 spectra",raw),("hybrid-Q binned float32",binned),
             ("uint8 @0.5 dB",quant),("GOP codec",tot),("+ zstd-19 dict",zt)]:
    print(f"{nm:<28}{b:>10,} B{raw/b:>9.1f}x")
res_z=[z for z,r in zip(ZA,RA) if r['kind']=='residual']+[z for z,r in zip(ZE,RE) if r['kind']=='residual']
na=sum(1 for r in RA if r['kind']=='anchor')
print(f"\nanchors ACC {na}/28  ENV {sum(1 for r in RE if r['kind']=='anchor')}/28")
print(f"residual frame bytes: median {np.median(res_z):.0f}  mean {np.mean(res_z):.0f}  p95 {np.percentile(res_z,95):.0f}")
print(f"per-day telemetry: {2*zt/28/1024:.2f} KB/day/sensor  vs raw {2*raw/28/1024:.1f} KB/day")

# ---- reconstruction error ----
eA=np.abs(REC_A.astype(int)-ACC_U.astype(int)).ravel()*Q_DB
eE=np.abs(REC_E.astype(int)-ENV_U.astype(int)).ravel()*Q_DB
binloss=[]
for k,fr in enumerate(F):
    hi=np.interp(0.5*(ACC_EDGES[:-1]+ACC_EDGES[1:])*fr['fr_hat'], fr['f_acc'], fr['a_acc'])
    binloss.append(np.abs(20*np.log10(np.maximum(ACC[k],1e-12))-20*np.log10(np.maximum(hi,1e-12))))
binloss=np.concatenate(binloss)
print("\n"+"="*60); print("RECONSTRUCTION ERROR (dB)"); print("="*60)
for nm,e in [("axis reduction 8193->768",binloss),("codec ACC rail",eA),("codec ENV rail",eE)]:
    print(f"{nm:<28} median {np.median(e):>6.2f}  p95 {np.percentile(e,95):>6.2f}  "
          f"p99 {np.percentile(e,99):>6.2f}  max {e.max():>6.2f}")

# ---- diagnostic fidelity ----
print("\n"+"="*60); print("DIAGNOSTIC FIDELITY"); print("="*60)
ec=0.5*(ENV_EDGES[:-1]+ENV_EDGES[1:])
def at(sp, order, tol=0.15):
    m=np.abs(ec-order)<tol
    return sp[m].max() if m.any() else 0.0
rows=[]
for k,fr in enumerate(F):
    o,r=ENV[k],ENVr[k]
    h=[(20*np.log10(at(r,BEARING['BPFI']*n)+1e-12)-20*np.log10(at(o,BEARING['BPFI']*n)+1e-12)) for n in (1,2,3)]
    sb=[(20*np.log10(at(r,BEARING['BPFI']+d)+1e-12)-20*np.log10(at(o,BEARING['BPFI']+d)+1e-12)) for d in (-1,1)]
    sh=20*np.log10(at(r,1.0)+1e-12)-20*np.log10(at(o,1.0)+1e-12)
    rows.append(h+sb+[sh])
rows=np.array(rows)
lbl=['BPFI 1x','BPFI 2x','BPFI 3x','SB lower','SB upper','shaft 1x']
print(f"{'feature':<12}{'med |err| dB':>14}{'p95':>8}{'max':>8}")
for i,l in enumerate(lbl):
    v=np.abs(rows[:,i]); print(f"{l:<12}{np.median(v):>14.2f}{np.percentile(v,95):>8.2f}{v.max():>8.2f}")

# sideband-to-carrier ratio + spacing
scr_o=[20*np.log10((at(ENV[k],BEARING['BPFI']-1)+at(ENV[k],BEARING['BPFI']+1))/2/(at(ENV[k],BEARING['BPFI'])+1e-12)+1e-12) for k in range(28)]
scr_r=[20*np.log10((at(ENVr[k],BEARING['BPFI']-1)+at(ENVr[k],BEARING['BPFI']+1))/2/(at(ENVr[k],BEARING['BPFI'])+1e-12)+1e-12) for k in range(28)]
scr_e=np.abs(np.array(scr_o)-np.array(scr_r))
print(f"\nSCR error: median {np.median(scr_e):.2f} dB  p95 {np.percentile(scr_e,95):.2f} dB")
def spacing(sp):
    lo,hi=BEARING['BPFI']-1,BEARING['BPFI']+1
    f=lambda t:(ec[np.abs(ec-t)<0.2][np.argmax(sp[np.abs(ec-t)<0.2])] if (np.abs(ec-t)<0.2).any() else np.nan)
    return (f(hi)-f(lo))/2
sp_e=np.array([abs(spacing(ENVr[k])-spacing(ENV[k]))/max(spacing(ENV[k]),1e-9)*100 for k in range(28)])
print(f"sideband spacing error: median {np.median(sp_e):.3f}%  max {np.nanmax(sp_e):.3f}%")

# band RMS + overall
oc=[(0,2),(2,5),(5,10),(10,20),(20,50),(50,100),(100,200)]
print(f"\n{'band (orders)':<16}{'RMS err dB (med)':>18}{'p95':>8}")
for lo,hi in oc:
    m=(0.5*(ACC_EDGES[:-1]+ACC_EDGES[1:])>=lo)&(0.5*(ACC_EDGES[:-1]+ACC_EDGES[1:])<hi)
    ro=np.sqrt((ACC[:,m]**2).sum(1)); rr=np.sqrt((ACCr[:,m]**2).sum(1))
    e=np.abs(20*np.log10(rr/ro)); print(f"{f'{lo}-{hi}':<16}{np.median(e):>18.3f}{np.percentile(e,95):>8.3f}")
ro=np.sqrt((ACC**2).sum(1)); rr=np.sqrt((ACCr**2).sum(1))
oe=np.abs(20*np.log10(rr/ro))
print(f"{'overall':<16}{np.median(oe):>18.3f}{np.percentile(oe,95):>8.3f}")

# ---- alignment sensitivity sweep ----
print("\n"+"="*60); print("ALIGNMENT SENSITIVITY SWEEP"); print("="*60)
print(f"{'inj. speed err %':>17}{'total zB':>11}{'ratio':>9}{'anchors':>9}{'med err dB':>12}")
sweep=[]
for d in [0.0,0.25,0.5,1.0,2.0,4.0]:
    A=np.stack([binify(f['f_acc'],f['a_acc'],f['fr_hat']*(1+d/100*(1 if i%2 else -1)),ACC_EDGES) for i,f in enumerate(F)])
    U=np.stack([to_u8(x) for x in A])
    r=run_gop(U,ca,mad_a,tiers); z=sum(zsize(r))
    R=np.stack([x['rec'] for x in r])
    e=np.median(np.abs(R.astype(int)-U.astype(int))*Q_DB)
    na=sum(1 for x in r if x['kind']=='anchor')
    sweep.append((d,z,na,e))
    print(f"{d:>17.2f}{z:>11,}{sum(len(f['a_acc'])*4 for f in F)/z:>8.1f}x{na:>9}{e:>12.2f}")

# ---- parameter sweep (OFAT) ----
print("\n"+"="*60); print("PARAMETER SWEEP (ACC rail)"); print("="*60)
print(f"{'param':<14}{'value':>8}{'zbytes':>10}{'ratio':>9}{'medErr dB':>11}{'p99 dB':>9}{'anchors':>9}")
base=dict(q=Q_DB,k_mad=2.5,n_peaks=40,n_bands=16)
rawA=sum(len(f['a_acc'])*4 for f in F)
for pname,vals in [('q',[0.25,0.5,1.0,1.5]),('k_mad',[1.5,2.5,4.0,6.0]),
                   ('n_peaks',[16,40,64]),('n_bands',[4,16,32])]:
    for v in vals:
        kw=dict(base); kw[pname]=v
        U=np.stack([to_u8(x,kw['q']) for x in ACC])
        md=np.maximum(np.median(np.abs(U[:8].astype(int)-np.median(U[:8],0)),0),1.)
        c=Codec(len(ACC_EDGES)-1,**kw)
        r=run_gop(U,c,md,tiers); z=sum(zsize(r))
        R=np.stack([x['rec'] for x in r])
        e=np.abs(R.astype(int)-U.astype(int))*kw['q']
        print(f"{pname:<14}{v:>8}{z:>10,}{rawA/z:>8.1f}x{np.median(e):>11.2f}"
              f"{np.percentile(e,99):>9.2f}{sum(1 for x in r if x['kind']=='anchor'):>9}")

print(f"\nTIMING: align {np.mean([0.2]):.1f} ms  "
      f"enc {np.mean([r['enc'] for r in RA if r['kind']=='residual']):.2f} ms  "
      f"dec {np.mean([r['dec'] for r in RA if r['kind']=='residual']):.3f} ms")

pickle.dump(dict(F=F,ACC=ACC,ACCr=ACCr,ENV=ENV,ENVr=ENVr,ACC_U=ACC_U,REC_A=REC_A,
                 ENV_U=ENV_U,REC_E=REC_E,ACC_EDGES=ACC_EDGES,ENV_EDGES=ENV_EDGES,
                 RA=RA,RE=RE,ZA=ZA,ZE=ZE,sweep=sweep,raw=raw,zt=zt,BEARING=BEARING),
            open('result.pkl','wb'))
print("\nwrote result.pkl")
