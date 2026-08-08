"""s11_energy.py - transmission economics under 'setup == 500 KB' and the
reconstruction evidence figure."""
import numpy as np, pandas as pd, subprocess, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from s10_cloud import load_payloads, Decoder, u8_to_amp, CEN, harmonic_families, sideband_family, BEARING

# Energy model. Session setup is stated to cost about what 500 KB of payload
# costs, which fixes the ratio; absolute values are LTE-M order-of-magnitude.
E_ACQ, E_CONN = 1.2, 20.0
E_BYTE = E_CONN/500_000            # 40 uJ/byte
ONSET = [30., 50., 70.]

def run(scn, lines, tag='e'):
    open(f'/tmp/{tag}.txt','w').write(''.join(f'{a} {b} {c}\n' for a,b,c in lines))
    subprocess.run(['./fw/sim',f'scen/{scn}.bin',f'/tmp/{tag}.csv',f'/tmp/{tag}.txt'],
                   capture_output=True)
    d = pd.read_csv(f'/tmp/{tag}.csv'); c = d[d.connected==1]
    ch = d[d.decision=='up_change']
    det, del_ = [], []
    for on in ONSET:
        a = ch[ch.day>=on]
        if len(a):
            dd = float(a.day.iloc[0]); det.append(dd-on)
            after = c[c.day>=dd]
            del_.append(float(after.day.iloc[0])-on if len(after) else np.nan)
        else: det.append(np.nan); del_.append(np.nan)
    return dict(d=d, up=int(d.decision.str.startswith('up_').sum()), conn=len(c),
        fpc=float(c.conn_frames.mean()) if len(c) else 0, kb=float(c.conn_bytes.sum())/1024,
        E=E_ACQ*len(d)+E_CONN*len(c)+E_BYTE*c.conn_bytes.sum(),
        Er=E_CONN*len(c), Eb=E_BYTE*c.conn_bytes.sum(), Ea=E_ACQ*len(d),
        det=det, dele=del_)

print("="*112)
print("TRANSMISSION ECONOMICS  (session setup == 500 KB of payload)")
print("="*112)
print(f"{'connect every':>14}{'conns':>7}{'frames/conn':>13}{'KB':>7}"
      f"{'E acq':>8}{'E radio':>9}{'E bytes':>9}{'E total':>9}{'detect':>9}{'delivered':>11}")
sweep=[]
for days in [1,2,3,5,7,10,14,21,28]:
    r = run('S4_duty_gradual', [(0,'conn_interval_s',days*86400),
                                (0,'batch_max_age_s',days*2*86400)], f'i{days}')
    sweep.append((days,r))
    print(f"{str(days)+' d':>14}{r['conn']:>7}{r['fpc']:>13.1f}{r['kb']:>7.1f}"
          f"{r['Ea']:>8.0f}{r['Er']:>9.0f}{r['Eb']:>9.2f}{r['E']:>9.0f}"
          f"{np.nanmean(r['det']):>9.2f}{np.nanmean(r['dele']):>11.2f}")

print("\n" + "="*112)
print("DOES A STEP CHANGE NEED TO BREAK THE SCHEDULE?  (connect every 14 d)")
print("="*112)
print(f"{'policy':>40}{'conns':>7}{'E total':>9}{'detect (d)':>12}{'delivered (d)':>15}")
burst=[]
for lbl, ex in [('3-day cadence, no burst',[(0,'conn_interval_s',3*86400),(0,'conn_burst_max',0)]),
                ('21-day, no burst',[(0,'conn_interval_s',21*86400),(0,'batch_max_age_s',42*86400),(0,'conn_burst_max',0)]),
                ('21-day, burst on every change',[(0,'conn_interval_s',21*86400),(0,'batch_max_age_s',42*86400),(0,'conn_burst_max',3),(0,'drift_breaks_schedule',1)]),
                ('21-day, burst on episode onset',[(0,'conn_interval_s',21*86400),(0,'batch_max_age_s',42*86400),(0,'conn_burst_max',3)])]:
    r = run('S4_duty_gradual', ex, 'b')
    burst.append((lbl,r))
    print(f"{lbl:>40}{r['conn']:>7}{r['E']:>9.0f}"
          f"{np.nanmean(r['det']):>12.2f}{np.nanmean(r['dele']):>15.2f}")

base = run('S4_duty_gradual', [(0,'conn_interval_s',3*86400)],'base')
naive_conn = int((base['d'].truth_running==1).sum())
naive_E = E_ACQ*len(base['d']) + E_CONN*naive_conn + E_BYTE*naive_conn*32772
best = run('S4_duty_gradual', [(0,'conn_interval_s',21*86400),(0,'batch_max_age_s',42*86400),
                               (0,'conn_burst_max',3)],'best')
print(f"\nnaive (upload every running acquisition, raw spectrum, one session each):")
print(f"  {naive_conn} sessions, {naive_conn*32772/1024:.0f} KB, {naive_E:.0f} J")
print(f"tuned (21-day cadence, burst on onset): {best['conn']} sessions, {best['kb']:.1f} KB, {best['E']:.0f} J"
      f"  -> {100*(1-best['E']/naive_E):.1f}% energy saved, "
      f"{naive_conn/best['conn']:.0f}x fewer sessions, {naive_conn*32772/1024/best['kb']:.0f}x less data")
yrs = lambda E: 18500.0/ (E/90*365)      # 18.5 kJ usable from a D-cell lithium pack
print(f"battery life estimate (18.5 kJ pack): naive {yrs(naive_E):.1f} y, "
      f"3-day cadence {yrs(base['E']):.1f} y, 14-day cadence + burst {yrs(best['E']):.1f} y")

# ============================== FIGURE =====================================
TEAL,INK,CORAL,GREY,AMBER='#0FB5A6','#0E2A2F','#E4572E','#869A9C','#D99A2B'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.edgecolor':'#D4DBE3',
 'axes.labelcolor':INK,'text.color':INK,'xtick.color':GREY,'ytick.color':GREY,
 'axes.titlesize':9,'axes.titleweight':'bold','figure.facecolor':'white',
 'axes.grid':True,'grid.color':'#EEF2F6','grid.linewidth':.6})

recs = load_payloads('scen/S4_duty_gradual.pl'); dec = Decoder()
frames=[]
for r in recs:
    rec,kind = dec.decode(r['payload'])
    frames.append(dict(day=r['t']/86400, kind=kind, b=len(r['payload']),
        o=u8_to_amp(r['truth']), rc=u8_to_amp(rec),
        err=np.abs(rec.astype(int)-r['truth'].astype(int))*0.5))

fig=plt.figure(figsize=(15,9.4))
gs=GridSpec(3,3,figure=fig,hspace=.46,wspace=.26,left=.055,right=.975,top=.915,bottom=.07)
fig.suptitle('Cloud-side decode of what the device actually transmitted, and the transmission economics',
             fontsize=12,fontweight='bold',y=.965)

res=[f for f in frames if f['kind']=='residual']
tgt=res[len(res)//2] if res else frames[-1]
ax=fig.add_subplot(gs[0,:2])
ax.plot(CEN,tgt['o'],color=INK,lw=1.0,label='what the device measured')
ax.plot(CEN,tgt['rc'],color=TEAL,lw=1.0,ls='--',label='decoded in the cloud')
ax.fill_between(CEN,0,np.abs(tgt['o']-tgt['rc']),color=CORAL,alpha=.6,label='|error|')
ax.set_xlim(0,120); ax.set_ylim(0,np.percentile(tgt['o'],99.9)*1.2)
ax.set_title(f"(a) Residual frame, day {tgt['day']:.1f}: {tgt['b']} B on the wire rebuilt into a full spectrum")
ax.set_xlabel('order (x shaft speed)'); ax.set_ylabel('amplitude (g, linear)')
ax.legend(frameon=False,fontsize=7,ncol=3)

ax=fig.add_subplot(gs[0,2])
m=(CEN>3.0)&(CEN<8.0)
ax.plot(CEN[m],tgt['o'][m],color=INK,lw=1.3,marker='o',ms=2.4,label='measured')
ax.plot(CEN[m],tgt['rc'][m],color=TEAL,lw=1.3,ls='--',marker='s',ms=2.4,label='decoded')
sb=sideband_family(tgt['rc'],BEARING['BPFI'])
ax.axvline(BEARING['BPFI'],color=CORAL,lw=1,ls=':')
ax.text(BEARING['BPFI'],ax.get_ylim()[1]*.93,'BPFI',fontsize=6.5,color=CORAL,ha='center')
if sb:
    for s_ in (-1,1):
        ax.axvline(BEARING['BPFI']+s_*sb['spacing'],color=GREY,lw=.9,ls=':')
    ax.set_title(f"(b) Sideband family: spacing {sb['spacing']:.2f} ord, SCR {sb['scr_db']:.1f} dB")
else: ax.set_title('(b) BPFI region')
ax.set_xlabel('order'); ax.set_ylabel('amplitude (g)'); ax.legend(frameon=False,fontsize=6.5)

ax=fig.add_subplot(gs[1,0])
allerr=np.concatenate([f['err'] for f in frames])
anc=np.concatenate([f['err'] for f in frames if f['kind']=='anchor'])
rsd=np.concatenate([f['err'] for f in res]) if res else allerr
bins=np.arange(0,12.5,.5)
ax.hist(anc,bins=bins,color=GREY,alpha=.75,label=f'anchor frames (n={sum(1 for f in frames if f["kind"]=="anchor")})')
ax.hist(rsd,bins=bins,color=TEAL,alpha=.75,label=f'residual frames (n={len(res)})')
ax.set_yscale('log'); ax.axvline(2.0,color=CORAL,ls='--',lw=1.1)
ax.text(2.1,ax.get_ylim()[1]*.3,'sensor\nrepeatability',fontsize=6.5,color=CORAL)
ax.set_title('(c) Decode error per bin'); ax.set_xlabel('|error| (dB)'); ax.set_ylabel('bins')
ax.legend(frameon=False,fontsize=6.5)

ax=fig.add_subplot(gs[1,1])
d=[s[0] for s in sweep]; conn=[s[1]['conn'] for s in sweep]; E=[s[1]['E'] for s in sweep]
ax.plot(d,E,'-o',color=TEAL,ms=5,lw=1.5)
for x,y,c_ in zip(d,E,conn): ax.annotate(f'{c_}',(x,y),fontsize=6.5,xytext=(0,6),
    textcoords='offset points',ha='center',color=GREY)
ax.set_title('(d) Energy vs connection cadence (labels = sessions)')
ax.set_xlabel('connect every N days'); ax.set_ylabel('energy over 90 days (J)')

ax=fig.add_subplot(gs[1,2])
lbl=['acquire','radio\nsetup','payload\nbytes']
b1=[base['Ea'],base['Er'],base['Eb']]; b2=[best['Ea'],best['Er'],best['Eb']]
x=np.arange(3); w=.36
ax.bar(x-w/2,b1,w,color=GREY,label='3-day cadence')
ax.bar(x+w/2,b2,w,color=TEAL,label='21-day + onset burst')
for i,(a,b) in enumerate(zip(b1,b2)):
    ax.text(i-w/2,a+8,f'{a:.0f}',ha='center',fontsize=6.5)
    ax.text(i+w/2,b+8,f'{b:.1f}' if b<10 else f'{b:.0f}',ha='center',fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels(lbl,fontsize=7.5)
ax.set_title('(e) Where the joules go'); ax.set_ylabel('J'); ax.legend(frameon=False,fontsize=7)

ax=fig.add_subplot(gs[2,0])
names=[b[0].replace(' may break it',' breaks').replace('schedule only, nothing breaks it','no burst') for b in burst]
det=[np.nanmean(b[1]['det']) for b in burst]; dele=[np.nanmean(b[1]['dele']) for b in burst]
y=np.arange(len(names))
ax.barh(y-.2,dele,.38,color=CORAL,label='delivered to cloud')
ax.barh(y+.2,det,.38,color=TEAL,label='detected on device')
ax.set_yticks(y); ax.set_yticklabels([n[:26] for n in names],fontsize=6.5); ax.invert_yaxis()
ax.set_title('(f) Burst on onset: delivery without the sessions'); ax.set_xlabel('days after fault onset')
ax.legend(frameon=False,fontsize=6.5)

ax=fig.add_subplot(gs[2,1])
by=[f['b'] for f in frames]; dy=[f['day'] for f in frames]
cols=[CORAL if f['kind']=='anchor' else TEAL for f in frames]
ax.bar(dy,by,width=1.1,color=cols)
ax.set_title('(g) Payload size per transmitted frame')
ax.set_xlabel('day'); ax.set_ylabel('bytes')
ax.legend(handles=[plt.Rectangle((0,0),1,1,color=CORAL),plt.Rectangle((0,0),1,1,color=TEAL)],
          labels=['anchor','residual'],frameon=False,fontsize=7)

ax=fig.add_subplot(gs[2,2])
sc=[]
for f in frames[::2]:
    fo=harmonic_families(f['o']); fr_=harmonic_families(f['rc'])
    so={round(x['f0'],1) for x in fo}; sr={round(x['f0'],1) for x in fr_}
    sc.append(len(so&sr)/max(len(so),1)*100)
ax.plot([f['day'] for f in frames[::2]],sc,'-o',color=TEAL,ms=4,lw=1.3)
ax.set_ylim(0,105); ax.axhline(100,color=GREY,ls=':',lw=1)
ax.set_title('(h) Harmonic families recovered after decode')
ax.set_xlabel('day'); ax.set_ylabel('% agreement with original')
plt.savefig('/mnt/user-data/outputs/fig3_cloud_energy.png',dpi=150,bbox_inches='tight',facecolor='white')
print('\nwrote fig3_cloud_energy.png')
