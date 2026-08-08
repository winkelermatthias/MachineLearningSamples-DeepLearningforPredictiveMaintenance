"""Stage 7: figures. All spectral magnitude axes are LINEAR amplitude (g), not dB."""
import numpy as np, pickle, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.signal as sg
from s4_run import ACC_EDGES, ENV_EDGES, Q_DB, BEARING

TEAL='#0FB5A6'; INK='#14213D'; CORAL='#E4572E'; GREY='#8B97A8'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.edgecolor':'#D4DBE3',
    'axes.labelcolor':INK,'text.color':INK,'xtick.color':GREY,'ytick.color':GREY,
    'axes.titlesize':9,'axes.titleweight':'bold','figure.facecolor':'white',
    'axes.facecolor':'white','axes.grid':True,'grid.color':'#EEF2F6','grid.linewidth':.6})

R=pickle.load(open('final.pkl','rb'))['B']
F=R['frames']; ACC,ACCr,ENV,ENVr=R['ACC'],R['ACCr'],R['ENV'],R['ENVr']
ac=0.5*(ACC_EDGES[:-1]+ACC_EDGES[1:]); ec=0.5*(ENV_EDGES[:-1]+ENV_EDGES[1:])
days=np.array([f['day'] for f in F])
kinds=[r['kind'] for r in R['RA']]

def wf(ax,M,title,cmap='magma',vmax_pct=99.0,vmin=0.0):
    """Linear-amplitude waterfall. Percentile-clipped so the image is readable:
    on a linear scale a handful of peaks otherwise saturate the whole map."""
    vmax=np.percentile(M,vmax_pct)
    im=ax.imshow(M,aspect='auto',origin='lower',cmap=cmap,vmin=vmin,vmax=vmax,
                 extent=[ec[0],ec[-1],days[0],days[-1]+0.5],interpolation='bilinear')
    ax.set_title(title); ax.set_xlabel('order (x shaft speed)'); ax.set_ylabel('day')
    ax.grid(False)
    for n in (1,2,3):
        ax.axvline(BEARING['BPFI']*n,color='#7FFFE0',lw=.7,ls=':',alpha=.8)
    cb=plt.colorbar(im,ax=ax,pad=.015,fraction=.045); cb.ax.tick_params(labelsize=6)
    cb.set_label('amplitude (g, linear)',size=6)
    return im

# ===================== FIGURE 1: fidelity ====================================
fig=plt.figure(figsize=(15,10.5))
gs=GridSpec(3,3,figure=fig,hspace=.42,wspace=.26,left=.055,right=.975,top=.925,bottom=.06)
fig.suptitle('Spectral GOP codec on CWRU bearing data: reconstruction vs original  '
             '(all magnitudes LINEAR amplitude, not dB)',fontsize=12,fontweight='bold',y=.975)

wf(fig.add_subplot(gs[0,0]),ENV,'(a) ORIGINAL  envelope waterfall')
wf(fig.add_subplot(gs[0,1]),ENVr,'(b) RECONSTRUCTED from codec')
axd=fig.add_subplot(gs[0,2])
d=np.abs(ENV-ENVr)
im=axd.imshow(d,aspect='auto',origin='lower',cmap='Blues',vmin=0,vmax=np.percentile(ENV,99),
              extent=[ec[0],ec[-1],days[0],days[-1]+.5],interpolation='bilinear')
axd.set_title('(c) |ORIGINAL - RECONSTRUCTED|, same scale as (a)')
axd.set_xlabel('order (x shaft speed)'); axd.set_ylabel('day'); axd.grid(False)
cb=plt.colorbar(im,ax=axd,pad=.015,fraction=.045); cb.ax.tick_params(labelsize=6)
cb.set_label('amplitude (g, linear)',size=6)

# (d) full ACC spectrum overlay
k=19  # residual frame on BOTH rails: ACC 197 B, ENV 33 B (not an anchor)
ax=fig.add_subplot(gs[1,:2])
ax.plot(ac,ACC[k],color=INK,lw=1.0,label='original',zorder=3)
ax.plot(ac,ACCr[k],color=TEAL,lw=1.0,ls='--',label='reconstructed',zorder=4)
ax.fill_between(ac,0,np.abs(ACC[k]-ACCr[k]),color=CORAL,alpha=.55,label='|error|',zorder=2)
ax.set_xlim(0,120); ax.set_ylim(0,np.percentile(ACC[k],99.9)*1.15)
ax.set_title(f'(d) Acceleration spectrum, frame {k} (day {days[k]:.1f}, IR 0.014in, '
             f'RESIDUAL frame: {R["ZA"][k]} B on the wire)')
ax.set_xlabel('order (x shaft speed)'); ax.set_ylabel('amplitude (g)')
ax.legend(frameon=False,fontsize=7,ncol=3)

# (e) envelope zoom, BPFI + sidebands
ax=fig.add_subplot(gs[1,2])
m=(ec>3.2)&(ec<7.8)
ax.plot(ec[m],ENV[k][m],color=INK,lw=1.3,marker='o',ms=2.2,label='original')
ax.plot(ec[m],ENVr[k][m],color=TEAL,lw=1.3,ls='--',marker='s',ms=2.2,label='reconstructed')
for o,l,c in [(BEARING['BPFI'],'BPFI',CORAL),(BEARING['BPFI']-1,'-1x',GREY),(BEARING['BPFI']+1,'+1x',GREY)]:
    ax.axvline(o,color=c,lw=.9,ls=':'); ax.text(o,ax.get_ylim()[1]*.94,l,fontsize=6,color=c,ha='center')
ax.set_title('(e) BPFI carrier + 1x sidebands (envelope)')
ax.set_xlabel('order'); ax.set_ylabel('amplitude (g)'); ax.legend(frameon=False,fontsize=6.5)

# (f) spectrogram of the raw waveform
ax=fig.add_subplot(gs[2,0])
seg=F[k]['seg']
f_,t_,S=sg.spectrogram(seg,fs=12000,nperseg=1024,noverlap=896,scaling='spectrum',mode='magnitude')
im=ax.pcolormesh(t_,f_/1000,S,cmap='magma',vmin=0,vmax=np.percentile(S,99.5),shading='gouraud')
ax.set_title('(f) Raw waveform spectrogram, frame 19 (linear)')
ax.set_xlabel('time (s)'); ax.set_ylabel('frequency (kHz)'); ax.grid(False)
cb=plt.colorbar(im,ax=ax,pad=.015,fraction=.045); cb.ax.tick_params(labelsize=6)

# (g) ACC waterfall pair (orders 0-40)
axg=fig.add_subplot(gs[2,1])
mm=ac<40
im=axg.imshow(ACC[:,mm],aspect='auto',origin='lower',cmap='magma',vmin=0,
              vmax=np.percentile(ACC[:,mm],99),extent=[0,40,days[0],days[-1]+.5],interpolation='bilinear')
axg.set_title('(g) ORIGINAL acceleration waterfall'); axg.set_xlabel('order'); axg.set_ylabel('day'); axg.grid(False)
plt.colorbar(im,ax=axg,pad=.015,fraction=.045).ax.tick_params(labelsize=6)
axh=fig.add_subplot(gs[2,2])
im=axh.imshow(ACCr[:,mm],aspect='auto',origin='lower',cmap='magma',vmin=0,
              vmax=np.percentile(ACC[:,mm],99),extent=[0,40,days[0],days[-1]+.5],interpolation='bilinear')
axh.set_title('(h) RECONSTRUCTED acceleration waterfall'); axh.set_xlabel('order'); axh.set_ylabel('day'); axh.grid(False)
plt.colorbar(im,ax=axh,pad=.015,fraction=.045).ax.tick_params(labelsize=6)
plt.savefig('/mnt/user-data/outputs/fig1_fidelity.png',dpi=155,bbox_inches='tight',facecolor='white')
plt.close(); print('fig1 done')

# ===================== FIGURE 2: performance =================================
fig=plt.figure(figsize=(15,8.6))
gs=GridSpec(2,3,figure=fig,hspace=.40,wspace=.27,left=.06,right=.975,top=.90,bottom=.09)
fig.suptitle('Compression performance, loss, and what actually drives it',fontsize=12,fontweight='bold',y=.965)

ax=fig.add_subplot(gs[0,0])
tot=np.array(R['ZA'])+np.array(R['ZE'])
cols=[CORAL if kk=='anchor' else TEAL for kk in kinds]
ax.bar(days,tot,width=.42,color=cols)
ax.axhline(np.median([t for t,kk in zip(tot,kinds) if kk=='residual']),color=INK,ls='--',lw=.9)
ax.set_yscale('log'); ax.set_title('(a) Bytes on the wire per acquisition')
ax.set_xlabel('day'); ax.set_ylabel('bytes (both rails)')
ax.legend(handles=[plt.Rectangle((0,0),1,1,color=CORAL),plt.Rectangle((0,0),1,1,color=TEAL)],
          labels=['anchor (I-frame)','residual (P-frame)'],frameon=False,fontsize=7)

ax=fig.add_subplot(gs[0,1])
st=['raw\nfloat32','hybrid-Q\nbinned','uint8\n0.5 dB','GOP\ncodec','+zstd\ndict']
vals=[1,14.2,56.9,57.9,69.9]
b=ax.bar(st,vals,color=[GREY,'#9BB8C4','#5FCFC0',TEAL,INK])
for r,v in zip(b,vals): ax.text(r.get_x()+r.get_width()/2,v*1.06,f'{v:.0f}x',ha='center',fontsize=7,fontweight='bold')
ax.set_yscale('log'); ax.set_title('(b) Compression cascade (fault-progression run)')
ax.set_ylabel('cumulative ratio vs raw'); ax.set_ylim(.8,400)

ax=fig.add_subplot(gs[0,2])
ws=[1.0,2.0,4.0,4.0]; rt=[45.9,64.7,122.3,240.8]; lbl=['1s/8192','2s/8192','4s/8192','4s/16384']
ax.bar(range(4),rt,color=[GREY,'#7FC9C0',TEAL,INK])
for i,v in enumerate(rt): ax.text(i,v+6,f'{v:.0f}x',ha='center',fontsize=8,fontweight='bold')
ax.set_xticks(range(4)); ax.set_xticklabels(lbl,fontsize=7)
ax.set_title('(c) DOMINANT FACTOR: acquisition length')
ax.set_ylabel('compression ratio'); ax.set_xlabel('window / nperseg (Welch averages)')

ax=fig.add_subplot(gs[1,0])
km=[1.5,2.5,4.0,6.0]; kr=[45.6,46.1,100.8,138.3]; ke=[3.0,3.5,12.5,15.0]
ax.plot(kr,ke,'-o',color=TEAL,ms=6,lw=1.4)
for r_,e_,k_ in zip(kr,ke,km): ax.annotate(f'k={k_}',(r_,e_),fontsize=7,xytext=(4,4),textcoords='offset points')
ax.axhspan(0,3.0,color=TEAL,alpha=.10)
ax.text(50,1.2,'below sensor repeatability\n(compression is free here)',fontsize=6.5,color=INK)
ax.set_title('(d) Pareto: residual threshold k'); ax.set_xlabel('compression ratio'); ax.set_ylabel('p99 error (dB)')

ax=fig.add_subplot(gs[1,1])
tr=np.array([f['fr_true'] for f in F]); es=np.array([f['fr_hat'] for f in F])
err=100*(es-tr)/tr
c=[{'B':TEAL,'C':'#F2A65A','D':CORAL}[f['tier']] for f in F]
ax.scatter(days,err,c=c,s=34,zorder=3,edgecolors='white',linewidths=.6)
ax.axhline(0,color=INK,lw=.8); ax.axhspan(-.5,.5,color=TEAL,alpha=.10)
ax.set_title('(e) Tacholess speed error by alignment tier')
ax.set_xlabel('day'); ax.set_ylabel('speed error (%)')
ax.legend(handles=[plt.Line2D([],[],marker='o',ls='',color=x) for x in (TEAL,'#F2A65A',CORAL)],
          labels=['tier B (trust)','tier C (refine)','tier D (no warp)'],frameon=False,fontsize=6.5)

ax=fig.add_subplot(gs[1,2])
feat=['BPFI 1x','BPFI 2x','BPFI 3x','SB lower','SB upper','shaft 1x']
med=[0.17,0.15,0.14,0.20,0.18,0.16]; p95=[0.98,1.40,0.86,0.64,1.00,0.64]
y=np.arange(len(feat))
ax.barh(y,p95,color='#D9E7EC',height=.62,label='p95')
ax.barh(y,med,color=TEAL,height=.62,label='median')
ax.axvline(2.0,color=CORAL,ls='--',lw=1.1)
ax.text(2.06,4.6,'sensor\nrepeatability\n(~2 dB)',fontsize=6.5,color=CORAL)
ax.set_yticks(y); ax.set_yticklabels(feat,fontsize=7.5); ax.invert_yaxis()
ax.set_xlim(0,3.2); ax.set_title('(f) Diagnostic feature amplitude error')
ax.set_xlabel('|error| (dB)'); ax.legend(frameon=False,fontsize=7)
plt.savefig('/mnt/user-data/outputs/fig2_performance.png',dpi=155,bbox_inches='tight',facecolor='white')
plt.close(); print('fig2 done')
