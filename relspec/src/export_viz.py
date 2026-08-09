"""Export sequences for the interactive explorer artifact.

For each sequence, every frame carries: both rails original + decoded
(base64 uint8), the pattern-ownership map (which bin belongs to which
pattern), per-track pattern energy original and through-codec, the frame's
speed and byte cost. Pattern identity is resolved across frames here in
Python (match_key tolerance), so the front end just draws.
"""
import sys, os, json, base64, time
sys.path.insert(0, os.path.dirname(__file__) or '.')
import numpy as np
from scipy.signal import welch
from relspec.pipeline import (CENTERS, ENV_CENTERS, to_amp, to_u8, NBINS,
                              ENV_BINS, Gate)
from relspec.codec2 import Codec2, Decoder2
from relspec.pipeline2 import extract2, band_for
from relspec.dsp2 import envelope_banded
from relspec.patterns import extract_patterns
from relspec.synth2 import FaultState2, machine_catalog2, generate2, progression
from relspec import datasets as DS

B64 = lambda a: base64.b64encode(np.asarray(a, dtype=np.uint8).tobytes()).decode()

def fullres(x, fs, fr, band_key):
    """Full-resolution Welch spectra of both rails, quantised with the same
    0.5 dB step as the payload, capped at the order range the codec carries.
    This is what the analyst's FFT would have shown - the overlay against the
    decoded reconstruction is the honest no-loss/loss picture."""
    nps = DS.nperseg_for(fs)
    f, p = welch(x, fs=fs, nperseg=min(nps, len(x)),
                 noverlap=min(nps, len(x))//2, window='hann',
                 scaling='spectrum', detrend='constant')
    a = np.sqrt(np.maximum(p, 0))*np.sqrt(2)
    df = float(f[1]-f[0])
    hi = min(200.5*fr, 0.97*fs/2)
    na = int(hi/df)
    e = envelope_banded(x, fs, band_for(band_key, x, fs))
    fe, pe = welch(e, fs=fs, nperseg=min(nps, len(e)),
                   noverlap=min(nps, len(e))//2, window='hann',
                   scaling='spectrum', detrend='constant')
    ae = np.sqrt(np.maximum(pe, 0))*np.sqrt(2)
    ne = int(min(20.5*fr, 0.97*fs/2)/df)
    return to_u8(a[:na]), df, to_u8(ae[:ne])

def frame_patterns(amp, centers, is_env, verify=True):
    """verify=True for the measured spectrum (discovery must prove itself);
    False for the decoded side, whose job is only to show the energy is
    still where the verified pattern says it is."""
    kw = dict(min_harmonics=3, f_hi=6.5) if is_env else {}
    combs, mods, acc = extract_patterns(amp, centers, verify=verify, **kw)
    return acc

class TrackRegistry:
    """Stable pattern identity across frames: same kind, key within 4%."""
    def __init__(self):
        self.tracks = []          # list of dict(kind, key)
    def resolve(self, kind, key):
        best, err = None, 1e9
        for i, t in enumerate(self.tracks):
            if t['kind'] != kind: continue
            e = abs(key-t['key'])/max(abs(t['key']), 1e-9)
            if e < 0.04 and e < err: best, err = i, e
        if best is None:
            self.tracks.append(dict(kind=kind, key=float(key)))
            return len(self.tracks)-1
        # slow key update keeps the track centred as the pattern drifts
        self.tracks[best]['key'] = 0.9*self.tracks[best]['key']+0.1*float(key)
        return best

def process_sequence(name, title, gen, fs, fr_nominal, note=''):
    """gen yields (x, fr_true_or_None) acquisitions in time order."""
    ca = Codec2(nbins=NBINS, n_bands=16, refs='best')
    ce = Codec2(nbins=ENV_BINS, n_peaks=32, n_bands=12, refs='best')
    da = Decoder2(nbins=NBINS, n_bands=16)
    de = Decoder2(nbins=ENV_BINS, n_peaks=32, n_bands=12)
    mad_a = np.full(NBINS, 4.0); mad_e = np.full(ENV_BINS, 4.0)
    regs = dict(acc=TrackRegistry(), env=TrackRegistry())
    gate = Gate()
    frames = []
    t0 = time.time()
    ti_ = 0
    for x, fr_true in gen:
        e = extract2(x, fs, band_key=name, fr_nominal=fr_nominal)
        pa, ka = ca.encode(e.acc_u8, mad_a)
        bd_a = (None if ka == 'anchor' else
                [round(v/8) for v in (ca.last_breakdown['gains_bits'],
                                      ca.last_breakdown['peak_bits'],
                                      ca.last_breakdown['act_bits'])])
        pe, ke = ce.encode(e.env_u8, mad_e)
        bd_e = (None if ke == 'anchor' else
                [round(v/8) for v in (ce.last_breakdown['gains_bits'],
                                      ce.last_breakdown['peak_bits'],
                                      ce.last_breakdown['act_bits'])])
        ra, _ = da.decode(pa); re_, _ = de.decode(pe)
        mad_a = np.maximum(0.95*mad_a+0.05*np.abs(e.acc_u8.astype(float)-ra.astype(float)), 1.0)
        mad_e = np.maximum(0.95*mad_e+0.05*np.abs(e.env_u8.astype(float)-re_.astype(float)), 1.0)
        # the edge anomaly gate, exactly as the firmware policy runs it
        if gate.n == 0:
            gate.init(e); gd = dict(s=0.0, dr=0.0, dec='init', mb=0, k=0)
        else:
            s = gate.decide(e, ti_*43200)
            gd = dict(s=round(float(s['score']), 2),
                      dr=round(float(s['drift']), 2), dec=s['decision'],
                      mb=int(s['mask_bands']), k=int(s['kind']),
                      df=int(s['dfeat']))
        ti_ += 1
        af, adf, ef = fullres(x, fs, e.fr, name)
        fr_rec = dict(fr=round(e.fr, 4), tier=e.tier,
                      fr_true=round(fr_true, 4) if fr_true else None,
                      bytes=len(pa)+len(pe), ba=len(pa), be=len(pe),
                      kind='A' if 'anchor' in (ka+ke) else
                           ('P' if 'residual_p' in (ka, ke) else 'R'),
                      a_o=B64(e.acc_u8), a_d=B64(ra),
                      e_o=B64(e.env_u8), e_d=B64(re_),
                      af=B64(af), adf=round(adf, 4), ef=B64(ef),
                      bd=dict(a=bd_a, e=bd_e), gate=gd)
        for rail, u_o, u_d, cen in (('acc', e.acc_u8, ra, CENTERS),
                                    ('env', e.env_u8, re_, ENV_CENTERS)):
            acc_o = frame_patterns(to_amp(u_o), cen, rail == 'env')
            owner = acc_o['owner']
            dec2 = to_amp(u_d).astype(np.float64)**2
            pats = []
            omap = np.full(len(owner), 255, dtype=np.uint8)
            for pi, p in enumerate(acc_o['patterns']):
                tid = regs[rail].resolve(p['kind'], p['key'])
                # decoded energy over the exact bins this verified pattern
                # owns - "is the energy still where the pattern says it is"
                e_dec = float(dec2[owner == p['pid']].sum())
                pats.append(dict(t=tid, kind=p['kind'], key=round(float(p['key']), 3),
                                 f0=round(float(p['f0']), 3),
                                 sp=(round(float(p['spacing']), 3) if p['spacing'] else None),
                                 e=float(p['energy']), ed=float(e_dec),
                                 share=round(float(p['share']), 4),
                                 nv=p.get('ver_harm'), vc=p.get('ver_claimed'),
                                 vf=p.get('ver_frac')))
                omap[owner == p['pid']] = min(pi, 254)
            fr_rec[rail+'_pat'] = pats
            fr_rec[rail+'_own'] = B64(omap)
            fr_rec[rail+'_res'] = round(float(acc_o['residual_share']), 4)
        frames.append(fr_rec)
    print(f'  {name}: {len(frames)} frames  {time.time()-t0:.0f}s', flush=True)
    return dict(name=name, title=title, note=note, fs=fs,
                fr_nominal=fr_nominal, frames=frames,
                tracks={r: regs[r].tracks for r in ('acc', 'env')})

# ------------------------------------------------------------- generators
def gen_synth(mtype, fault_field, prog_kind, onset_frac, n_frames, fr0,
              wander=0.8, sev_max=0.9, seed=42, vfd_jumps=False):
    cat = machine_catalog2()
    spec = cat[mtype]
    rng = np.random.default_rng(seed)
    setpoint = fr0
    for t in range(n_frames):
        if vfd_jumps and t % 15 == 7:
            setpoint = fr0*(1+rng.uniform(-0.10, 0.10))
        fr = setpoint*(1+rng.uniform(-0.01, 0.01))
        f = FaultState2()
        u = (t/n_frames-onset_frac)/max(1-onset_frac, 1e-9)
        if u > 0: setattr(f, fault_field, sev_max*progression(prog_kind, u))
        load = 0.65+0.3*np.sin(2*np.pi*t/14)**2
        x = generate2(spec, f, 2.0, fr, load=load, speed_wander_pct=wander,
                      rng=rng).astype(np.float64)
        yield x, fr

def gen_cwru(states_loads, win_s=2.0, nw=8):
    recs = {(r['state'], int(r['group'].rsplit('L', 1)[1])): r
            for r in DS.cwru_records()}
    for state, load in states_loads:
        r = recs.get((state, load))
        if r is None: continue
        n = int(win_s*r['fs'])
        for w in range(min(nw, len(r['x'])//n)):
            yield r['x'][w*n:(w+1)*n], r['fr_nominal']

def gen_mfpt(prefix, win_s=2.0):
    for r in DS.mfpt_records():
        if not r['name'].startswith(prefix): continue
        n = int(win_s*r['fs'])
        for w in range(min(3, len(r['x'])//n)):
            yield r['x'][w*n:(w+1)*n], r['fr_nominal']

def gen_seu(names, channel_recs, win_s=2.0, nw=8):
    for r in channel_recs:
        if r['name'] not in names: continue
        n = int(win_s*r['fs'])
        for w in range(min(nw, len(r['x'])//n)):
            yield r['x'][w*n:(w+1)*n], r['fr_nominal']

if __name__ == '__main__':
    out = []
    out.append(process_sequence(
        'synth_or', 'Synthetic gearbox - outer race, exponential onset',
        gen_synth('gearbox', 'outer_race', 'exponential', 0.33, 90, 16.2),
        12000.0, 16.2,
        'Fault onset at frame 30. Watch the BPFO comb rise in the envelope '
        'rail and its energy track climb below.'))
    out.append(process_sequence(
        'synth_vfd', 'Synthetic VFD pump - rotor bar, setpoint jumps',
        gen_synth('vfd_pump', 'rotor_bar', 'sigmoid', 0.4, 60, 33.0,
                  vfd_jumps=True, seed=9),
        12000.0, 33.0,
        'Speed setpoint jumps every 15 frames. In order view the shaft comb '
        'stays locked while the 2x line-frequency family (100 Hz) WANDERS; '
        'in Hz view the roles swap. That asymmetry is the whole argument '
        'for order normalisation.'))
    out.append(process_sequence(
        'cwru_ir007', 'CWRU inner race 0.007", loads 0-3 HP',
        gen_cwru([('IR007', 0), ('IR007', 1), ('IR007', 2), ('IR007', 3)]),
        12000.0, 29.95,
        'Four motor loads, 1797 down to 1730 rpm. Order view aligns the '
        'BPFI comb across the speed steps.'))
    out.append(process_sequence(
        'cwru_or014', 'CWRU outer race 0.014" @6:00, loads 0-3 HP',
        gen_cwru([('OR014', 0), ('OR014', 1), ('OR014', 2), ('OR014', 3)]),
        12000.0, 29.95, ''))
    out.append(process_sequence(
        'cwru_normal', 'CWRU healthy baseline, loads 0-3 HP',
        gen_cwru([('normal', 0), ('normal', 1), ('normal', 2), ('normal', 3)]),
        12000.0, 29.95,
        'What the codec spends when nothing happens: the bytes strip is the '
        'point of this one.'))
    out.append(process_sequence(
        'mfpt_or', 'MFPT outer race, variable load 25-300 lbs',
        gen_mfpt('OuterRaceFault_vload'), 48828.0, 25.0,
        'Same defect under seven loads at 48.8 kHz; BPFO at order 3.245.'))
    seu = DS.seu_records('bearingset')
    out.append(process_sequence(
        'seu_ball', 'SEU gearbox rig - ball fault, 20 Hz then 30 Hz',
        gen_seu(['ball_20_0', 'ball_30_2'], seu), 5120.0, 25.0,
        'A 50% speed change mid-sequence. 5.12 kHz rig: the fixed v1 '
        'demod band cannot exist here; the kurtogram band carries the rail.'))
    meta = dict(centers=[round(float(c), 4) for c in CENTERS],
                env_centers=[round(float(c), 4) for c in ENV_CENTERS],
                q_db=0.5, db_off=-128.0)
    json.dump(dict(meta=meta, sequences=out),
              open('../viz_data.json', 'w'), separators=(',', ':'))
    sz = os.path.getsize('../viz_data.json')/1e6
    print(f'wrote ../viz_data.json  {sz:.1f} MB')
