"""Fleet-scale run: synthesise, extract, gate, encode, transmit, decode, diagnose.

Waveforms are generated, consumed and discarded one acquisition at a time. At no
point does the run hold more than a single acquisition of raw signal, which is
the same constraint the firmware operates under and the reason this scales to a
fleet on a laptop.
"""
from __future__ import annotations
import numpy as np, pandas as pd, time, json
from dataclasses import asdict
from .synth import machine_catalog, FaultState, progression, generate
from .pipeline import (extract, Codec, Decoder, Gate, NBINS, NBANDS, EDGES,
                       CENTERS, ENV_CENTERS, to_amp, Q_DB)
from .diagnose import harmonic_families, sideband_family, match_forcing, classify

E_ACQ, E_CONN = 1.2, 20.0
E_BYTE = E_CONN/500_000          # session setup costs what 500 KB costs

FAULT_PLAN = [
    ('outer_race',     'exponential'), ('inner_race', 'sigmoid'),
    ('rolling_element','linear'),      ('imbalance',  'linear'),
    ('misalignment',   'step'),        ('looseness',  'exponential'),
    ('gear_wear',      'sigmoid'),     ('cavitation', 'intermittent'),
    ('lubrication',    'exponential'), (None, None),
]

def plan_fleet(specs, days=90, wake_h=6, seed=5):
    """Assign each channel a fault mode, onset and progression, plus a duty
    pattern. Roughly a third stay healthy: a detector evaluated only on faulty
    assets has no false-alarm rate."""
    rng = np.random.default_rng(seed)
    n_acq = int(days*24/wake_h)
    plans = {}
    for i, (chid, s) in enumerate(sorted(specs.items())):
        mode, prog = FAULT_PLAN[i % len(FAULT_PLAN)]
        if rng.random() < 0.30: mode, prog = None, None
        onset = float(rng.uniform(0.25, 0.65))*days if mode else None
        duty = ['continuous', 'duty12', 'irregular'][i % 3]
        blk = None
        if duty == 'irregular':
            blk = rng.random(n_acq) < 0.55
            for k in range(1, n_acq):
                if rng.random() < 0.75: blk[k] = blk[k-1]
        plans[chid] = dict(mode=mode, prog=prog, onset=onset, duty=duty,
                           blocks=blk, sev_max=float(rng.uniform(0.55, 1.0)),
                           load_period=float(rng.choice([7., 14., 30.])),
                           wander=float(rng.uniform(0.15, 0.9)))
    return plans, n_acq

def _running(duty, k, blocks):
    if duty == 'continuous': return True
    if duty == 'duty12':     return (k % 4) in (1, 2)
    return bool(blocks[k])

def run_fleet(db, specs, days=90, wake_h=6, seconds=1.0, seed=5, verbose=True):
    plans, n_acq = plan_fleet(specs, days, wake_h, seed)
    rng = np.random.default_rng(seed+1)
    cat = machine_catalog()
    wake_s = wake_h*3600
    t0 = np.datetime64('2026-01-01T00:00:00')

    acq, feat, frames, spec_rows, bdelta, truth, txs, txf = [], [], [], [], [], [], [], []
    hfam, sbfam = [], []
    ids = dict(acq=0, tx=0, fam=0, sb=0, anchor=0)
    anchors = []
    t_start = time.time()

    for ci, (chid, S) in enumerate(sorted(specs.items())):
        P = plans[chid]; spec = S['spec']; fr0 = S['fr']
        codec = Codec(); dec = Decoder(); gate = Gate()
        env_chid = chid+1                     # env rail channel
        queue, q_bytes, last_conn, burst_t0, burst_used = [], 0, None, None, 0

        for k in range(n_acq):
            t = k*wake_s
            ts = t0+np.timedelta64(t, 's')
            run = _running(P['duty'], k, P['blocks'])
            load = 0.55+0.45*np.sin(2*np.pi*(t/86400)/P['load_period'])
            fs_ = FaultState()
            sev = 0.0
            if run and P['mode']:
                u = (t/86400-P['onset'])/(days-P['onset'])
                if u > 0:
                    sev = P['sev_max']*progression(P['prog'], u)
                    setattr(fs_, P['mode'], sev)
            fr_true = fr0*(1+0.02*(load-0.75))
            x = generate(spec, fs_, seconds, fr_true, load, P['wander'], run, rng)

            e = extract(x)
            ids['acq'] += 1; aid = ids['acq']

            # ---- operating state: level split learned per channel elsewhere;
            # here the synthetic truth of "running" is not used, the level is.
            db_lvl = 20*np.log10(max(e.acc_rms, 1e-9))
            state = 'running' if db_lvl > -37.0 else 'idle'
            if state == 'idle':
                acq.append((aid, chid, ts, seconds, 12000.0, state, 'no_quiet',
                            None, e.fr, e.conf, e.tier, e.coh_s, 0))
                truth.append((aid, run, P['mode'], sev, fr_true, load))
                continue

            if gate.n == 0: gate.init(e)
            g = gate.decide(e, t)
            dec_name = g['decision']
            payload = b''
            kind = None
            if dec_name.startswith('up_'):
                payload, kind = codec.encode(e.acc_u8, gate.bin_mad)
                rec, _ = dec.decode(payload)
                err = np.abs(rec.astype(int)-e.acc_u8.astype(int))*Q_DB
                if kind == 'anchor':
                    ids['anchor'] += 1
                    anchors.append((ids['anchor'], chid, aid, ts, None, NBINS,
                                    20.0, 200.0, Q_DB, -128.0,
                                    e.acc_u8.tobytes(), b'', b''))
                frames.append((aid, ids['anchor'] if anchors else None, kind,
                               payload, len(payload), True,
                               float(np.median(err)), float(np.percentile(err, 99))))
                spec_rows.append((aid, 'acc', rec.tobytes()))
                spec_rows.append((aid, 'env', e.env_u8.tobytes()))
                for b in range(NBANDS):
                    bdelta.append((aid, b, b*200/NBANDS, (b+1)*200/NBANDS,
                                   float(g['band_delta'][b]), bool(g['over_mask'][b])))
                queue.append((aid, len(payload), t, g.get('urgent', False)))
                q_bytes += len(payload)

            acq.append((aid, chid, ts, seconds, 12000.0, state, dec_name,
                        g.get('kind'), e.fr, e.conf, e.tier, e.coh_s, len(payload)))
            feat.append((aid, e.acc_rms, e.vel_rms, e.acc_kurt, e.acc_crest,
                         e.env_rms, e.env_kurt, e.env_crest,
                         float(g['score']), float(g['drift']), int(g['mask_bands'])))
            truth.append((aid, run, P['mode'], sev, fr_true, load))

            # ---- connection scheduling: rare by default, broken only by onset
            urgent = any(q[3] for q in queue)
            due = (last_conn is None and queue) or \
                  (last_conn is not None and (t-last_conn) >= gate.conn_interval_s)
            full = len(queue) >= 12 or q_bytes >= 11000
            fire = False
            if urgent:
                if burst_t0 is None or t-burst_t0 > 2*86400:
                    burst_t0, burst_used = t, 0
                if burst_used < 3: burst_used += 1; fire = True
            if due or full: fire = True
            if fire and queue:
                ids['tx'] += 1
                trig = 'onset_burst' if urgent else ('queue_full' if full else 'schedule')
                txs.append((ids['tx'], S['sensor_id'], ts, len(queue), q_bytes,
                            trig, E_CONN, E_BYTE*q_bytes))
                for a_, _b, _t, _u in queue: txf.append((ids['tx'], a_))
                queue, q_bytes, last_conn = [], 0, t

        if verbose and (ci+1) % 4 == 0:
            print(f'  {ci+1}/{len(specs)} channels, {ids["acq"]} acquisitions, '
                  f'{time.time()-t_start:.0f}s')

    # ------------------------------------------------------------ load to db
    def ins(table, rows, cols):
        if rows:
            ph = ','.join(['?']*len(cols.split(',')))
            db.con.executemany(f'INSERT INTO {table} ({cols}) VALUES ({ph})', rows)
    ins('acquisition', acq, 'acq_id,channel_id,ts,duration_s,fs_hz,op_state,decision,'
        'change_kind,fr_hz,fr_conf,align_tier,coherence_s,payload_bytes')
    ins('feature', feat, 'acq_id,acc_rms,vel_rms,acc_kurt,acc_crest,env_rms,'
        'env_kurt,env_crest,score,drift,mask_bands')
    ins('codec_anchor', anchors, 'anchor_id,channel_id,acq_id,valid_from,valid_to,'
        'n_bins,ord_fine,ord_max,q_db,db_offset,bins,peak_idx,mad')
    ins('codec_frame', frames, 'acq_id,anchor_id,frame_kind,payload,n_bytes,'
        'decoded_ok,recon_med_db,recon_p99_db')
    ins('spectrum', spec_rows, 'acq_id,rail,bins')
    ins('band_delta', bdelta, 'acq_id,band,ord_lo,ord_hi,delta_db,over_mask')
    ins('transmission', txs, 'tx_id,sensor_id,ts,n_frames,n_bytes,trigger,setup_j,payload_j')
    ins('transmission_frame', txf, 'tx_id,acq_id')
    ins('synth_truth', truth, 'acq_id,running,fault_mode,severity,true_fr_hz,load_frac')
    return dict(n_acq=ids['acq'], n_tx=ids['tx'], secs=time.time()-t_start)

# ------------------------------------------------------------- diagnostics
def diagnose_fleet(db, max_per_channel=8):
    """Run harmonic and sideband detection on decoded spectra, resolve every
    family against the ontology, and write findings with evidence links."""
    forcing = db.q('''SELECT channel_id, forcing_id, label, family, order_value,
                             fault_modes, bearing_id FROM forcing_frequency_k''')
    by_ch = {c: g.to_dict('records') for c, g in forcing.groupby('channel_id')}
    rows = db.q('''SELECT s.acq_id, s.rail, s.bins, a.channel_id, a.ts,
                          f.vel_rms, f.env_kurt
                   FROM spectrum s JOIN acquisition a ON a.acq_id=s.acq_id
                   LEFT JOIN feature f ON f.acq_id=s.acq_id
                   WHERE s.rail='acc' ORDER BY a.channel_id, a.ts''')
    # resolve channel -> asset/component once, not once per finding
    cm = db.q('''SELECT ch.channel_id, a.asset_id, c.component_id FROM channel ch
        JOIN sensor sn ON sn.sensor_id=ch.sensor_id
        JOIN component c ON c.component_id=sn.component_id
        JOIN asset a ON a.asset_id=c.asset_id''')
    chan_map = {int(t.channel_id): (int(t.asset_id), int(t.component_id))
                for t in cm.itertuples()}
    hf, sb, find, ev = [], [], [], []
    ids = dict(f=0, s=0, d=0)
    seen = {}
    for r in rows.itertuples():
        n = seen.get(r.channel_id, 0)
        if n >= max_per_channel: continue
        seen[r.channel_id] = n+1
        amp = to_amp(np.frombuffer(r.bins, dtype=np.uint8))
        fams = harmonic_families(amp)
        sbs = []
        for f in fams[:4]:
            s = sideband_family(amp, f['f0'])
            if s: sbs.append(s)
        forc = by_ch.get(r.channel_id, [])
        for f in fams:
            m, err = match_forcing(f['f0'], forc)
            ids['f'] += 1
            hf.append((ids['f'], r.acq_id, 'acc', f['f0'], f['score'],
                       f['n_harm'], m['forcing_id'] if m is not None else None,
                       err))
        for s in sbs:
            ids['s'] += 1
            sb.append((ids['s'], r.acq_id, 'acc', s['carrier'], s['spacing'],
                       s['scr_db'], s['n_pairs'], None, None))
        feats = dict(vel_rms=r.vel_rms or 0.0, env_kurt=r.env_kurt or 3.0)
        for d in classify(fams, sbs, feats, forc)[:2]:
            ids['d'] += 1
            aid_row = chan_map[r.channel_id]
            find.append((ids['d'], aid_row[0], aid_row[1],
                         d.get('bearing_id'), r.ts, None, d['fault_mode'],
                         d['severity'], d['confidence'],
                         f"{d['label']} comb, {d['n_harm']} harmonics, "
                         f"{d['err_pct']:.1f}% from kinematic order"
                         + (f", sidebands at {d['sideband_spacing']:.2f} ord"
                            if d['sideband_spacing'] else ''),
                         r.acq_id))
            ev.append((ids['d'], r.acq_id, 'harmonic', ids['f'], 1.0))
    def ins(t, rws, cols):
        if rws:
            ph = ','.join(['?']*len(cols.split(',')))
            db.con.executemany(f'INSERT INTO {t} ({cols}) VALUES ({ph})', rws)
    ins('harmonic_family', hf, 'family_id,acq_id,rail,f0_order,score,n_harmonics,'
        'forcing_id,match_err_pct')
    ins('sideband_family', sb, 'sb_id,acq_id,rail,carrier_order,spacing_order,'
        'scr_db,n_pairs,carrier_forcing_id,spacing_forcing_id')
    ins('finding', find, 'finding_id,asset_id,component_id,bearing_id,opened_at,'
        'closed_at,fault_mode,severity,confidence,rationale,first_acq_id')
    ins('finding_evidence', ev, 'finding_id,acq_id,kind,ref_id,weight')
    return dict(families=len(hf), sidebands=len(sb), findings=len(find))

# ------------------------------------------------- materialised pyramid
def refresh_pyramid(db):
    """DuckDB stand-in for the TimescaleDB continuous aggregates. Peak-hold
    element-wise max, not mean: averaging across zoom levels erases the
    transients you zoomed out to find."""
    out = {}
    # rails have different bin counts, so they are collapsed independently.
    # Both land on a common 128-column render rail: the waterfall is an image
    # and the row budget is fixed no matter how many bins the codec used.
    for rail, nfine in (('acc', 512), ('env', 384)):
        rows = db.q('''SELECT a.ts, a.channel_id, a.fr_hz, s.bins
                       FROM spectrum s JOIN acquisition a ON a.acq_id=s.acq_id
                       WHERE s.rail = ?''', rail)
        if len(rows) == 0: continue
        arr = np.stack([np.frombuffer(b, dtype=np.uint8) for b in rows.bins])
        take = arr[:, :nfine]
        grp = nfine//128
        render = take.reshape(len(arr), 128, grp).max(2)
        db.con.executemany('INSERT INTO spec_render VALUES (?,?,?,?,?)',
            [(t, int(c), rail, float(f), render[i].tobytes())
             for i, (t, c, f) in enumerate(zip(rows.ts, rows.channel_id, rows.fr_hz))])
        df = pd.DataFrame(dict(ts=pd.to_datetime(rows.ts),
                               channel_id=rows.channel_id, fr=rows.fr_hz,
                               i=range(len(rows))))
        for level, freq in [('1h', 'h'), ('6h', '6h'), ('1d', 'D'), ('7d', '7D')]:
            df['bucket'] = df.ts.dt.floor(freq)
            recs = []
            for (b, c), g in df.groupby(['bucket', 'channel_id']):
                mx = render[g.i.values].max(0)
                recs.append((level, b, int(c), rail, len(g), float(g.fr.mean()),
                             mx.tobytes()))
            db.con.executemany('INSERT INTO wf_level VALUES (?,?,?,?,?,?,?)', recs)
            out[f'{rail}_{level}'] = out.get(f'{rail}_{level}', 0)+len(recs)
    return out

def refresh_order_trend(db):
    """One row per kinematic line per day. This is the table trending UIs and
    models should read; it exists because the ontology knows which bin each
    forcing frequency lands in."""
    forcing = db.q('''SELECT channel_id, forcing_id, label, family, order_value
                      FROM forcing_frequency_k WHERE order_value < 200''')
    rows = db.q('''SELECT a.ts, a.channel_id, s.rail, s.bins
                   FROM spectrum s JOIN acquisition a ON a.acq_id=s.acq_id
                   WHERE s.rail='acc' ''')
    if len(rows) == 0: return 0
    arr = np.stack([np.frombuffer(b, dtype=np.uint8) for b in rows.bins]).astype(float)
    amp_db = arr*Q_DB-128.0
    day = pd.to_datetime(rows.ts).dt.floor('D')
    recs = []
    for ch, fg in forcing.groupby('channel_id'):
        m = (rows.channel_id == ch).values
        if not m.any(): continue
        sub = amp_db[m]; d = pd.Series(day[m].values, name='day')
        bi = np.abs(CENTERS[None, :] - fg.order_value.values[:, None]).argmin(1)
        # one groupby over all forcing lines at once
        wide = pd.DataFrame(sub[:, bi], columns=[f'f{i}' for i in range(len(bi))])
        wide['day'] = d.values
        gmax = wide.groupby('day').max()
        gcnt = wide.groupby('day').size()
        meta = list(fg.itertuples())
        for b in gmax.index:
            row_vals = gmax.loc[b].values; n = int(gcnt.loc[b])
            for i, mt in enumerate(meta):
                recs.append((b, int(ch), 'acc', int(mt.forcing_id), mt.label,
                             mt.family, float(mt.order_value),
                             float(row_vals[i]), n))
    db.con.executemany('INSERT INTO order_trend VALUES (?,?,?,?,?,?,?,?,?)', recs)
    return len(recs)
