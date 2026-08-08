"""Pattern extraction over stored spectra, plus the loss evaluation.

The loss question this answers: after the codec, does the pattern layer see the
same patterns carrying the same energy? Run the identical extraction on the
spectrum the device held and on the spectrum the cloud decoded, then compare
pattern by pattern. Per-bin error is not the metric, because peak-preserving
bin collapse produces large per-bin numbers while leaving every pattern intact.
"""
from __future__ import annotations
import numpy as np, pandas as pd, time
from .pipeline import to_amp, CENTERS, ENV_CENTERS, Q_DB, Decoder
from .patterns import extract_patterns, match_key

def _load_pairs(db):
    """Decoded spectrum (what the cloud has) paired with the device's own
    quantised spectrum (what was measured). The codec_frame payload is replayed
    to recover the decoded side so the comparison uses the real decoder."""
    rows = db.q('''SELECT s.acq_id, s.rail, s.bins, a.channel_id, a.ts,
                          cf.payload, cf.frame_kind
                   FROM spectrum s
                   JOIN acquisition a  ON a.acq_id = s.acq_id
                   LEFT JOIN codec_frame cf ON cf.acq_id = s.acq_id
                   ORDER BY a.channel_id, s.rail, a.ts''')
    return rows

def run_patterns(db, rail='acc', verbose=True):
    rows = _load_pairs(db)
    rows = rows[rows.rail == rail]
    centers = CENTERS if rail == 'acc' else ENV_CENTERS
    pat_ids, pat_rows, pe_rows, eb_rows = {}, [], [], []
    # continue the id sequence across rails; each rail is a separate call
    got = db.q('SELECT COALESCE(MAX(pattern_id),0) AS m FROM pattern')
    nid = int(got.m[0])
    base_id = nid
    t0 = time.time()
    for ch, g in rows.groupby('channel_id'):
        known = {}
        for r in g.itertuples():
            amp = to_amp(np.frombuffer(r.bins, dtype=np.uint8))
            # The envelope rail is the modulation domain: demodulation maps
            # sidebands to baseband lines, so a comb here IS a modulation
            # frequency. It needs a looser harmonic requirement than the
            # acceleration rail, because envelope combs typically show three or
            # four harmonics before the demodulator's low-pass rolls them off.
            if rail == 'env':
                combs, mods, acc = extract_patterns(amp, centers,
                                                    min_harmonics=3, f_hi=6.5)
            else:
                combs, mods, acc = extract_patterns(amp, centers)
            n_c = sum(1 for p in acc['patterns'] if p['kind'] == 'comb')
            eb_rows.append((r.acq_id, rail, acc['e_total'],
                            acc['e_total']-acc['e_residual'], acc['e_residual'],
                            acc['residual_share'], n_c,
                            len(acc['patterns'])-n_c))
            seen_pid = {}
            for p in acc['patterns']:
                if p['energy'] <= 0: continue
                k = match_key(p['key'], known.get(p['kind'], []))
                if k is None:
                    k = round(float(p['key']), 3)
                    known.setdefault(p['kind'], []).append(k)
                sig = (int(ch), rail, p['kind'], k)
                if sig not in pat_ids:
                    nid += 1; pat_ids[sig] = nid
                    pat_rows.append([nid, int(ch), rail, p['kind'], k,
                                     r.ts, r.ts, 0])
                pid = pat_ids[sig]
                # Two nearby patterns can collapse onto the same stable key
                # within one acquisition. Merge their energy rather than
                # emitting a duplicate: the key names one pattern, and energy
                # is additive over disjoint bin sets by construction.
                if pid in seen_pid:
                    j = seen_pid[pid]
                    e = pe_rows[j][6] + p['energy']
                    pe_rows[j] = (pe_rows[j][0], pid, pe_rows[j][2], pe_rows[j][3],
                                  max(pe_rows[j][4], p['snr'] or 0), pe_rows[j][5],
                                  e, 10*np.log10(max(e, 1e-30)),
                                  pe_rows[j][8]+p['share'], pe_rows[j][9]+p['n_bins'])
                    continue
                pat_rows[pid-base_id-1][6] = r.ts; pat_rows[pid-base_id-1][7] += 1
                seen_pid[pid] = len(pe_rows)
                pe_rows.append((r.acq_id, pid, p['f0'], p['spacing'], p['snr'],
                                p['n_pairs'], p['energy'],
                                10*np.log10(max(p['energy'], 1e-30)),
                                p['share'], p['n_bins']))
        if verbose:
            print(f'  channel {ch}: {len(pat_ids)} patterns so far, '
                  f'{time.time()-t0:.0f}s', flush=True)
    ins = lambda t, r, c: db.con.executemany(
        f"INSERT INTO {t} ({c}) VALUES ({','.join(['?']*len(c.split(',')))})", r) if r else None
    ins('pattern', [tuple(x) for x in pat_rows],
        'pattern_id,channel_id,rail,kind,pkey,first_seen,last_seen,n_obs')
    ins('pattern_energy', pe_rows,
        'acq_id,pattern_id,f0,spacing,snr,n_pairs,energy,energy_db,share,n_bins')
    ins('energy_balance', eb_rows,
        'acq_id,rail,e_total,e_patterned,e_residual,residual_share,n_combs,n_mods')
    return dict(patterns=len(pat_rows), observations=len(pe_rows),
                secs=round(time.time()-t0, 1))

# ------------------------------------------------------------------ loss
def run_loss(db, rail='acc', device_truth=None, verbose=True):
    """device_truth: {acq_id: uint8 array} of the pre-transmission spectrum.
    Without it the ledger is skipped, because comparing the decoded spectrum to
    itself would report zero loss and mean nothing."""
    if not device_truth:
        return dict(skipped='no device truth supplied')
    rows = _load_pairs(db)
    rows = rows[(rows.rail == rail) & rows.acq_id.isin(device_truth)]
    centers = CENTERS if rail == 'acc' else ENV_CENTERS
    out = []
    for r in rows.itertuples():
        rec = to_amp(np.frombuffer(r.bins, dtype=np.uint8))
        org = to_amp(device_truth[r.acq_id])
        co, mo, ao = extract_patterns(org, centers)
        cr, mr, ar = extract_patterns(rec, centers)
        rk = {p['kind']: [q for q in ar['patterns'] if q['kind'] == p['kind']]
              for p in ao['patterns']}
        for p in ao['patterns']:
            if p['energy'] <= 0: continue
            cands = rk.get(p['kind'], [])
            best, err = None, 1e9
            for q in cands:
                e = abs(q['key']-p['key'])/max(p['key'], 1e-9)
                if e < 0.05 and e < err: best, err = q, e
            if best is None:
                out.append((r.acq_id, p['key'], p['kind'], p['energy'],
                            None, None, None, None, False))
            else:
                sp_err = None
                if p['spacing'] and best['spacing']:
                    sp_err = abs(best['spacing']-p['spacing'])/p['spacing']*100
                out.append((r.acq_id, p['key'], p['kind'], p['energy'],
                            best['energy'],
                            10*np.log10(max(best['energy'], 1e-30)
                                        / max(p['energy'], 1e-30)),
                            err*100, sp_err, True))
    if out:
        db.con.executemany(
            'INSERT INTO pattern_loss (acq_id,pkey,kind,energy_orig,energy_recon,'
            'energy_err_db,f0_err_pct,spacing_err_pct,recovered) '
            'VALUES (?,?,?,?,?,?,?,?,?)', out)
    return dict(compared=len(out),
                recovered=sum(1 for o in out if o[8]))

def report(db):
    q = lambda s: db.q(s)
    print('\n== energy balance: how much of the spectrum the patterns explain ==')
    print(q('''SELECT rail, count(*) AS n,
      round(median(residual_share)*100,1) AS med_residual_pct,
      round(quantile_cont(residual_share,0.95)*100,1) AS p95_residual_pct,
      round(median(n_combs),1) AS med_combs, round(median(n_mods),1) AS med_mods
      FROM energy_balance GROUP BY 1''').to_string(index=False))
    print('\n== pattern inventory ==')
    print(q('''SELECT kind, count(*) AS n_patterns,
      round(median(n_obs),1) AS med_obs, max(n_obs) AS max_obs
      FROM pattern GROUP BY 1''').to_string(index=False))
    print('\n== top tracked patterns by observation count ==')
    print(q('''SELECT p.channel_id, p.kind, round(p.pkey,3) AS pkey, p.n_obs,
      round(median(pe.energy_db),1) AS med_e_db,
      round(median(pe.share)*100,1) AS med_share_pct,
      round(median(pe.spacing),3) AS med_spacing
      FROM pattern p JOIN pattern_energy pe ON pe.pattern_id=p.pattern_id
      GROUP BY 1,2,3,4 ORDER BY p.n_obs DESC LIMIT 12''').to_string(index=False))
    ls = q('SELECT * FROM loss_summary')
    if len(ls):
        print('\n== LOSS: pattern energy preserved through the codec ==')
        print(ls.round(3).to_string(index=False))
        print(q('''SELECT kind,
          sum(CASE WHEN recovered THEN 1 ELSE 0 END) AS recovered,
          count(*) AS total FROM pattern_loss GROUP BY 1''').to_string(index=False))
    print('\n== pattern energy tracked over time (largest rises) ==')
    print(q('''SELECT channel_id, kind, round(pkey,3) AS pkey, n_days,
      round(base_db,1) AS base_db, round(recent_db,1) AS recent_db,
      round(recent_db-base_db,1) AS rise_db
      FROM pattern_energy_rise
      WHERE base_db IS NOT NULL AND recent_db IS NOT NULL
      ORDER BY recent_db-base_db DESC LIMIT 10''').to_string(index=False))
