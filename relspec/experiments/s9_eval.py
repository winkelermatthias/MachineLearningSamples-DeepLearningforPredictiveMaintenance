"""s9_eval.py - efficacy evaluation of the edge gate."""
import pandas as pd, numpy as np, json, os, subprocess
from s8_scenarios import build, write_bin, run_sim, SCENARIOS, NACQ, WAKE_S, NSAMP

ONSET = {'S3_duty_step_fault': [45.0],
         'S7_incipient_blend': [25.0],
         'S4_duty_gradual':    [30.0, 50.0, 70.0],
         'S6_irreg_incipient': [50.0]}
NOFAULT = ['S1_always_healthy','S2_duty_healthy','S5_duty_loadcycle']

# energy model (assumptions, stated in the report)
E_ACQ, E_CONN, E_BYTE = 1.2, 8.0, 0.0015     # J, J, J/byte

def cfgfile(path, lines):
    with open(path,'w') as f:
        for t,k,v in lines: f.write(f"{t} {k} {v}\n")
    return path

def metrics(name, csv, meta):
    d = pd.read_csv(csv)
    up = d[d.decision.str.startswith('up_')]
    ch = d[d.decision=='up_change']
    hb = d[d.decision=='up_heartbeat']
    conn = d[d.connected==1]
    weeks = (NACQ*WAKE_S)/ (7*86400)
    m = dict(
        scenario=name, n_acq=len(d),
        n_idle=int((d.state=='idle').sum()), n_transient=int((d.state=='transient').sum()),
        n_unknown=int((d.state=='unknown').sum()),
        state_acc=float(((d.state=='running')==(d.truth_running==1))[d.state!='unknown'].mean()),
        disc_valid=bool(d.disc_valid.iloc[-1]), disc_thr=float(d.disc_thr_db.iloc[-1]),
        disc_sep=float(d.disc_sep.iloc[-1]),
        n_upload=len(up), n_change=len(ch), n_heartbeat=len(hb),
        uploads_per_week=len(up)/weeks,
        max_uploads_in_any_week=int(max(
            ((up.t_s>=w*7*86400)&(up.t_s<(w+1)*7*86400)).sum() for w in range(int(weeks)+1))),
        n_conn=len(conn), conn_per_week=len(conn)/weeks,
        bytes_total=int(conn.conn_bytes.sum()),
        frames_per_conn=float(conn.conn_frames.mean()) if len(conn) else 0.0,
        n_ratelimited=int((d.decision=='no_ratelimit').sum()),
        n_refractory=int((d.decision=='no_refractory').sum()),
    )
    m['energy_J'] = E_ACQ*len(d) + E_CONN*len(conn) + E_BYTE*m['bytes_total']
    # naive: upload every running acquisition, one connection each, raw spectrum
    nrun = int((d.truth_running==1).sum())
    m['energy_naive_J'] = E_ACQ*len(d) + E_CONN*nrun + E_BYTE*nrun*32772
    m['energy_saving'] = 1 - m['energy_J']/m['energy_naive_J']

    lat = []
    for on in ONSET.get(name, []):
        after = ch[ch.day >= on]
        lat.append(float(after.day.iloc[0]-on) if len(after) else np.nan)
    m['detect_latency_days'] = lat
    fp_window = d[(d.decision=='up_change')]
    if name in NOFAULT:
        m['false_positive_changes'] = len(fp_window)
    else:
        first = min(ONSET.get(name,[1e9]))
        m['false_positive_changes'] = int((fp_window.day < first-1).sum())
    return m, d

# ---------------------------------------------------------------- main
if __name__ == '__main__':
    os.makedirs('scen', exist_ok=True)
    rows, frames_cache = [], {}

    print("="*116)
    print("A. BASELINE CONFIGURATION")
    print("="*116)
    hdr = (f"{'scenario':<22}{'stAcc':>7}{'sep':>6}{'thr':>7}{'idle':>6}{'up':>5}"
           f"{'chg':>5}{'hb':>4}{'up/wk':>7}{'maxwk':>6}{'conn':>6}{'f/cn':>6}"
           f"{'KB':>7}{'lat_d':>16}{'FP':>4}{'saveE':>7}")
    print(hdr)
    for name in SCENARIOS:
        fr, meta = build(name); frames_cache[name] = (fr, meta)
        write_bin(f'scen/{name}.bin', fr, meta)
        run_sim(f'scen/{name}.bin', f'scen/{name}.csv')
        m,_ = metrics(name, f'scen/{name}.csv', meta); rows.append(m)
        print(f"{name:<22}{m['state_acc']*100:>6.1f}%{m['disc_sep']:>6.2f}"
              f"{m['disc_thr']:>7.1f}{m['n_idle']:>6}{m['n_upload']:>5}{m['n_change']:>5}"
              f"{m['n_heartbeat']:>4}{m['uploads_per_week']:>7.2f}{m['max_uploads_in_any_week']:>6}"
              f"{m['n_conn']:>6}{m['frames_per_conn']:>6.1f}{m['bytes_total']/1024:>7.1f}"
              f"{str([round(x,2) for x in m['detect_latency_days']]):>16}"
              f"{m['false_positive_changes']:>4}{m['energy_saving']*100:>6.1f}%")

    # ---------------- B. ablation: idle gating off ----------------
    print("\n"+"="*116)
    print("B. ABLATION: idle gating disabled (idle frames enter the baseline)")
    print("="*116)
    print(hdr)
    abl = cfgfile('scen/abl.txt', [(0,'idle_gating_enable',0)])
    for name in ['S2_duty_healthy','S3_duty_step_fault','S4_duty_gradual','S6_irreg_incipient']:
        run_sim(f'scen/{name}.bin', f'scen/{name}.abl.csv', abl)
        m,_ = metrics(name+'*', f'scen/{name}.abl.csv', None); m['ablation']='idle_off'; rows.append(m)
        print(f"{name+' [OFF]':<22}{m['state_acc']*100:>6.1f}%{m['disc_sep']:>6.2f}"
              f"{m['disc_thr']:>7.1f}{m['n_idle']:>6}{m['n_upload']:>5}{m['n_change']:>5}"
              f"{m['n_heartbeat']:>4}{m['uploads_per_week']:>7.2f}{m['max_uploads_in_any_week']:>6}"
              f"{m['n_conn']:>6}{m['frames_per_conn']:>6.1f}{m['bytes_total']/1024:>7.1f}"
              f"{str([round(x,2) for x in m['detect_latency_days']]):>16}"
              f"{m['false_positive_changes']:>4}{m['energy_saving']*100:>6.1f}%")

    # ---------------- C. sensitivity sweep ----------------
    print("\n"+"="*116)
    print("C. SENSITIVITY SWEEP (s_hi) on S7 incipient (blended) and S5 load-cycling")
    print("="*116)
    print(f"{'s_hi':>6}{'S7 uploads':>12}{'S7 change':>11}{'S7 latency (days)':>26}"
          f"{'S5 uploads':>12}{'S5 false pos':>14}")
    for s_hi in [2.0, 2.5, 3.0, 4.0, 5.5]:
        c = cfgfile(f'scen/s{s_hi}.txt', [(0,'s_hi',s_hi),(0,'s_lo',s_hi*0.62)])
        run_sim('scen/S7_incipient_blend.bin', 'scen/s4s.csv', c)
        m4,_ = metrics('S7_incipient_blend','scen/s4s.csv',None)
        run_sim('scen/S5_duty_loadcycle.bin','scen/s5s.csv', c)
        m5,_ = metrics('S5_duty_loadcycle','scen/s5s.csv',None)
        print(f"{s_hi:>6.1f}{m4['n_upload']:>12}{m4['n_change']:>11}"
              f"{str([round(x,2) for x in m4['detect_latency_days']]):>26}"
              f"{m5['n_upload']:>12}{m5['false_positive_changes']:>14}")

    # ---------------- D. batching sweep ----------------
    print("\n"+"="*116)
    print("D. BATCHING: connections vs uploads (S4)")
    print("="*116)
    print(f"{'batch_frames':>13}{'conn/day':>10}{'uploads':>9}{'conns':>7}"
          f"{'frames/conn':>13}{'energy J':>10}{'vs naive':>10}")
    for bf, cpd, foc in [(1,4,1),(4,2,1),(4,2,0),(6,1,0),(8,1,0),(12,1,0)]:
        c = cfgfile('scen/b.txt', [(0,'batch_max_frames',bf),(0,'conn_max_per_day',cpd),(0,'flush_on_change',foc)])
        run_sim('scen/S4_duty_gradual.bin','scen/s4b.csv', c)
        m,_ = metrics('S4','scen/s4b.csv',None)
        print(f"{str(bf)+('/prio' if foc else ''):>13}{cpd:>10}{m['n_upload']:>9}{m['n_conn']:>7}"
              f"{m['frames_per_conn']:>13.1f}{m['energy_J']:>10.0f}{m['energy_saving']*100:>9.1f}%")

    # ---------------- E. cloud control commands ----------------
    print("\n"+"="*116)
    print("E. CLOUD CONTROL COMMANDS (S4 gradual)")
    print("="*116)
    tests = {
      'baseline (no pushes)': [],
      'reset_baseline @day40': [(40*86400,'cmd_reset_baseline',1)],
      'reset_discovery @day40': [(40*86400,'cmd_reset_discovery',1)],
      'budget 14->6 @day0': [(0,'upload_max_per_window',6)],
      'budget 14->6, reserve 4': [(0,'upload_max_per_window',6),(0,'upload_reserve_change',4)],
      'heartbeat 3d->7d': [(0,'heartbeat_s',7*86400)],
      'wake 6h (kept), tighter s_hi=3': [(0,'s_hi',3.0),(0,'s_lo',1.9)],
      'idle gating OFF': [(0,'idle_gating_enable',0)],
      'force_anchor @day60': [(60*86400,'cmd_force_anchor',1)],
    }
    print(f"{'push':<34}{'up':>5}{'chg':>5}{'hb':>4}{'conn':>6}{'KB':>8}"
          f"{'ratelim':>9}{'latency (days)':>24}")
    for lbl, lines in tests.items():
        c = cfgfile('scen/ctl.txt', lines) if lines else None
        run_sim('scen/S4_duty_gradual.bin','scen/s4c.csv', c)
        m,_ = metrics('S4_duty_gradual','scen/s4c.csv',None)
        print(f"{lbl:<34}{m['n_upload']:>5}{m['n_change']:>5}{m['n_heartbeat']:>4}"
              f"{m['n_conn']:>6}{m['bytes_total']/1024:>8.1f}{m['n_ratelimited']:>9}"
              f"{str([round(x,2) for x in m['detect_latency_days']]):>24}")

    json.dump(rows, open('scen/metrics.json','w'), indent=1, default=str)
    print("\nwrote scen/metrics.json")
