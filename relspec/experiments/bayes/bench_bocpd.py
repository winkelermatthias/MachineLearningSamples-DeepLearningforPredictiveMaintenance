"""EXPERIMENT C benchmark: BOCPD vs the shipped v1 Gate decision layer.

Both detectors consume the SAME evidence: the Gate's scalar score sequence
(pm + mask + CUSUM-drift terms, exactly what Gate.decide thresholds), taken
from a default-configuration Gate replay so the scoring front-end is the
shipped one. The contest is purely the decision layer:

  baseline  the ACTUAL v1 Gate (imported): threshold + 2-of-3 persistence
            votes + hysteresis + floor gate, alarm = decision 'up_change'.
            Sensitivity knob swept: s_hi (s_lo tracks at s_hi-1.5).
  candidate BOCPD (bocpd.py) over the same score stream: alarm =
            P(changepoint within last w frames) >= p*, p* swept.

Corpus: 36 campaigns x 80 half-day acquisitions from synth2 through the real
extract2 (seeds 1xxx; a disjoint 16-campaign dev slice, seeds 5xxx, picks
BOCPD's hazard/w so the eval set never tunes the candidate's extra knobs):

  no-fault x12   false-alarm set - defines each detector's operating point
  abrupt   x8    severity steps 0 -> sev_max in one acquisition
  ramp     x8    severity grows linear/exponential from onset, reaching only
                 0.45-0.6 by end of campaign - the CUSUM's home turf
  nuisance x8    healthy, but the speed setpoint jumps 8-12% mid-campaign
                 and wander x2.5 - alarms here are reported separately

Latency is counted in acquisitions (2/day); a miss is censored at the
remaining campaign length so a detector cannot buy latency by not detecting.
Detection = first alarm at or after onset; pre-onset alarms on fault
campaigns are tallied as early alarms.

PROMOTION GATE: at each detector's best zero-FA operating point, BOCPD mean
(censored) latency <= Gate's, AND BOCPD reliability bins monotone (higher
P(change) bin -> higher precision, bins with n >= 25).

Run: python3 bench_bocpd.py   (writes bocpd_metrics.json next to itself)
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))
sys.path.insert(0, HERE)

from relspec.pipeline import Gate
from relspec.pipeline2 import extract2
from relspec.synth2 import FaultState2, machine_catalog2, generate2, progression
from bocpd import BOCPD, fit_prior

T_STEPS, WIN_S, WARMUP = 80, 2.0, 20          # Gate.warmup_n = 20
STEP_S = 43200                                # 2 acquisitions/day, fleet2 clock

NOMINAL_FR = dict(centrifugal_pump=29.5, induction_motor=24.83, gearbox=16.2,
                  centrifugal_fan=12.4, screw_compressor=48.6, belt_fan=18.7,
                  vfd_pump=33.0, recip_compressor=12.15, slow_mixer=6.4)
MACHS = list(NOMINAL_FR)
ABRUPT = [('centrifugal_pump', 'outer_race'), ('induction_motor', 'inner_race'),
          ('gearbox', 'gear_wear'), ('centrifugal_fan', 'imbalance'),
          ('screw_compressor', 'rolling_element'), ('belt_fan', 'belt_wear'),
          ('vfd_pump', 'rotor_bar'), ('recip_compressor', 'looseness')]
RAMP = [('centrifugal_pump', 'cavitation'), ('induction_motor', 'misalignment'),
        ('gearbox', 'outer_race'), ('centrifugal_fan', 'looseness'),
        ('screw_compressor', 'lubrication'), ('belt_fan', 'imbalance'),
        ('vfd_pump', 'outer_race'), ('slow_mixer', 'gear_wear')]

GATE_SHI = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0]
BOCPD_THR = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
DEV_GRID = [(h, w) for h in (1/100., 1/300., 1/1000.) for w in (4, 8, 12)]


# ------------------------------------------------------------------ corpus
def build_campaigns(seed_base, n_nofault, n_abrupt, n_ramp, n_nuis, tag):
    rng = np.random.default_rng(seed_base)
    camps, i = [], 0

    def base(kind, mtype):
        nonlocal i
        c = dict(cid=f'{tag}{i:02d}', kind=kind, mtype=mtype,
                 fr=NOMINAL_FR[mtype]*(1+rng.uniform(-0.03, 0.03)),
                 wander=float(rng.uniform(0.2, 1.2)),
                 onset=int(rng.integers(36, 51)),
                 seed=int(seed_base+i)); i += 1
        return c

    for j in range(n_nofault):
        camps.append(base('nofault', MACHS[j % len(MACHS)]))
    for j in range(n_abrupt):
        m, flt = ABRUPT[j % len(ABRUPT)]
        c = base('abrupt', m)
        c.update(fault=flt, sev_max=float(rng.uniform(0.5, 0.75)))
        camps.append(c)
    for j in range(n_ramp):
        m, flt = RAMP[j % len(RAMP)]
        c = base('ramp', m)
        c.update(fault=flt, sev_max=float(rng.uniform(0.45, 0.6)),
                 prog='linear' if j % 2 == 0 else 'exponential')
        camps.append(c)
    for j in range(n_nuis):
        c = base('nuisance', MACHS[(j*2+1) % len(MACHS)])
        c.update(jump=float(rng.choice([-1, 1])*rng.uniform(0.08, 0.12)))
        camps.append(c)
    return camps


def run_campaign(c):
    """Generate + extract one campaign; return only what Gate.decide reads."""
    spec = machine_catalog2()[c['mtype']]
    rng = np.random.default_rng(c['seed'])
    setpoint, steps = c['fr'], []
    for t in range(T_STEPS):
        if c['kind'] == 'nuisance' and t == c['onset']:
            setpoint = c['fr']*(1+c['jump'])
        wander = c['wander']*(2.5 if c['kind'] == 'nuisance' and t >= c['onset'] else 1.0)
        day = t/2.0
        load = float(np.clip(0.6+0.35*np.sin(2*np.pi*day/7.0)**2
                             + rng.uniform(-0.05, 0.05), 0.3, 1.0))
        fr_true = setpoint*(1+rng.uniform(-0.015, 0.015))
        f = FaultState2()
        if c['kind'] == 'abrupt' and t >= c['onset']:
            setattr(f, c['fault'], c['sev_max'])
        elif c['kind'] == 'ramp' and t >= c['onset']:
            u = (t-c['onset'])/max(T_STEPS-1-c['onset'], 1)
            setattr(f, c['fault'], c['sev_max']*progression(c['prog'], u))
        x = generate2(spec, f, WIN_S, fr_true, load=load,
                      speed_wander_pct=wander, rng=rng)
        e = extract2(x.astype(np.float64), spec.fs, band_key=c['cid'],
                     fr_nominal=c['fr'])
        steps.append(dict(acc_rms=e.acc_rms, vel_rms=e.vel_rms,
                          acc_kurt=e.acc_kurt, acc_crest=e.acc_crest,
                          env_rms=e.env_rms, env_kurt=e.env_kurt,
                          env_crest=e.env_crest,
                          band_db=e.band_db.astype(np.float64),
                          acc_u8=e.acc_u8.copy()))
    return dict(c, steps=steps)


# ---------------------------------------------------------------- replay
class _Lite:
    """Attribute bag with exactly the fields Gate.init/decide read. Faithful
    replay: Gate touches nothing else of Extract (verified against pipeline.py;
    bin_med/bin_mad are updated from acc_u8 but never enter score/decide)."""
    __slots__ = ('acc_rms', 'vel_rms', 'acc_kurt', 'acc_crest',
                 'env_rms', 'env_kurt', 'env_crest', 'band_db', 'acc_u8')

    def __init__(self, d):
        for k in self.__slots__: setattr(self, k, d[k])


def replay_gate(steps, s_hi, s_lo):
    """Run the ACTUAL v1 Gate over a campaign. Returns (alarm steps, score
    trace, floor_ok trace). Alarm = decision 'up_change', the same event
    fleet2 counts."""
    g = Gate(s_hi=s_hi, s_lo=s_lo)
    alarms, scores, floors = [], [np.nan], [False]  # step 0 seeds the baseline
    for t, st in enumerate(steps):
        e = _Lite(st)
        if t == 0:
            g.init(e); continue
        s = g.decide(e, t*STEP_S)
        scores.append(s['score']); floors.append(bool(s['floor_ok']))
        if s['decision'] == 'up_change':
            alarms.append(t)
    return alarms, np.array(scores), floors


def run_bocpd(scores, hazard, w):
    """BOCPD over the default-Gate score stream. Prior fit on the warmup
    slice (the Gate gets the same 20 frames of grace); decisions only legal
    once the run-length support clears w, mirrored in the returned mask."""
    warm = scores[1:WARMUP+1]
    mu0, beta0 = fit_prior(warm)
    det = BOCPD(mu0, beta0, hazard=hazard, w=w)
    p = np.zeros(len(scores))
    for t in range(WARMUP+1, len(scores)):
        p[t] = det.step(scores[t])['p_recent']
    legal_from = WARMUP+1+w
    p[:legal_from] = 0.0
    return p, legal_from


# ---------------------------------------------------------------- scoring
def eval_operating(camps, alarms_by_cid):
    """FA / latency bookkeeping at one threshold. Latency in acquisitions,
    misses censored at remaining campaign length."""
    fa_nofault = fa_nuis = early = detected = 0
    lat_cens, lat_det, kind_lat = [], [], {'abrupt': [], 'ramp': []}
    for c in camps:
        al = alarms_by_cid[c['cid']]
        if c['kind'] == 'nofault':
            fa_nofault += bool(al)
        elif c['kind'] == 'nuisance':
            fa_nuis += bool(al)
        else:
            early += any(a < c['onset'] for a in al)
            post = [a for a in al if a >= c['onset']]
            cens = T_STEPS-c['onset']
            if post:
                lat = post[0]-c['onset']
                detected += 1; lat_det.append(lat)
                lat_cens.append(lat); kind_lat[c['kind']].append(lat)
            else:
                lat_cens.append(cens); kind_lat[c['kind']].append(cens)
    n_fault = sum(c['kind'] in ('abrupt', 'ramp') for c in camps)
    return dict(fa_nofault=fa_nofault, fa_nuisance=fa_nuis, early=early,
                detected=detected, n_fault=n_fault,
                mean_lat_censored=float(np.mean(lat_cens)),
                mean_lat_detected=float(np.mean(lat_det)) if lat_det else None,
                mean_lat_abrupt=float(np.mean(kind_lat['abrupt'])),
                mean_lat_ramp=float(np.mean(kind_lat['ramp'])))


def pick_op(sweep, fa_budget):
    """Best (lowest censored latency) threshold within the FA budget on the
    no-fault set; None if no threshold qualifies."""
    ok = [r for r in sweep if r['fa_nofault'] <= fa_budget]
    return min(ok, key=lambda r: r['mean_lat_censored']) if ok else None


def reliability(camps, p_by_cid, legal_from):
    """Reliability bins for P(recent change): per bin, precision = fraction
    of frames where a fault is actually present (fault campaign, t>=onset).
    Nuisance/no-fault frames count as negatives - a nuisance speed step is
    exactly the change the detector must NOT believe in."""
    edges = np.linspace(0, 1, 11)
    cnt = np.zeros(10, int); pos = np.zeros(10, int)
    for c in camps:
        p = p_by_cid[c['cid']]
        for t in range(legal_from, T_STEPS):
            y = c['kind'] in ('abrupt', 'ramp') and t >= c['onset']
            b = min(int(p[t]*10), 9)
            cnt[b] += 1; pos[b] += int(y)
    prec = [float(pos[b]/cnt[b]) if cnt[b] else None for b in range(10)]
    occ = [(b, prec[b]) for b in range(10) if cnt[b] >= 25]
    monotone = all(occ[i+1][1] >= occ[i][1]-1e-9 for i in range(len(occ)-1))
    return dict(bin_edges=edges.tolist(), count=cnt.tolist(), pos=pos.tolist(),
                precision=prec, monotone=bool(monotone),
                bins_checked=[b for b, _ in occ])


# ------------------------------------------------------------------ main
def extract_all(camps, tag):
    t0 = time.time()
    try:
        from multiprocessing import Pool
        with Pool(4) as pool:
            out = pool.map(run_campaign, camps)
    except Exception:
        out = [run_campaign(c) for c in camps]
    print(f'  {tag}: {len(out)} campaigns x {T_STEPS} steps extracted '
          f'in {time.time()-t0:.0f}s')
    return out


def sweep_detectors(camps, hazard, w):
    """Default-Gate score streams once; then both threshold sweeps. Also a
    DIAGNOSTIC hybrid: BOCPD probability additionally gated by the Gate's
    floor_ok physical-delta guard, to isolate how much of the Gate's FA
    immunity is that guard rather than its persistence logic."""
    scores, floors = {}, {}; t_bocpd = 0.0; n_bocpd = 0
    for c in camps:
        _, sc, fl = replay_gate(c['steps'], 4.0, 2.5)  # shipped config front-end
        scores[c['cid']], floors[c['cid']] = sc, fl
    gate_sweep = []
    for s_hi in GATE_SHI:
        al = {c['cid']: replay_gate(c['steps'], s_hi, max(s_hi-1.5, 1.0))[0]
              for c in camps}
        gate_sweep.append(dict(s_hi=s_hi, **eval_operating(camps, al)))
    p_by_cid = {}; legal_from = WARMUP+1+w
    for c in camps:
        t0 = time.time()
        p_by_cid[c['cid']], legal_from = run_bocpd(scores[c['cid']], hazard, w)
        t_bocpd += time.time()-t0; n_bocpd += T_STEPS-WARMUP-1
    bocpd_sweep, floor_sweep = [], []
    for thr in BOCPD_THR:
        al = {cid: list(np.where(p >= thr)[0]) for cid, p in p_by_cid.items()}
        bocpd_sweep.append(dict(thr=thr, **eval_operating(camps, al)))
        alf = {cid: [t for t in np.where(p >= thr)[0] if floors[cid][t]]
               for cid, p in p_by_cid.items()}
        floor_sweep.append(dict(thr=thr, **eval_operating(camps, alf)))
    return (gate_sweep, bocpd_sweep, floor_sweep, p_by_cid, legal_from,
            1e6*t_bocpd/max(n_bocpd, 1))


def main():
    print('== EXPERIMENT C: BOCPD vs v1 Gate decision layer ==')
    # ---- dev slice picks BOCPD's (hazard, w); eval never tunes them
    dev = build_campaigns(5000, 6, 4, 4, 2, 'd')
    dev = extract_all(dev, 'dev')
    best, dev_rows = None, []
    for hz, w in DEV_GRID:
        _, bs, _, _, _, _ = sweep_detectors(dev, hz, w)
        op = pick_op(bs, 0)
        lat = op['mean_lat_censored'] if op else float('inf')
        dev_rows.append(dict(hazard=hz, w=w, zero_fa_lat=None if op is None else lat,
                             thr=None if op is None else op['thr']))
        if best is None or lat < best[2]:
            best = (hz, w, lat)
    hazard, w = best[0], best[1]
    print(f'  dev pick: hazard=1/{round(1/hazard)}, w={w} '
          f'(zero-FA censored latency {best[2]:.2f})')

    # ---- main corpus
    camps = build_campaigns(1000, 12, 8, 8, 8, 'm')
    camps = extract_all(camps, 'main')
    gate_sweep, bocpd_sweep, floor_sweep, p_by_cid, legal_from, us_step = \
        sweep_detectors(camps, hazard, w)

    print('\n  Gate sweep (s_hi | FA_nofault FA_nuis early det/n lat_cens lat_abrupt lat_ramp):')
    for r in gate_sweep:
        print(f"    {r['s_hi']:5.1f} | {r['fa_nofault']:2d} {r['fa_nuisance']:2d} "
              f"{r['early']:2d}  {r['detected']:2d}/{r['n_fault']}  "
              f"{r['mean_lat_censored']:6.2f}  {r['mean_lat_abrupt']:6.2f}  "
              f"{r['mean_lat_ramp']:6.2f}")
    print('  BOCPD sweep (p*  | same columns):')
    for r in bocpd_sweep:
        print(f"    {r['thr']:5.2f} | {r['fa_nofault']:2d} {r['fa_nuisance']:2d} "
              f"{r['early']:2d}  {r['detected']:2d}/{r['n_fault']}  "
              f"{r['mean_lat_censored']:6.2f}  {r['mean_lat_abrupt']:6.2f}  "
              f"{r['mean_lat_ramp']:6.2f}")

    print('  BOCPD+floor_ok diagnostic sweep (p* | same columns):')
    for r in floor_sweep:
        print(f"    {r['thr']:5.2f} | {r['fa_nofault']:2d} {r['fa_nuisance']:2d} "
              f"{r['early']:2d}  {r['detected']:2d}/{r['n_fault']}  "
              f"{r['mean_lat_censored']:6.2f}  {r['mean_lat_abrupt']:6.2f}  "
              f"{r['mean_lat_ramp']:6.2f}")

    ops = {}
    for fa in (0, 1):
        g, b = pick_op(gate_sweep, fa), pick_op(bocpd_sweep, fa)
        fb = pick_op(floor_sweep, fa)
        ops[f'fa{fa}'] = dict(gate=g, bocpd=b, bocpd_floor=fb)
        gl = 'none' if g is None else f"s_hi={g['s_hi']} lat={g['mean_lat_censored']:.2f}"
        bl = 'none' if b is None else f"p*={b['thr']} lat={b['mean_lat_censored']:.2f}"
        fl = 'none' if fb is None else f"p*={fb['thr']} lat={fb['mean_lat_censored']:.2f}"
        print(f'  operating point FA<={fa}: gate {gl} | bocpd {bl} | bocpd+floor {fl}')

    cal = reliability(camps, p_by_cid, legal_from)
    print('\n  reliability (bin: n, precision):')
    for bb in range(10):
        if cal['count'][bb]:
            print(f"    [{bb/10:.1f},{(bb+1)/10:.1f}) n={cal['count'][bb]:4d} "
                  f"prec={cal['precision'][bb]:.3f}")
    print(f"  monotone over bins n>=25: {cal['monotone']}  "
          f"(bins {cal['bins_checked']})")

    g0, b0 = ops['fa0']['gate'], ops['fa0']['bocpd']
    lat_ok = (g0 is not None and b0 is not None and
              b0['mean_lat_censored'] <= g0['mean_lat_censored']+1e-9)
    proved = bool(lat_ok and cal['monotone'])
    print(f'\n  PROMOTION: latency_ok={lat_ok} calibration_monotone='
          f'{cal["monotone"]} -> proved={proved}')

    out = dict(config=dict(T=T_STEPS, warmup=WARMUP, win_s=WIN_S,
                           hazard=hazard, w=w, gate_shi=GATE_SHI,
                           bocpd_thr=BOCPD_THR, seeds=dict(dev=5000, main=1000),
                           corpus=dict(nofault=12, abrupt=8, ramp=8, nuisance=8),
                           bocpd_us_per_step=us_step),
               dev_tuning=dev_rows,
               gate_sweep=gate_sweep, bocpd_sweep=bocpd_sweep,
               bocpd_floor_sweep=floor_sweep,
               operating_points=ops, calibration=cal, proved=proved)
    with open(os.path.join(HERE, 'bocpd_metrics.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print('  wrote bocpd_metrics.json')
    return out


if __name__ == '__main__':
    main()
