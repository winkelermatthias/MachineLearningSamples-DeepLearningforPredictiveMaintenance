"""EXPERIMENT D benchmark: Fast-SC cyclic coherence + Improved Envelope
Spectrum (IES) vs the shipped envelope path, judged on low-severity faults.

Baseline (exactly the pipeline2 envelope rail, its own functions reused):
  kurtogram_band -> envelope_banded -> Welch(nperseg_for(fs)) -> order axis
  -> line SNR at the fault order.
Candidate: fast_sc coherence -> IES. Two variants, declared up front:
  PRIMARY   ies_full  - |coherence| averaged over [400 Hz, 0.94*nyq]
  secondary ies_band  - crude target-agnostic IESFOgram (top 1-of-8 slice);
            reported, but it does NOT decide the gate.

Both spectra are judged by the same line_snr_db with guard/noise windows
fixed in HERTZ (12 / 90 Hz), which reproduces the shipped 4/30-bin defaults
on the 2.93 Hz Welch grid and scales them fairly onto the ~0.25 Hz alpha
grid. Detection = line SNR > 6 dB. Speed is the known/nameplate fr for both
methods: the contest is spectrum quality, not speed estimation.

Corpus (all deterministic, seeds fixed):
  synth2  outer-race campaigns, machines with an SKF6205 DE bearing
          (BPFO 3.5848x): centrifugal_pump, induction_motor, vfd_pump.
          severities {0.05, 0.10, 0.15, 0.20} x noise_g multipliers
          {1, 4, 10} x 3 seeds, 4 s records; healthy twins (sev 0) at every
          (machine, noise, seed) for the false-line count, checked at both
          BPFO and BPFI orders.
  CWRU    smallest fault sizes only: OR007 (all three load-zone positions)
          and IR007 at all four motor loads, plus the 4 normal records
          (false lines checked at BPFO and BPFI). First 4 s of each file.

PROMOTION GATE (primary variant): median SNR gain >= 3 dB at BOTH of the
two lowest synth severities, OR >= 2 fault cases detected that the baseline
misses with ZERO new false lines on healthy records.

Run: python3 bench_fastsc.py   (writes fastsc_metrics.json next to itself)
"""
import os, sys, json, time, copy
import numpy as np
from scipy.signal import welch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))
sys.path.insert(0, HERE)

from relspec.dsp2 import kurtogram_band, envelope_banded, line_snr_db
from relspec.datasets import nperseg_for
from relspec import datasets as DS
from relspec.synth2 import FaultState2, machine_catalog2, generate2
from fastsc import fast_sc, ies_full, ies_top_band, selfcheck

SEV = [0.05, 0.10, 0.15, 0.20]
NOISE = [1.0, 4.0, 10.0]
SEEDS = [0, 1, 2]
MACHS = {'centrifugal_pump': 29.5, 'induction_motor': 24.83, 'vfd_pump': 33.0}
SECONDS = 4.0
ALPHA_MAX = 285.0
F_LO, F_HI_FRAC = 400.0, 0.94
DETECT_DB = 6.0
BPFO, BPFI = 3.5848, 5.4152          # SKF6205 orders (synth DE + CWRU DE)

def snr_orders(o, a, target, fr, guard_hz=12.0, noise_hz=90.0, rel_tol=0.015):
    """line_snr_db with windows fixed in Hz so coarse and fine axes are
    judged over the same physical neighbourhood."""
    step_hz = float(np.median(np.diff(o))) * fr
    return line_snr_db(o, a, target, rel_tol=rel_tol,
                       guard=max(2, int(round(guard_hz / step_hz))),
                       noise_w=max(8, int(round(noise_hz / step_hz))))

def baseline_snr(x, fs, fr, targets):
    """The shipped path: kurtogram band, banded envelope, Welch, orders."""
    band, _ = kurtogram_band(x, fs)
    e = envelope_banded(x, fs, band)
    nps = nperseg_for(fs)
    fe, pe = welch(e, fs=fs, nperseg=min(nps, len(e)),
                   noverlap=min(nps, len(e)) // 2, window='hann',
                   scaling='spectrum', detrend='constant')
    ae = np.sqrt(np.maximum(pe, 0))
    o = fe / fr
    return {k: snr_orders(o, ae, t, fr) for k, t in targets.items()}

def candidate_snr(x, fs, fr, targets):
    alpha, f, g = fast_sc(x, fs, alpha_max=ALPHA_MAX)
    fhi = F_HI_FRAC * fs / 2
    ef = ies_full(f, g, F_LO, fhi)
    eb, band = ies_top_band(alpha, f, g, F_LO, fhi)
    o = alpha / fr
    return ({k: snr_orders(o, ef, t, fr) for k, t in targets.items()},
            {k: snr_orders(o, eb, t, fr) for k, t in targets.items()}, band)

def run_synth():
    cat = machine_catalog2()
    rows = []
    for mi, (name, fr) in enumerate(MACHS.items()):
        for ni, mult in enumerate(NOISE):
            spec = copy.deepcopy(cat[name])
            spec.noise_g *= mult
            for si, seed in enumerate(SEEDS):
                for vi, sev in enumerate([0.0] + SEV):
                    rng = np.random.default_rng(40000 + 1000*mi + 100*ni
                                                + 10*si + vi)
                    x = generate2(spec, FaultState2(outer_race=sev), SECONDS,
                                  fr, load=0.75, speed_wander_pct=0.3,
                                  rng=rng).astype(np.float64)
                    targets = ({'BPFO': BPFO} if sev > 0 else
                               {'BPFO': BPFO, 'BPFI': BPFI})
                    b = baseline_snr(x, spec.fs, fr, targets)
                    cf, cb, band = candidate_snr(x, spec.fs, fr, targets)
                    for k in targets:
                        rows.append(dict(corpus='synth', machine=name,
                                         sev=sev, noise=mult, seed=seed,
                                         line=k, base=b[k], ies=cf[k],
                                         ies_band=cb[k], band=list(band)))
    return rows

def run_cwru():
    if not os.path.isdir(DS.CWRU_ROOT):
        return None
    rows = []
    for r in DS.cwru_records():
        st = r['state']
        if st == 'normal':
            targets = {'BPFO': r['lines']['BPFO'], 'BPFI': r['lines']['BPFI']}
        elif st.startswith('OR007'):
            targets = {'BPFO': r['lines']['BPFO']}
        elif st.startswith('IR007'):
            targets = {'BPFI': r['lines']['BPFI']}
        else:
            continue
        fs, fr = r['fs'], r['fr_nominal']
        n = int(SECONDS * fs)
        if len(r['x']) < int(2 * fs):
            continue
        x = r['x'][:min(n, len(r['x']))]
        b = baseline_snr(x, fs, fr, targets)
        cf, cb, band = candidate_snr(x, fs, fr, targets)
        for k in targets:
            rows.append(dict(corpus='cwru', state=st, file=r['name'],
                             group=r['group'], sev=(0.0 if st == 'normal'
                                                    else -1.0),
                             line=k, base=b[k], ies=cf[k], ies_band=cb[k],
                             band=list(band)))
    return rows

def med(v):
    v = [x for x in v if np.isfinite(x)]
    return float(np.median(v)) if v else float('nan')

def main():
    t0 = time.time()
    print('== 0. selfcheck: known-alpha cyclostationary signal ==')
    sc = selfcheck()
    err = abs(sc['alpha_hat'] - sc['alpha_true'])
    ok = err <= sc['da'] and sc['peak'] > 2 * sc['floor_med'] \
        and sc['stationary_max'] < 2 * sc['stationary_med']
    print(f"  alpha_hat {sc['alpha_hat']:.2f} Hz (true {sc['alpha_true']}, "
          f"da {sc['da']:.3f})  peak {sc['peak']:.3f} vs floor "
          f"{sc['floor_med']:.3f}; stationary max/med "
          f"{sc['stationary_max']:.3f}/{sc['stationary_med']:.3f}  "
          f"-> {'OK' if ok else 'FAIL'}")
    assert ok, 'Fast-SC selfcheck failed; benchmark results would be void'

    print('\n== 1. synth2 outer-race, low severity ==')
    rows = run_synth()
    fault = [r for r in rows if r['corpus'] == 'synth' and r['sev'] > 0]
    healthy = [r for r in rows if r['corpus'] == 'synth' and r['sev'] == 0]
    print('  sev   noise |  base   ies   gain | ies_band | det b/i/ib (of 9)')
    for sev in SEV:
        for mult in NOISE:
            g = [r for r in fault if r['sev'] == sev and r['noise'] == mult]
            db = sum(r['base'] > DETECT_DB for r in g)
            di = sum(r['ies'] > DETECT_DB for r in g)
            dib = sum(r['ies_band'] > DETECT_DB for r in g)
            print(f"  {sev:.2f}  x{mult:4.0f} | {med([r['base'] for r in g]):5.1f} "
                  f"{med([r['ies'] for r in g]):5.1f} "
                  f"{med([r['ies'] - r['base'] for r in g]):+5.1f} | "
                  f"{med([r['ies_band'] for r in g]):7.1f} |  {db}/{di}/{dib}")

    print('\n== 2. CWRU smallest faults (007) + normals ==')
    crows = run_cwru()
    if crows is None:
        print('  CWRU_ROOT absent on disk; real-data leg skipped')
        crows = []
    else:
        for st in sorted({r['state'] for r in crows}):
            g = [r for r in crows if r['state'] == st]
            db = sum(r['base'] > DETECT_DB for r in g)
            di = sum(r['ies'] > DETECT_DB for r in g)
            print(f"  {st:<10} n={len(g):2d} | base {med([r['base'] for r in g]):5.1f} "
                  f"ies {med([r['ies'] for r in g]):5.1f} "
                  f"gain {med([r['ies'] - r['base'] for r in g]):+5.1f} | "
                  f"band {med([r['ies_band'] for r in g]):5.1f} | det {db}/{di}")

    # ------------------------------------------------------------- gate
    all_fault = fault + [r for r in crows if r['sev'] < 0]
    all_healthy = healthy + [r for r in crows if r['sev'] == 0]
    gains = {s: med([r['ies'] - r['base'] for r in fault if r['sev'] == s])
             for s in SEV[:2]}
    g1 = all(np.isfinite(gains[s]) and gains[s] >= 3.0 for s in SEV[:2])
    extra = [r for r in all_fault
             if r['ies'] > DETECT_DB >= r['base']]
    lost = [r for r in all_fault if r['base'] > DETECT_DB >= r['ies']]
    fl_base = sum(r['base'] > DETECT_DB for r in all_healthy)
    fl_ies = sum(r['ies'] > DETECT_DB for r in all_healthy)
    new_false = sum(r['ies'] > DETECT_DB >= r['base'] for r in all_healthy)
    g2 = len(extra) >= 2 and new_false == 0
    proved = g1 or g2
    # threshold robustness: the healthy margin is what a second opinion
    # lives on, so show the extra-detect count survives a stricter line
    robust = {str(t): dict(
        extra=sum(r['ies'] > t >= r['base'] for r in all_fault),
        healthy_false=sum(r['ies'] > t for r in all_healthy))
        for t in (6.0, 8.0, 10.0)}
    # secondary variant, reported only
    extra_b = sum(r['ies_band'] > DETECT_DB >= r['base'] for r in all_fault)
    newf_b = sum(r['ies_band'] > DETECT_DB >= r['base'] for r in all_healthy)
    gains_b = {s: med([r['ies_band'] - r['base'] for r in fault
                       if r['sev'] == s]) for s in SEV[:2]}

    print('\n== 3. gate (primary = full-band IES) ==')
    print(f"  median gain @sev0.05 {gains[0.05]:+.1f} dB, "
          f"@sev0.10 {gains[0.10]:+.1f} dB (need both >= +3)  -> {g1}")
    print(f"  extra detects {len(extra)}, lost detects {len(lost)}, "
          f"healthy false lines base/ies {fl_base}/{fl_ies} "
          f"(new: {new_false})  -> {g2}")
    print(f"  secondary ies_band: gains {gains_b[0.05]:+.1f}/"
          f"{gains_b[0.10]:+.1f} dB, extra {extra_b}, new false {newf_b}")
    print('  robustness: ' + ', '.join(
        f"thr {t}: extra {v['extra']} / healthy false {v['healthy_false']}"
        for t, v in robust.items()))
    print(f"  PROVED: {proved}")

    out = dict(selfcheck=sc,
               params=dict(sev=SEV, noise=NOISE, seeds=SEEDS,
                           machines=list(MACHS), seconds=SECONDS,
                           alpha_max=ALPHA_MAX, f_lo=F_LO,
                           f_hi_frac=F_HI_FRAC, detect_db=DETECT_DB,
                           nw=256, hop=32, guard_hz=12.0, noise_hz=90.0),
               synth_rows=rows, cwru_rows=crows,
               gate=dict(gain_sev005=gains[0.05], gain_sev010=gains[0.10],
                         g1_median_gain=g1, extra_detects=len(extra),
                         lost_detects=len(lost),
                         false_lines_base=fl_base, false_lines_ies=fl_ies,
                         new_false_lines=new_false, g2_extra_detects=g2,
                         proved=proved, threshold_robustness=robust,
                         band_variant=dict(gain_sev005=gains_b[0.05],
                                           gain_sev010=gains_b[0.10],
                                           extra_detects=extra_b,
                                           new_false_lines=newf_b)),
               runtime_s=time.time() - t0)
    with open(os.path.join(HERE, 'fastsc_metrics.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nwrote fastsc_metrics.json  ({out['runtime_s']:.0f}s)")

if __name__ == '__main__':
    main()
