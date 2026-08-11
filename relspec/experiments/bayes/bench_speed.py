"""EXPERIMENT B benchmark: Bayesian speed posterior vs estimate_speed2.

Corpus (fixed seeds 1xxx; the tuning dev slice used 5xxx and is disjoint):

  MAIN        9 catalog machines x 8 variants = 72 acquisitions: healthy at
              low/high wander, low-SNR (noise x6), machine-appropriate
              faults at three severities, a ramp profile, and an +8%
              off-nominal setpoint. No outer_race here on purpose - that
              failure mode is quarantined in the adversarial set.
  ADVERSARIAL 3 machines carrying an SKF6205 (BPFO 3.5848x) x severe
              outer_race x {noise x1, x3} = 12 acquisitions. The envelope
              comb is BPFO-spaced and BPFO/3 = 1.1949x sits inside the
              search span: the documented case where the envelope HPS is
              confidently wrong. The bench VERIFIES the baseline fails.

Ground truth is the fr handed to generate2 (the profile is mean-centred, so
the record-average speed equals it). Baselines: estimate_speed2 (the
envelope rail, the stated comparison) and extract2's fused fr (the shipped
two-rail gate, reported for honesty - the envelope rail alone is blind on
healthy machines and loses badly; the fused estimator is the real bar).

PROMOTION GATE (combined corpus, bayes vs estimate_speed2):
  p95 error improved >= 25%; median not worse than baseline +10%;
  adversarial gross (>5%) strictly fewer; runtime < 3x per acquisition.

Run: python3 bench_speed.py   (writes speed_metrics.json next to itself)
"""
import sys, os, json, time
from dataclasses import replace
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))
sys.path.insert(0, HERE)

# The posterior was promoted into the shipped pipeline (pipeline2.speed3
# section): this bench now measures the INTEGRATED path, and 'extract2
# fused' below exercises the wired-in flag (BAYES_SPEED default on) with
# its guards. speed_bayes.py remains as the experiment record.
from relspec.pipeline2 import (estimate_speed2, extract2, band_for,
                               estimate_speed3 as speed_bayes,
                               KS, K_PRIOR, SIGMA_CAL)
from relspec.synth2 import FaultState2, machine_catalog2, generate2

SECONDS = 4.0
NOMINAL_FR = dict(centrifugal_pump=29.5, induction_motor=24.83, gearbox=16.2,
                  centrifugal_fan=12.4, screw_compressor=48.6, belt_fan=18.7,
                  vfd_pump=33.0, recip_compressor=12.15, slow_mixer=6.4)
# machine-appropriate non-bearing-outer fault for the main corpus
MAIN_FAULT = dict(centrifugal_pump='inner_race', induction_motor='misalignment',
                  gearbox='gear_wear', centrifugal_fan='imbalance',
                  screw_compressor='rolling_element', belt_fan='belt_wear',
                  vfd_pump='rotor_bar', recip_compressor='looseness',
                  slow_mixer='gear_wear')
ADV_MACHINES = ('centrifugal_pump', 'induction_motor', 'vfd_pump')


def build_corpus():
    """(case_id, subset, mtype, fr_nominal, fault dict, noise_mult, wander,
    profile, fr_off, seed) - everything a case needs, all deterministic."""
    cases, i = [], 0
    for mtype, fn in NOMINAL_FR.items():
        fld = MAIN_FAULT[mtype]
        for tag, fault, nm, wander, prof, off in (
                ('healthy',      {},          1, 0.3, 'steady', None),
                ('healthy_wndr', {},          1, 1.2, 'steady', None),
                ('healthy_snr',  {},          6, 0.8, 'steady', None),
                ('fault_mild',   {fld: 0.3},  1, 0.5, 'steady', None),
                ('fault_mod',    {fld: 0.6},  1, 1.0, 'steady', None),
                ('fault_snr',    {fld: 0.5},  5, 0.6, 'steady', None),
                ('healthy_ramp', {},          1, 0.5, 'ramp',   None),
                ('fault_offnom', {fld: 0.4},  1, 0.4, 'steady', 0.08)):
            cases.append((f'M{i:02d}_{mtype}_{tag}', 'main', mtype, fn,
                          fault, nm, wander, prof, off, 1000+i))
            i += 1
    for mtype in ADV_MACHINES:
        for sev in (0.6, 0.9):
            for nm in (1, 3):
                cases.append((f'A{i:02d}_{mtype}_or{sev}_n{nm}', 'adversarial',
                              mtype, NOMINAL_FR[mtype], {'outer_race': sev},
                              nm, 0.4, 'steady', None, 1000+i))
                i += 1
    return cases


def main():
    cat = machine_catalog2()
    cases = build_corpus()
    rows = []
    t_base = t_bayes = t_fused = 0.0
    for cid, subset, mtype, fn, fault, nm, wander, prof, off, seed in cases:
        spec = cat[mtype]
        sp = replace(spec, noise_g=spec.noise_g*nm) if nm != 1 else spec
        rng = np.random.default_rng(seed)
        fr_true = fn*(1+off) if off is not None else \
            fn*(1+rng.uniform(-0.02, 0.02))
        x = generate2(sp, FaultState2(**fault), SECONDS, fr_true,
                      speed_wander_pct=wander, profile=prof,
                      rng=rng).astype(np.float64)
        band = band_for(cid, x, sp.fs)     # shared by every estimator

        t0 = time.perf_counter()
        fr_b, conf_b = estimate_speed2(x, sp.fs, band, fn)
        t1 = time.perf_counter()
        fr_p, sig, k_map, w = speed_bayes(x, sp.fs, band, fn)
        t2 = time.perf_counter()
        e = extract2(x, sp.fs, band_key=cid, fr_nominal=fn)
        t3 = time.perf_counter()
        t_base += t1-t0; t_bayes += t2-t1; t_fused += t3-t2

        rows.append(dict(
            case=cid, subset=subset, mtype=mtype, fr_true=fr_true,
            fr_base=fr_b, conf_base=conf_b, fr_bayes=fr_p, sigma=sig,
            k_map=k_map, w_map=float(np.max(w)), fr_fused=e.fr, tier=e.tier,
            err_base=100*abs(fr_b-fr_true)/fr_true,
            err_bayes=100*abs(fr_p-fr_true)/fr_true,
            err_fused=100*abs(e.fr-fr_true)/fr_true,
            covered=bool(abs(fr_p-fr_true) <= 2*sig)))

    def agg(key, sel=None):
        e = np.array([r[key] for r in rows if sel is None or r['subset'] == sel])
        return dict(median=float(np.median(e)), p95=float(np.percentile(e, 95)),
                    gross=int(np.sum(e > 5.0)), n=len(e))
    A = {m: {'combined': agg(f'err_{m}'), 'main': agg(f'err_{m}', 'main'),
             'adversarial': agg(f'err_{m}', 'adversarial')}
         for m in ('base', 'bayes', 'fused')}
    cov = float(np.mean([r['covered'] for r in rows]))
    n = len(rows)
    rt = dict(base_ms=1e3*t_base/n, bayes_ms=1e3*t_bayes/n,
              fused_ms=1e3*t_fused/n, ratio_bayes_vs_base=t_bayes/t_base)

    cb, cp = A['base']['combined'], A['bayes']['combined']
    ab, ap = A['base']['adversarial'], A['bayes']['adversarial']
    gate = dict(
        p95_improvement_pct=100*(cb['p95']-cp['p95'])/cb['p95'],
        p95_ok=bool(cp['p95'] <= 0.75*cb['p95']),
        median_ok=bool(cp['median'] <= 1.10*cb['median']),
        adv_gross_base=ab['gross'], adv_gross_bayes=ap['gross'],
        adv_gross_ok=bool(ap['gross'] < ab['gross']),
        baseline_fails_adversarial=bool(ab['gross'] > 0),
        runtime_ratio=rt['ratio_bayes_vs_base'],
        runtime_ok=bool(rt['ratio_bayes_vs_base'] < 3.0))
    gate['proved'] = bool(gate['p95_ok'] and gate['median_ok'] and
                          gate['adv_gross_ok'] and gate['runtime_ok'])

    print(f'corpus: {n} acquisitions '
          f'({A["base"]["main"]["n"]} main + {ab["n"]} adversarial)\n')
    print(f'{"":16s}{"median%":>9s}{"p95%":>9s}{"gross>5%":>10s}')
    for m, lbl in (('base', 'estimate_speed2'), ('bayes', 'bayes posterior'),
                   ('fused', 'extract2 fused')):
        c = A[m]['combined']; a = A[m]['adversarial']
        print(f'{lbl:16s}{c["median"]:9.3f}{c["p95"]:9.3f}'
              f'{c["gross"]:7d} ({a["gross"]} adv)')
    print(f'\nbaseline fails adversarial: {ab["gross"]}/{ab["n"]} gross '
          f'(median adv err {ab["median"]:.1f}%) - failure mode verified')
    print(f'+/-2 sigma empirical coverage: {100*cov:.0f}%')
    print(f'runtime/acq: base {rt["base_ms"]:.1f}ms  bayes {rt["bayes_ms"]:.1f}ms '
          f'({rt["ratio_bayes_vs_base"]:.2f}x)  extract2 {rt["fused_ms"]:.1f}ms')
    print('\nGATE:', json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                 for k, v in gate.items()}))

    out = dict(config=dict(seconds=SECONDS, seeds='1000+', KS=list(KS),
                           K_PRIOR=list(K_PRIOR), sigma_cal=SIGMA_CAL,
                           impl='relspec.pipeline2.estimate_speed3 (integrated)',
                           note='dev/tuning slice used seeds 5xxx, disjoint'),
               aggregates=A, coverage_2sigma=cov, runtime=rt, gate=gate,
               rows=rows)
    with open(os.path.join(HERE, 'speed_metrics.json'), 'w') as fo:
        json.dump(out, fo, indent=1, default=float)
    print(f'\nwrote speed_metrics.json  proved={gate["proved"]}')


if __name__ == '__main__':
    main()
