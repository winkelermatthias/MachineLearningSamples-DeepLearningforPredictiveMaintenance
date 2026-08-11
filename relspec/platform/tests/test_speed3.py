"""Adversarial speed regression: the BPFO/3 lock.

On a severe outer-race fault the envelope comb is BPFO-spaced (3.58x shaft
for the catalog's SKF6205), BPFO/3 = 1.19x lands inside the +/-35% search
span, and the envelope HPS estimator is CONFIDENTLY wrong - the documented
failure mode that motivated the Bayesian posterior (pipeline2.speed3). This
test pins both halves of that story: the baseline must still fail (if it
stops failing, the synthetic case has gone soft and proves nothing), and
extract2 with BAYES_SPEED on must sail through with a calibrated sigma that
covers the truth. Case and seed mirror row A78 of the promotion benchmark
(experiments/bayes/bench_speed.py, seeds 1xxx).
"""
import numpy as np


def test_bpfo3_adversarial_speed():
    from relspec import pipeline2
    from relspec.pipeline2 import extract2, estimate_speed2, estimate_speed3
    from relspec.synth2 import FaultState2, machine_catalog2, generate2

    fn = 24.83                                   # induction_motor nominal
    rng = np.random.default_rng(1078)
    fr_true = fn*(1+rng.uniform(-0.02, 0.02))
    x = generate2(machine_catalog2()['induction_motor'],
                  FaultState2(outer_race=0.9), 4.0, fr_true,
                  speed_wander_pct=0.4, rng=rng).astype(np.float64)

    assert pipeline2.BAYES_SPEED, 'posterior speed path must ship enabled'
    e = extract2(x, 12000.0, band_key='adv-a78', fr_nominal=fn)
    band = pipeline2._band_cache['adv-a78']

    # the trap is real: envelope rail alone locks near BPFO/3 (~19% high)
    # with conf above the fallback gate's own 1.3 acceptance bar
    fr_e, conf_e = estimate_speed2(x, 12000.0, band, fn)
    assert abs(fr_e-fr_true)/fr_true > 0.05 and conf_e > 1.3

    # the shipped path is immune and honest about its uncertainty
    assert abs(e.fr-fr_true)/fr_true < 0.01
    fr_b, sig, k_map, w = estimate_speed3(x, 12000.0, band, fn)
    assert e.fr == fr_b                          # extract2 took the posterior
    assert k_map == 1.0 and float(np.max(w)) >= 0.5
    assert abs(fr_b-fr_true) <= 2*sig            # +/-2 sigma covers truth
