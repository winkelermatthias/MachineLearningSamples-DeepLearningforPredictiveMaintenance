# Bayesian SP: prove-then-integrate results

Four candidates from `docs/SP_RESEARCH.md` were prototyped against hard
promotion gates by an autonomous workflow (grid search skipped by
decision). Two passed and are integrated behind flags; two failed and
remain as experiments. The negative results are reported with the same
prominence as the wins — both rejections were decided by measurement,
not taste. All benchmarks are deterministic (fixed seeds) and live in
`experiments/bayes/` with their metrics JSONs; suites after
integration: platform 24/24, edge 7/7, firmware green.

## 1. Bayesian speed posterior — PROVED, integrated

**Method.** Rao-Blackwellized posterior over rotation speed: one
2-state Kalman (f_r, ḟ_r) per harmonic hypothesis k ∈ {⅓, ½, 1, 2, 3}
with a discrete posterior over k, measurement likelihoods from the same
envelope-HPS and acc-comb surfaces the shipped estimator uses, evidence
fused across both rails until the mixture collapses.

**Benchmark.** 84 acquisitions: synth2 across the machine catalog +
low-SNR variants + an adversarial set engineered so the envelope rail's
strongest comb sits at ~1.19× f_r (the historical "confidently wrong"
BPFO/3 failure). Ground truth = synthesis speed.

**Numbers** (combined corpus; baseline = `estimate_speed2`):

| | median err | p95 err | gross (>5%) |
|---|---|---|---|
| estimate_speed2 | 13.34% | 32.25% | 62 |
| Bayes posterior | **0.145%** | **0.349%** | **0** |

(The baseline's collapse is the adversarial set doing its job — on
benign data it holds its published 0.29%/0.65%; the posterior wins
there too and eliminates the failure mode entirely.) Empirical ±2σ
coverage: 1.0. Runtime 2.4× baseline (11 ms vs 4.6 ms/acq) — inside
the 3× gate.

**Integration.** `relspec/src/relspec/pipeline2.py: estimate_speed3`,
default-on via module flag, `estimate_speed2` intact as fallback; σ
propagates into the extractor's confidence. `platform/tests/
test_speed3.py` reproduces the promotion win against the *integrated*
path.

## 2. Fast-SC / improved envelope spectrum — PROVED (as second opinion), integrated

**Method.** Antoni's STFT-based Fast Spectral Correlation
(bin-shifted cross-products, frame-DFT over cyclic frequency α,
coherence normalization), self-checked on a known-α synthetic
(α̂ error 0.25 Hz). IES = |coherence| integrated over the spectral
axis.

**Benchmark.** Low-severity outer-race campaigns (sev 0.05–0.2 ×
seeds × SNR) vs the shipped kurtogram-band + envelope-Welch path,
scoring BPFO fault-line SNR and detect/no-detect at 6 dB.

**Numbers.** Gate criterion G1 (median SNR gain ≥3 dB at the two
lowest severities) FAILED: +1.14 dB at sev 0.05, −4.66 dB at sev 0.10.
Gate criterion G2 PASSED: **18 cases detected that the baseline
misses, with zero new false lines on healthy records** — but the IES
also *loses 13 cases the baseline catches*. The two front-ends see
different physics; neither dominates.

**Integration — shaped by that result.** Fast-SC ships as a
**second opinion only**: `src/relspec/fastsc.py` +
`GET /v1/acquisitions/{id}/ies` (computed from the stored raw waveform;
honest 404 on codec-only frames; ETag-cached; flag-guarded). The
shipped detection path is untouched — the endpoint adds the 18 without
risking the 13. Codec and rails unchanged. `platform/tests/test_ies.py`
covers the endpoint against a campaign with a stored raw waveform.

## 3. BOCPD gate — NOT PROVEN, rejected

**Method.** Hand-rolled Adams–MacKay BOCPD (Student-t/NIG conjugate
predictive, constant hazard, run-length pruning, ~90 lines of numpy)
over the same score sequence the v1 Gate consumes; the actual v1 Gate
(imported) as baseline. 36 campaigns: 12 no-fault, 8 abrupt, 8 ramp,
8 speed-wander nuisance.

**The number that rejected it.** The v1 Gate found a clean operating
point: s_hi = 2.5 → **0 false alarms across all 20 no-fault/nuisance
campaigns, 14/16 faults detected, mean latency 1.0 acquisition on
abrupt faults**. BOCPD, across hazard × window × threshold sweeps
(thresholds to 0.99), **had no zero-false-alarm operating point at
all** on the same corpus — its best zero-FA-adjacent configuration
carried 31.75-acquisition latency. Runtime was fine (53 µs/step); the
detector just doesn't beat a well-tuned CUSUM-with-persistence on this
signal. The 2006-era gate earns its keep. Experiment retained at
`experiments/bayes/bocpd.py` for future revisits (e.g. richer
multivariate inputs).

## 4. Per-track Kalman (level + slope) — NOT PROVEN, rejected

**Method.** 2-state Kalman per pattern track over E_dB with Huberized
updates and online noise estimation; change scored by normalized
innovation; slope state → predicted threshold-crossing time. Benchmark:
189 track histories from 27 campaigns (quiet / ramp / step / noisy /
drifting), against a faithful reimplementation of the shipped static
Welford z.

**Split verdict, honestly scored.** Two of three gate criteria passed
— detection latency ≥1 frame earlier at matched false-alarm rate, and
no quiet-track FA regression. The third failed: **crossing-time
predictions landed within ±30% on fewer than 60% of ramp cases** — the
slope state is too noisy at our per-day acquisition cadence to promise
"days to alert" with a straight face, and that promise was the point.
Rejected for now; the detection-latency half may return once
fingerprint-cadence (per-acquisition) trends give the filter 10–50×
more observations per day to smooth over. Experiment retained at
`experiments/bayes/kalman_track.py`.

## Scoreboard

| candidate | gate | verdict | shipped as |
|---|---|---|---|
| speed posterior | p95 −25%, fewer gross, <3× runtime | **pass** (p95 −98.9%, 0 gross) | `estimate_speed3`, default on |
| Fast-SC / IES | +3 dB median OR extra detects, 0 false | **pass** (G2: +18 detects, 0 false) | `/v1/acquisitions/{id}/ies` second opinion |
| BOCPD gate | ≤ CUSUM latency at zero FA | **fail** (no zero-FA point exists) | experiment only |
| track Kalman | latency + FA + ±30% crossing | **fail** (crossing criterion) | experiment only |

The gates worked in both directions: the two integrations are backed by
numbers, and the two rejections cost a day of compute instead of a
production regression.
