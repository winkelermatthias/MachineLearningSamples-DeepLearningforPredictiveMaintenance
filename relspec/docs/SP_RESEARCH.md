# Signal-processing research: grid search, Bayesian filtering, and what
# they should actually be aimed at

Research memo. Every proposal below is stated against what the pipeline
measurably does today, with the number it is expected to move and the
harness that would prove it. Nothing here is implemented; this is the
map for the next deepening.

## 0. Where we stand (the numbers these ideas must beat)

| outcome | today | weak point |
|---|---|---|
| speed estimate | median 0.29%, p95 0.65% err | p95 tail = harmonic ambiguity (the envelope estimator was "confidently wrong" at BPFO/3 until the acc-comb corroboration hack) |
| detection | 24/24 faults, 0/22 false alarms (synthetic fleet) | thresholds hand-set; no calibrated false-alarm rate; detection latency untracked |
| pattern z | works (campaign-verified) | static Welford baseline; no growth rate; a slow ramp is only "high z", never "days to threshold" |
| envelope rail | 35% mean residual share | unexplained energy — either genuinely unpatterned or our band/envelope front-end is leaving signal on the table |
| bytes | 65–262× vs raw | dead-zone k, GOP trig, gate thresholds all hand-tuned; no measured Pareto frontier |
| health tier | max(floor, gate, z) + 5-frame hysteresis | ad-hoc smoothing; no probability attached to a tier |

## 1. Grid search — three places it pays

### 1.1 Hyperparameter Pareto sweep (highest leverage per line of code)
The pipeline has ~15 coupled tunables (kurtogram levels & min bandwidth,
comb tolerance and verify thresholds, dead-zone k/floor, GOP trig=0.85,
gate s_lo/s_hi/persistence votes, z teach guard, hysteresis N). Every
one was set by a person looking at one experiment. We already own the
labeled corpora (CWRU/MFPT/SEU + 48-asset synthetic fleet) and the eval
harnesses (`eval_real`, `eval_rd`, `fleet2`), so a sweep is pure
compute: objective = (detection latency, false alarms, bytes/frame,
speed p95) as a Pareto front, not a scalar. Plain grid explodes at 15
dims — use **Bayesian optimization** (GP/TPE, e.g. Optuna) seeded by a
coarse grid on the 4 most sensitive knobs; ~500 pipeline evaluations ≈
one afternoon of CPU. Deliverable: a measured frontier ("at 0 false
alarms, fastest detection costs N bytes/frame") and tuned defaults with
provenance instead of folklore. Expected: 10–25% byte reduction at
equal detection, or 1–2 frames earlier detection at equal bytes —
whichever point on the frontier we choose to ship.

### 1.2 Matched kinematic comb bank (grid over hypotheses, not bins)
Today pattern discovery is blind (greedy peaks/combs), which is why the
ledger shows tracks like "comb 2.03× · mesh?". The v1 ontology already
stores bearing/gear kinematics. Turn discovery around: for each
component, enumerate the **catalog hypothesis grid** {BPFO, BPFI, BSF,
FTF, GMF ± sidebands} × speed-uncertainty window, and score each
hypothesis with a comb filterbank energy + the existing verify_combs
test. Blind discovery stays for the unexpected; the bank gives named,
stable track identities — which directly strengthens the z model (fewer
track births/deaths, longer baselines) and makes the UI say "BPFO (SKF
6205)" instead of "comb 3.58×".

### 1.3 Fine (f0, Δ) grid refinement
The least-squares f0 refit is local. A dense 2-D grid over
(fundamental, sideband spacing) around each accepted family — cheap at
±tolerance resolution — separates close modulation families (mesh ±
shaft vs mesh ± cage) that the greedy pass currently merges. Feeds 1.2.

## 2. Bayesian filtering — five places, ranked

### 2.1 Speed as a posterior, not a point (kills the p95 tail)
The current estimator is Viterbi over an HPS surface plus a hand-built
"dual evidence" corroboration rule — a disguised, hard-coded Bayesian
update. Do it properly: state (f_r, ḟ_r), measurement likelihood = the
HPS/comb surfaces from BOTH rails, and — critically — an explicit
**mixture over harmonic index k** (the 1×/2×/3×/BPFO-3 ambiguity that
caused the confidently-wrong failures). A Rao-Blackwellized filter
(Kalman per hypothesis, discrete posterior over k) keeps every
hypothesis alive until evidence collapses it. Outputs: smoother f_r,
an honest σ(f_r) to propagate into order-bin width and the `conf`
field, and hypothesis switches logged instead of silently wrong.
Expected: p95 0.65% → ~0.3%, and graceful degradation at low SNR where
today the estimator commits early.

### 2.2 Per-track Kalman (level + slope): from z to "days to threshold"
The z model asks "is this track abnormal now". A two-state
linear-Gaussian model per track — E_dB level + slope, Kalman filtered —
adds what the product actually sells:
- **innovation-based z** (normalized innovation squared) replaces the
  static baseline z: adaptive to per-track noise, faster on steps,
  immune to the slow-ramp re-normalization we currently prevent with
  the |z|<3 teach guard;
- the **slope state is a growth rate with a credible interval** →
  "BPFO energy +0.4 dB/day ± 0.1 → crosses alert in 9–16 days". That
  line on the asset page is RUL-lite, and it falls out of a 40-line
  filter over data we already store. (Kalman-filter changepoint/trend
  characterization is standard practice in the monitoring literature.)

### 2.3 Gate → Bayesian online change-point detection (BOCPD)
The gate's median/MAD + CUSUM + persistence votes is a frequentist
approximation of BOCPD (Adams & MacKay). Replacing the decision layer
with a run-length posterior gives: calibrated P(change) instead of a
threshold crossing, a hazard prior that encodes expected fault rates,
and a directly tunable false-alarm budget (the thing we currently
demonstrate but cannot dial). Constant memory with pruning; the C
firmware port stays feasible. PHM applications report exactly our
target behavior: immediate on abrupt changes, drift flagged when
cumulative evidence clears noise.

### 2.4 Health tier → 4-state HMM
`max(floor, gate, pattern) + 5-frame hysteresis` becomes a hidden
Markov model with degradation-biased transitions and the same three
signals as emissions. Forward posterior per frame = P(healthy…critical)
— the hysteresis emerges from the transition prior instead of a magic
N=5, tier flapping becomes impossible by construction, and the UI can
finally show confidence on the tier chip. Drivers stay (emission
likelihoods per signal are exactly the "why").

### 2.5 Particle-filter RUL (phase 2)
Once 2.2's growth rates exist: a particle filter over a Paris-law-style
degradation state gives full remaining-useful-life posteriors. Standard
in the prognostics literature; only worth building after slope
estimates prove stable on real fleets.

## 3. The modern SP the question points at (beyond the two names)

### 3.1 Cyclic spectral coherence / Fast-SC (the big one)
Kurtogram-band + envelope is the 2006 answer. The 2017+ answer is the
**spectral correlation/coherence** (second-order cyclostationarity),
computed cheaply via Fast-SC (STFT-based; cost ~N log N) or the Fast
Averaged Cyclic Periodogram. Integrating coherence over the carrier
band yields an **improved envelope spectrum**, and selecting the
carrier band per cyclic order (IESFOgram) replaces the kurtogram's
single-band compromise with a per-fault-family optimum. Why it matters
here specifically: (a) our envelope rail carries 35% unexplained
residual — CSCoh separates true cyclostationary fault energy from
random noise where the plain envelope cannot; (b) kurtogram already
cost us one −4.6 dB regression before the bandwidth guard; CSCoh is
robust to exactly those interference cases. Integration: cloud-side as
a second-opinion detector on stored/decoded frames first (no codec
change), promoted into the rail definition only if the eval harness
shows it beats envelope-Welch at equal bytes. Expected: measurable
detection-SNR gain at low severity (earliest detection is where
products win), and a defensible explanation of the envelope residual.

### 3.2 Blind deconvolution preconditioning (MED/CYCBD)
Maximum-cyclostationarity blind deconvolution (CYCBD) sharpens impact
trains before enveloping, using the cyclic frequency from the kinematic
bank (1.2) as its target. Helps most at low SNR/long transmission
paths. Cheap to trial inside the same second-opinion harness.

### 3.3 Already specified, still unbuilt
Time-synchronous averaging + angle resampling (the streaming extractor
in the README's known limits) remains the gear-side counterpart to 3.1
and benefits directly from 2.1's smoother speed posterior.

### 3.4 Super-resolution line estimation
Bayesian harmonic regression (Bretthorst-style) on the few bins around
a verified line beats FFT grid resolution ~10× for f0 — sharper
sideband spacing (slip, cage ratio) and better anchors for 1.3 —
without touching frame size.

## 4. Priority and proof plan

| # | item | cost | moves | proof harness |
|---|---|---|---|---|
| 1 | 1.1 BO/grid Pareto sweep | low (compute) | bytes, latency, FA | existing eval_* + Optuna driver |
| 2 | 2.2 per-track Kalman + growth | low | z quality, adds RUL-lite | extend test_health campaigns (ramp → predicted crossing vs actual) |
| 3 | 2.1 speed posterior w/ harmonic mixture | medium | speed p95, conf honesty | eval_dsp speed benchmark incl. adversarial BPFO/3 cases |
| 4 | 3.1 Fast-SC / IES second opinion | medium | detection SNR, envelope residual story | eval_real detection at low severity, CWRU+MFPT |
| 5 | 2.3 BOCPD gate | medium | calibrated FA rate | fleet2 false-alarm sweep vs CUSUM |
| 6 | 2.4 health HMM | low | tier stability + probabilities | test_health hysteresis cases |
| 7 | 1.2 kinematic comb bank | medium | track naming/stability | ledger churn metric on fleet2 |
| 8 | 3.2 CYCBD, 2.5 PF-RUL, 3.4 | later | — | after the above land |

Ordering rationale: 1 and 2 are near-free against infrastructure that
already exists and improve shipped numbers immediately; 3 fixes our
only known "confidently wrong" failure mode; 4 is the biggest
detection-quality ceiling-raiser but must prove itself against the
codec's byte budget before touching the rail format.

## References

- Antoni, Xin, Hamzaoui — Fast computation of the spectral correlation
  (Fast-SC); Fast Averaged Cyclic Periodogram (ISA Trans. 2022).
- Antoni — Cyclic spectral analysis of rolling-element bearing signals:
  facts and fictions (JSV 2007).
- Spectral-coherence-derived indicators for bearing diagnosis
  (IES/IESFOgram line of work).
- Adams & MacKay — Bayesian Online Changepoint Detection (2007).
- BOCPD for degradation monitoring and prognostics (PHM Society, 2024).
- Kalman-filter changepoint detection and trend characterization;
  change-point-based adaptive Kalman filtering (noise-adaptive state
  estimators, 2024).
