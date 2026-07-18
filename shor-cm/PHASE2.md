# PHASE 2: Blind cascade and SimForge

Additive to the existing program. Reframes the target from "classify
MAFAULDA" to a deployable blind-inference cascade, and inverts the data
policy so limited labels cannot be overfitted.

## 0. Objectives, pre-registered

- O1 Blind shaft speed: top-3 candidates with calibrated confidence, from
  vibration alone. No tacho, no filename, no nominal speed.
- O2 Pattern identification and energy ledger: decompose the spectrum into
  named physical patterns, each with an energy share; shares plus residual
  sum to 1.
- O3 Fault mode: from pattern-level evidence, not raw bins, with an
  explicit "unknown" abstention.
- O4 Severity: calibrated ordinal stage per fault, with monotonicity as a
  hard requirement, not a hope.

Cascade order matters: O2-O4 consume O1's winning hypothesis, and O1's
confidence propagates. A wrong speed with high confidence is the worst
failure mode, so O1 carries the strictest calibration requirement.

## 1. Data policy inversion (the anti-overfit core)

MAFAULDA has ~1951 runs, 1 machine, 1 sensor set. Any model with real
capacity will memorize it. Therefore:

- TRAIN on synthetic (SimForge, unlimited, physics-derived, never fitted
  to MAFAULDA).
- TUNE on MAFAULDA dev (the 80% outside the vault), used only for
  transfer-gap measurement and calibration fitting.
- FINAL on the sealed vault, budget unchanged (3 evaluations).
- The north star becomes TRANSFER: performance of synthetic-trained models
  on real data. The dev-vs-vault gap detects benchmark overfitting; the
  synthetic-vs-real gap detects simulator overfitting. Both are
  first-class scoreboard columns.
- Features stay physics-level: orders, coherence ratios, snapped
  rationals, pattern energies, slip, PPA z. No raw-bin models in phase 2
  (VibFM enters later as a comparison arm, same rules).
- Augmentations derived from a MAFAULDA run INHERIT that run's vault flag
  and speed-band group. Otherwise augmentation is a leakage machine.

## 2. O1: blind speed, top-3 with confidence

### 2.0 Metadata prior (blind on speed, not on names)
No tacho and no speed reading, but the asset hierarchy names are always
available and are parsed first: equipment + component + position ("Cooling
Pump P-101, Motor, DE") identify machine type. The prior sets (a) which
pattern families to expect (vane pass, blade pass, mesh, belt), (b)
whether electrical logic applies at all: motors and generators only, with
attenuated leakage on the directly coupled driven end and essentially
none further down the train, and (c) plausible speed ranges per machine
type. SimForge emits asset names as part of ground truth so this prior is
trainable and its failure (wrong or missing names) is testable.

### 2.1 Candidate generation (recall stage)
Union of four cheap detectors, each with known failure modes so the union
covers them:
- cepstrum peaks (fails on sparse harmonics)
- harmonic comb score over an f0 grid, 2 to 120 Hz, log-spaced
  (fails toward octave errors)
- autocorrelation of the envelope (catches bearing-dominated signals where
  the shaft family is weak)
- the 2LF ladder (motor/generator points only, per the metadata prior):
  the dominant electrical vibration sits at twice line frequency, 100 or
  120 Hz nominal for mains, 2*f_e for VFD. Detect it two ways: the 2LF
  line directly at low frequency, and, more robustly, the SPACING of
  peak families in the high-frequency acceleration spectrum, where
  electrical content appears as sidebands spaced 2LF around carriers
  (slot pass, rotor bar, PWM switching). Spacing survives bad low-band
  SNR because it needs relative, not absolute, positions. Each detected
  2LF yields three ranked shaft candidates at 2LF/2, 2LF/4, 2LF/6 (pole
  pairs 1, 2, 3), sitting just BELOW those values for induction machines
  by the slip, exactly on them for synchronous machines.
Keep top-8 raw candidates plus their octave neighbors {x2, x0.5, x1.5}.

### 2.2 Rational-structure scoring (the Shor move, novel bit)
For each candidate c: angular-resample assuming c (comb-refined), take the
fine order spectrum, snap the top-30 peaks with continued fractions.
Score components, all computed per candidate:
- S_rat: energy fraction explained by small-q rationals (q <= 8)
- S_1x: presence and prominence of order 1.0 exactly
- S_coh: mean coherence ratio over the snapped set (blocks under
  hypothesis c)
- S_odd: odd-harmonic energy share

Octave errors have crisp signatures here and become NEGATIVE evidence:
- c = f0/2: every peak snaps to even integers, order 1 empty, S_odd ~ 0
- c = 2 f0: everything snaps with q = 2, half-integer forest, order 1 weak
- c = 3/2 f0: q = 3 forest
A candidate whose snap ledger is dominated by one denominator family is an
alias of a better candidate; fold that mass into the alias target.

Electrical components handled explicitly, and the distinction matters:
- MAINS-FED machines: line hum at 50/60 Hz and harmonics is truly fixed
  Hz. Detected by its known values and masked before scoring.
- VFD-FED machines: the drive synthesizes a variable fundamental f_e, and
  shaft speed = f_e * (1 - s) * 2 / poles, so running below or above
  nominal MOVES the whole electrical family with the operating point.
  These lines must never be masked as fixed hum. In the order domain they
  sit near p/(1-s), pole pairs times a 0.5 to 3% motor-slip offset:
  near-rational and unsnappable, superficially bearing-like. The
  discriminator is the jitter sign: the drive holds f_e stable while the
  shaft wanders under load, so the electrical line's order-domain wander
  is ANTI-correlated with speed wander, where a bearing tone's phase walk
  is random. ELEC evidence is therefore separable from NEARRAT evidence.
- Better still, VFD lines are a speed sensor: a stable high-Q line plus a
  pole-pair hypothesis p in {1,2,3,4} and a slip prior yields shaft
  candidates directly (a fourth candidate generator in 2.1), with the
  exact 2x relation between f_e and its magnetic 2*f_e component as an
  internal consistency check.
SimForge generates both machine populations constantly, so the scorer
cannot survive by treating electrical content as one fixed-Hz bucket.

### 2.3 Confidence
Softmax over structure scores gives a raw ranking; calibration (isotonic)
is fitted ONCE on MAFAULDA dev, mapping score margin to
P(candidate within 1% of truth). Output contract:

```json
{"speed_candidates": [
  {"hz": 29.94, "confidence": 0.87, "evidence": {"S_rat": 0.71, "S_1x": 9.2}},
  {"hz": 59.88, "confidence": 0.09, "alias_of": 0},
  {"hz": 14.97, "confidence": 0.03, "alias_of": 0}],
 "abstain": false}
```

### 2.4 O1 metrics, pre-registered
- top-1 within 1%: >= 95% on MAFAULDA dev (MAVEN reached 0.14% median
  error with nominal hints; blind must trade some accuracy)
- top-3 recall within 1%: >= 99%
- calibration ECE <= 0.05 on the vault
- octave-error share of failures reported separately (it is the number
  that matters for downstream damage)
- degradation curve vs injected line hum and VFD confusers at 0 to +10 dB

## 3. O2: pattern ledger

Extend template_pursuit into a full decomposition with a fixed vocabulary:
- HARM(k0): integer harmonic family from base order k0 (usually 1)
- HALF: half-order family (looseness)
- NEARRAT(o, slip): drifting tone at kinematic order o, found by slip_scan
  and confirmed by PPA z (bearing tones)
- SIDEBAND(oc, os): fan around carrier oc spaced os (gear mesh, inner race
  modulation)
- BAND(lo, hi): broadband hump (resonance excitation)
- FIXEDHZ(f): truly non-synchronous line (mains hum, fixed-speed
  neighbor machine)
- ELEC(p, s): electrical family (motors/generators only, gated by the
  metadata prior). Low band: the 2LF line near order 2p/(1-s). High band:
  sideband combs spaced 2LF (or 2 f_e) around slot-pass / rotor-bar / PWM
  carriers in the acceleration spectrum, detected by comb SPACING.
  Identified against NEARRAT by frequency stability and wander
  ANTI-correlated with shaft speed. Amplitude expectation decays with
  distance from the motor in the kinematic graph; strong ELEC on a
  far-end pump bearing is itself an anomaly worth flagging
- FLOOR: residual

Greedy extraction in that order, energy accounted in the order domain,
output ledger: pattern, parameters, energy share, coherence ratio, PPA z.
Shares sum to 1 by construction. Metric on SimForge (where composition is
known truth): attribution error per pattern < 10% absolute share;
residual < 15% on faulted runs. Stability check: ledger from first half vs
second half of a record, share drift < 5%.

## 4. O3: fault mode

Input is the ledger, never raw bins. Two arms, compared honestly:
- RULES: transparent mapping (HARM dominated by 1x -> imbalance; strong
  2x with 2x phase signature -> misalignment; HALF present -> looseness;
  NEARRAT at bearing orders + PPA z > 4 -> bearing, subtype by which
  order; SIDEBAND at mesh -> gear). Ships with confidence from evidence
  margins.
- ML: gradient boosting on ledger features, trained on SimForge only.
Abstention: if no rule fires and ML margin is low, output "unknown" with
the ledger attached. Coverage-accuracy curve is the metric, not bare
accuracy: at 90% coverage, macro-F1 >= 0.85 on MAFAULDA dev,
synthetic-trained. The rules arm is the floor the ML arm must beat to
justify itself.

## 5. O4: severity

Per fault mode, an ordinal stage 1 to 4 built on physical drivers:
- imbalance: 1x velocity amplitude against ISO 10816 zone boundaries for
  the machine class (SimForge knows its class; real deployments ask)
- misalignment: 2x/1x ratio plus 2x absolute level
- bearing: composite of NEARRAT energy, count of visible bearing-order
  harmonics, sideband richness around them, and PPA z; stages anchored to
  the classic incipient/moderate/severe envelope progression
- looseness: half-family richness and harmonic count
Model: ordinal regression with monotonic constraints on the drivers
(lightgbm monotone_constraints). Limited labels are survivable because the
model only orders a handful of physical quantities.
Metrics: Spearman rho >= 0.8 on each MAFAULDA severity ladder (imbalance
masses 6 to 35 g, misalignment mm steps), ordinal accuracy within one
stage >= 90%, ZERO tolerated monotonicity violations on synthetic sweeps
(guardrail, not metric).

## 6. SimForge: the synthetic upsampling engine

### 6.1 Principles
Written from physics and standards knowledge (bearing kinematics, ISO
10816/13373 signatures, gear sideband theory), NEVER fitted to MAFAULDA.
MAFAULDA is validation of realism, not a target. Domain randomization
wide enough that the real machine is one draw from the prior.

### 6.2 Forward model, layer by layer
1. Machine archetype (kinematic graph): motor-only; motor+coupling+pump;
   belt-driven fan (belt order = pulley ratio, a new rational family);
   single-stage gearbox (tooth counts 17 to 113, coprime-biased);
   two-stage gearbox. Planetary parked for phase 3.
   Bearing geometries drawn from the kinematic-builder bearing database
   (100+ real entries), giving true BPFO/BPFI/BSF/FTF spreads instead of
   one rig's numbers.
2. Speed profile: constant, linear drift up to 3%/min, cyclic load wander
   0.1 to 3% at 0.1 to 2 Hz, plus rare speed steps. Range 5 to 120 Hz.
3. Fault injection with severity as a continuous latent s in [0, 1]:
   - imbalance: 1x acceleration amplitude ~ s * omega^2, correct
     speed scaling built in
   - misalignment: 2x (+3x) with deterministic 2x-vs-1x phase relation,
     parallel vs angular variants differ in axial share (multi-channel)
   - looseness: half-order family + waveform clipping nonlinearity
   - bearing: impulse train at characteristic rate with per-impact jitter
     (slip), amplitude modulation by cage or shaft depending on race,
     impulses exciting 1 to 3 resonance bands; severity drives impulse
     energy, harmonic count, and sideband richness
   - gear: mesh tone + sidebands at shaft orders, severity drives sideband
     spread; local tooth fault adds once-per-rev modulation
4. Transmission path: 2 to 5 randomized SDOF resonances (fn 300 Hz to
   8 kHz, Q 5 to 40), channel-specific, plus cross-channel leakage.
5. Sensor and acquisition: accelerometer sensitivity error, 16-bit
   quantization, occasional clipping, DC drift, dropouts, sample-rate
   family {12.8, 25, 25.6, 50} kHz. Corruption modes feed H6.
6. Electrical environment, always on at random levels, two populations:
   - mains-fed: fixed 50 or 60 Hz hum family, shaft near synchronous
     speed minus load-dependent slip
   - VFD-fed: movable fundamental f_e spanning the speed range (below
     nominal AND slightly above), pole pairs p in {1,2,3,4}, motor slip
     0.5 to 3% varying with load and anti-correlated with shaft wander,
     switching carrier 2 to 16 kHz with sidebands at +/- f_e and
     +/- 2 f_e, and exact 2*f_e magnetic vibration
   Electrical content is generated ONLY when the measured component is
   a motor or generator, at full strength, with attenuated leakage on the
   coupled driven end and near-zero further down the train. HF structure
   included: slot-pass / rotor-bar / PWM carriers with sideband families
   spaced exactly 2LF (mains) or 2 f_e (VFD), because the spacing
   detector must have something true to find. Every run carries asset
   names (equipment, component, position) as ground truth for the
   metadata prior.
   Plus one neighboring machine (either population) at an unrelated
   speed. These are what make blind speed hard in the field, and the
   VFD case is the one that punishes any fixed-Hz assumption.

### 6.3 Scale and labels
Every run ships with complete ground truth: speed profile, kinematics,
fault mode, severity latent, full pattern composition (which is what
makes O2's attribution metric possible at all). Target corpus v1: 50k
runs, ~10 s each, regeneratable from seeds, so nothing is stored but a
manifest.

### 6.4 Hybrid augmentation of MAFAULDA (used sparingly, eval-side only)
- speed re-synthesis: tacho-based angular resample to new speed profiles,
  same fault at unseen speeds (legitimate because tacho truth exists)
- background swap: healthy-run floor + order-domain-masked fault content
  from another run at controlled SNR
- sensor corruption overlays for H6
Leakage rule from section 1 applies to every derived run.

### 6.5 Realism gates (SimForge's own adversary)
- Discriminator test: classifier on physics features, real vs synthetic,
  unlabeled. AUC target BAND 0.5 to 0.8. Above 0.8 the simulator is
  distinguishable in ways that matter; document which features leak.
- Marginal matching: distributions of spectral slope, kurtosis, crest
  factor, floor shape within 2-sigma bands of MAFAULDA dev.
- Reverse validation, the one that counts: cascade trained on SimForge
  only must hit floor targets on MAFAULDA dev (O1 top-1 >= 90%, O3
  macro-F1 >= 0.75 at 90% coverage) BEFORE any tuning. The
  synthetic-to-real transfer gap is tracked on the scoreboard forever.

## 7. Experiments, gated like everything else

- E1 SimForge v1 + realism gates (blocks all downstream)
- E2 O1 blind speed on dev + confuser degradation curves
- E3 O1 calibration, one vault access shared with E5
- E4 O2 attribution on SimForge truth + stability on real
- E5 O3 rules vs ML, coverage-accuracy, transfer gap; vault access
- E6 O4 severity ladders + monotonicity sweeps
Each is a loop iteration: mechanism tests first, ledger entry always,
promotion gate on the transfer metric.

## 8. Team additions (additive)

- simforge-engineer: owns the forward model; forbidden from reading any
  MAFAULDA-fitted statistic; realism gates run by the adversary, not by
  this agent.
- calibration-auditor (folded into metrics-keeper): owns ECE, reliability
  diagrams, and the confidence contract of O1/O3 outputs.
- adversary gains: augmentation-leakage audit (derived runs vs vault
  flags), discriminator test execution, confuser stress runs.

## 9. Order of work

1. SimForge core (archetypes 1 to 3, faults, path, confusers) + gates
2. O1 scorer + calibration
3. O2 ledger decomposition
4. O3 rules arm, then ML arm
5. O4 severity
6. Vault evaluation, report, decide phase 3 (planetary, VibFM arm, field
   data transfer via MAVEN)

## 10. Open TODOs (Matthias)

1. Bearing database export from the kinematic builder for SimForge priors.
2. ISO 10816 machine-class assumption for O4 on MAFAULDA-like rigs
   (suggest Class I, small machines).
3. Approve the inverted data policy (synthetic-train, real-eval) as the
   program default.
4. Line frequency prior: 50, 60, or both (suggest both), and the
   VFD-fed share of the SimForge population (suggest 50%, matching the
   MachineDoctor installed base better than a mains-only world).
5. Whether O1 abstention is allowed in production or top-3 is always
   emitted (suggest: always emit, abstain flag for confidence < 0.5).
