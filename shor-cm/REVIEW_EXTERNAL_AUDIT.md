# External audit of the SHOR-CM program (iterations 1–32)

Date: 2026-07-18. Auditor: independent review session (branch
`claude/vibration-grover-ensemble-research-4saxj6`), with four parallel
deep-dives: core algorithms, validation methodology, program timeline,
and 2024–2026 state of the art. This document is the honest external
assessment the program's own constitution asks for. It feeds `PHASE3.md`.

---

## 1. What was actually built (progress summary)

A 32-iteration autonomous research loop (2026-07-11 → 07-16) produced:

- **peak-Shor speed discovery** (`shorcm/peakshor.py`): classical
  continued-fraction rationalization of spectral peak-pair ratios
  (`Fraction.limit_denominator`) + sideband-spacing votes + a hand-weighted
  structure score + kinematic-sheet template match + octave arbitration.
  This is the *classical post-processing half* of Shor's algorithm. No
  quantum computation is involved anywhere in the codebase.
- **"Grover" amplification** (`shorcm/amplify.py`): a brute-force slip-grid
  scan with parabolic refinement, greedy harmonic-template matching
  pursuit, and one genuinely sound novel detector — predictive
  phase-aligned averaging with a permutation null (`ppa_zscore`).
- **Deployable cascade** (`shorcm/cascade.py`): hygiene → normalize →
  blind speed (isotonic-calibrated, abstaining) → dual pattern ledgers
  (Hz + order domain) → envelope decomposition → dual fault arms
  (transparent rules + frozen LightGBM) → severity head → subtype heads;
  streaming monitor with Theil-Sen/Mann-Kendall growth alarms
  (`shorcm/tracker.py`) and a Bayesian speed posterior
  (`shorcm/speedbelief.py`).
- **SimForge v1→v5** synthetic machine population: 7 kinematic archetypes,
  geometry-derived bearing orders, impulsive bearing rendering through a
  resonance kernel with load-zone/cage modulation (`simforge_v2.py:314-333`),
  3-channel rendering (`simforge_mc.py`), closed-form Rice/Bessel theory
  validated by ~1.3M pre-registered Monte-Carlo trials (`theory.py`,
  SYNTHVAL).
- **Governance machinery** that is ahead of most published work:
  mechanism-test-first (79 green tests), pre-registered judgments,
  promotion gate with α = 0.05/attempts, append-only ledger with kills as
  first-class results, sealed synthetic vault, frozen models with
  API-only certification, real data entering only through
  `shorcm/adapters.py`.

Self-reported end state (`HANDOFF_LOCAL.md`): speed top-1/top-3
0.614/0.837, ECE 0.038, blind 6-way fault 0.794, misalignment subtype
0.913, tracking FA 0.044 — **all on held-out SimForge, i.e.
synthetic-on-synthetic**.

Real-data results (frozen cascade, four GitHub-reachable datasets):

| Gate | Result |
|---|---|
| CWRU blind speed (C1) | **0.0 — FAIL** |
| MFPT/SEU blind speed (J1) | **0.2–0.3 — FAIL** |
| Wind-turbine speed octave (W4) | **0.12 — FAIL** |
| CWRU raw-spectrum defect orders (J3) | **0.0 — FAIL** |
| CWRU envelope subtype capture (C3) | IR 0.75 / OR 0.93 / B 0.69 — PASS |
| Envelope rescue on MFPT (J3env) | BPFO 0.90 / BPFI 0.71 — PASS, but fires on a baseline record |
| CWRU normals | 4/4 false bearing calls — FAIL |
| Severity vs. real severity (C4) | poor/negative Spearman — FAIL |
| WT run-to-failure alarm | day 31/50 — PASS (n = 1 sequence) |
| VFD growing-fault flagging (V1) | **0.0 — FAIL** |

**Honest verdict:** the program built an unusually well-governed research
machine and a genuinely good synthetic-physics stack, produced one real
transferable win (the envelope channel for bearing defect capture), and
demonstrated that its flagship blind-speed front-end **does not work on
any real dataset tested**. The founding target, MAFAULDA, was never
gated (egress-blocked for the whole program, `work/deviations.md`).

---

## 2. What is genuinely good and must be kept

1. **The leakage-safe sim-to-real protocol.** Train on synthetic, test on
   real, recording-level entry through adapters. The 2024–2025 literature
   (arXiv:2509.22267; PHME 2024 leakage-safe benchmarking) is only now
   forcing the field to abandon window-level splits that produce fake 99%
   CWRU numbers. This program's protocol is immune by construction —
   its most defensible methodological asset.
2. **Governance:** pre-registration, mechanism pins, shrinking-α promotion,
   kills ledgered as results, API-only certification (which caught the v4
   train/serve feature-order collapse 0.815→0.667 that OOF metrics could
   not see). Keep all of it.
3. **SimForge v2+ impulsive bearing physics and the Rice/Bessel theory
   layer** — real physics, honestly validated internally.
4. **The envelope channel** (iter 24): the single biggest real-data win,
   0.0 → 0.90 defect capture on real bearings.
5. **PPA permutation z-score** (`amplify.py:56`): a legitimate, novel,
   statistically principled coherent-gain detector.
6. **Candid self-auditing** (`work/audit_iter28.md`): 15 ranked findings,
   most closed by iter 32.

---

## 3. Mistakes made (ranked by severity)

**M1 — Real-data gates decoupled from the failing front-end.**
Pattern/envelope capture gates were scored at the *true* speed from
metadata (`73_real_transfer.py:103,108`), while the system's own blind
speed fails on every real dataset. The composed system (blind speed →
patterns → fault) was never honestly measured end-to-end on real data.
Headline capture claims therefore describe a system that cannot exist in
deployment (no tacho, no metadata).

**M2 — Tuning-on-test via version iteration.** CWRU and real-transfer
gates were re-scored across v4 → v4nom → v5 re-freezes (C2 recall
0.533→0.90→0.85; J2 0.53→1.0). The sealed vault protected only synthetic
data; the real gates had no vault, so the multiple-comparisons brake the
program was proudest of did not cover the numbers that matter most.

**M3 — No external baseline, ever.** No envelope-analysis-at-BPFO
baseline, no spectral kurtosis/kurtogram, no cyclic spectral coherence,
no published-CWRU comparison. The only contrast is the internal rules
arm (0.45 accuracy) — a strawman. It is currently unknown whether the
entire Shor-inspired stack beats a 1990s envelope analyzer.

**M4 — Contract misses reframed instead of counted.** PHASE2
pre-registered speed top-1 ≥ 0.95; delivered 0.614. The miss was
recharacterized as "structural, not SNR-driven." The same move appears
for C6 divergence, O4 bearing monotonicity, and the misalignment-1x
rule. Each is individually arguable; collectively it converts
falsifiable targets into unfalsifiable physics claims. A pre-registered
contract that can be reframed after the result is not a contract.

**M5 — Moving-target headline metric.** "Joint 0.21 → 0.49" spans a
population change (1 → 7 archetypes), metric redefinitions
(blind vs sheet-conditioned), and re-baselining (iter-2 joint was 0.500
on v1.1, reset to 0.318 on v2). The arc is narrative, not measurement.

**M6 — Hand-tuned constants as soft leakage.** Dozens of magic numbers
(score weights 2.2/3.0/1.2/1.6, penalties −2.5/−3.0, octave margin 0.15,
`ax_ratio_2 > 0.61`, slip_mid 0.015, concentration 0.62 …) carry comments
citing measured behavior on CWRU/MFPT/MAFAULDA-adjacent data. Constants
tuned after seeing gate data are leakage that grouped CV cannot detect.

**M7 — Unit contract never resolved.** SimForge amplitudes are arbitrary
units; absolute features (a1, a2, e_half) and rule thresholds inherit
them. The audit itself rated this near-certain to break on field data
(F1/S2). Hygiene normalizes RMS but the absolute-feature surface remains.

**M8 — 19 iterations blind to train/serve skew.** The v4 feature-order
bug proved OOF promotion metrics could not see an entire class of
defects. API-only certification was added late; it should have been the
promotion metric from iteration 1. Similarly, U1–U11: many promoted
mechanisms shipped without pins and were only tested during the audit tail.

**M9 — Tiny-N real claims without uncertainty.** CWRU n=64 files, MFPT
n=20, wind turbine n=1 run-to-failure. No confidence interval appears on
any real gate. "First alarm day 31" from one sequence is an anecdote
presented with the typography of a result.

**M10 — The founding target was abandoned rather than unblocked.**
MAFAULDA was egress-blocked from day 1; the local-execution handoff that
solves this was written only at iteration 32. Five days of iteration
optimized against substitutes while the decisive reverse-validation gate
waited.

**M11 — Permissive capture metrics.** `near_pattern`
(`70_cwru_gate.py:50-65`) accepts NEARRAT/TONE, off-integer sidebands,
or any harmonic base > 1.5 at 6% tolerance against the order *and its
2×*. The program's own theory (SYNTHVAL C3) predicts ~1/3 false-snap
per random peak at these settings — and indeed the metric fires on
healthy records (C3 normal FPs, J3env baseline FP).

**M12 — Ensemble in name only at the fault stage.** The rules arm and
LightGBM arm are both emitted but never fused (only the misalignment
subtype ORs them, `cascade.py:201`). Fault abstention is an uncalibrated
0.1 probability-margin threshold; only the speed head got isotonic
calibration. The "ensemble techniques" ambition stopped halfway.

**M13 — Quantum branding outruns content.** No component is quantum;
"peak-Shor" is CF post-processing, "Grover slip-scan" is a grid argmax.
The 2024–2026 literature contains no genuine Shor/Grover diagnosis work
either — the framing is novel, but a reviewer who reads the code will
find the labels overstated. Say "quantum-inspired classical" or drop it.

---

## 4. Opportunities missed or badly explored

**O1 — Cyclostationary analysis (the field's correct tool).** The
coherent/incoherent block-averaging split is a poor-man's cyclic
spectral coherence. Fast-SC/CSCoh separates shaft-locked, gear, and
slipping-bearing families on a principled bi-frequency map, is
transmission-path-normalized (directly attacks M7), and is the standard
against which peak-Shor must be benchmarked. Never attempted.

**O2 — Informative-band selection.** Envelope band = max kurtosis over 4
fixed bands (`envelope.py:35-42`). The kurtogram (2006) and its
2024–2025 successors (Andgram, adaptive DTCWPT, infogram) are the
established solution and are cheap to implement.

**O3 — Joint state-space speed tracking.** The failed real-data
front-end retried variations of comb-STFT scoring. A Kalman/Vold-Kalman
(or particle) tracker over instantaneous phase — with SpeedBelief as the
prior and peak-Shor candidates as measurements — is the principled
formulation of what iterations 5–12 and 27 groped toward, and is the
standard tacho-less order-tracking approach in the literature.

**O4 — Prewhitening before envelope.** MED/MOMEDA or cepstral
prewhitening before demodulation is standard for incipient faults and
would likely have prevented the raw-spectrum J3 = 0.0 failure from
needing an envelope rescue at all.

**O5 — Conformal prediction.** Isotonic + margin thresholds give
marginal calibration with no coverage guarantee. Split-conformal
prediction sets and risk-controlled abstention (SCRC-style) are
drop-in upgrades over frozen models and are arriving in FDD literature
now (arXiv:2508.01208) — a cheap, publishable rigor upgrade.

**O6 — The benchmark ecosystem.** Paderborn KAt (real damages — the
honest test, ~6–7 points harder than CWRU), XJTU-SY, IMS, FEMTO
run-to-failure (for severity/RUL claims with n ≫ 1), HUST, and the
DG-PHM / DGFDBenchmark harness with ERM baselines. All missed; several
are plain GitHub clones like the datasets already used.

**O7 — SimForge as a foundation-model pretraining corpus.** No dominant
vibration foundation model exists (VibFM: ~400 h empirical data, masked
spectrogram). SimForge can generate *unbounded labeled* machines. A
masked-spectrogram SSL encoder pretrained on millions of SimForge
machines, evaluated zero-shot on real benchmarks, is the highest-upside
unexplored direction and unifies the program's simulator, transfer
protocol, and deployment story.

**O8 — The PHASE2 realism discriminator was never built.** PHASE2
specified a sim-vs-real discriminator band (0.5–0.8) gating SimForge
realism. No such experiment exists in `experiments/`. Simulator realism
was asserted, audited by eye, but never measured.

**O9 — Simulator gaps flagged and left open:** load-varying VFD lines
(F5 — a simulator *and* detector blind spot that produces real-world
false bearings), speed ramps within a record (F6), multi-fault machines
(F9), multi-resonance transmission paths, non-Gaussian noise bursts.

**O10 — Drift detection.** The monitor tracks growth but nothing detects
distribution shift / sensor degradation; concept-drift detection in CM
is an open, thin literature — an easy differentiator.

**O11 — Deep/learned components.** No learned denoiser, no 1-D CNN on
order/envelope spectra, no stacked meta-fusion of the two fault arms, no
deep ensemble. LightGBM-on-handcrafted-features was the ceiling.

**O12 — The remaining quantum analogies (Simon, QPE ladder, DJ triage,
QEC syndrome) in `EXPANSION.md`** are unexplored. Assessment: park them.
They earn implementation only if peak-Shor first proves it beats CSCoh
head-to-head under the PHASE3 baseline panel — otherwise they multiply
branding risk without evidenced value.

---

## 5. Where the field moved (2024–2026 positioning)

- Leakage-safe, recording-level evaluation is becoming mandatory
  (arXiv:2509.22267, PHME 2024) → our sim-to-real protocol is a
  first-class asset; foreground it.
- Domain generalization benchmarks (RESS 2024, DG-PHM) show fancy DG
  often loses to well-tuned ERM → strong argument for our
  physics-simulator-DG line, but it must be benchmarked on their harness.
- Test-time adaptation (TARD, learn-then-adapt) is the deployment-relevant
  frontier → candidate for the monitor loop.
- CS-SHAP brings interpretability into the cyclic-spectral domain →
  pairs naturally with a CSCoh stage and our transparent-rules arm.
- Quantum ML for diagnosis remains VQC-toy-level with no advantage
  evidence → our honest "quantum-inspired classical" framing is
  defensible and near-unique, but only with baselines (M3) fixed.

---

## 6. Disposition

Every mistake M1–M13 maps to a structural rule in `PHASE3.md`; every
opportunity O1–O12 maps to a thrust or backlog entry there. The v5
frozen models, the ledger, and this audit are the inputs to Phase 3
iteration 1.
