# SHOR-CM Phase 3 — Full End-to-End Instructions (Local Session, MAFAULDA-First)

**Audience:** the Claude Code session running on the local PC where the MAFAULDA
dataset already exists in a local folder. **Author:** external-audit session,
2026-07-18, branch `claude/vibration-grover-ensemble-research-4saxj6`.

This document is self-contained: it tells you what exists, what is trusted,
what is suspect, and exactly what to do — in order — from environment setup to
the pre-registered exit criteria. Read it fully before running anything.

Companion documents in the repo (read after this one):

- `REVIEW_EXTERNAL_AUDIT.md` — the independent audit of iterations 1–32
  (13 mistakes M1–M13, 12 missed opportunities O1–O12). Justifies every rule here.
- `PHASE3.md` — the constitution (amendments A1–A13, thrusts T1–T8). This
  handoff operationalizes it.
- `REPORT_SYNTHETIC_PROGRAM.md`, `work/loop_ledger.jsonl` — full Phase 1–2 history.
- `CLAUDE.md`, `PHASE2.md`, `EXPANSION.md` — prior constitutions (still valid
  where not amended).

---

## 0. Context in one page — what you are inheriting

**The system.** A vibration condition-monitoring cascade
(`shorcm/cascade.py`): hygiene → normalize → blind speed estimation
("peak-Shor" continued-fraction period finding, `shorcm/peakshor.py`) →
pattern ledgers (Hz + order domain) → envelope decomposition → dual fault arms
(transparent rules + frozen LightGBM) → severity + subtype heads → streaming
monitor with growth alarms (`shorcm/tracker.py`). Models frozen at v5 in
`models/`. Real data enters ONLY through `shorcm/adapters.py`. 79 tests green.

**What is trusted.** The governance machinery (pre-registration, promotion
gates, append-only ledger, API-only certification), the theory layer
(`shorcm/theory.py`, validated by ~1.3M pre-registered Monte-Carlo trials),
the envelope channel (real bearing-defect capture 0.90 on CWRU/MFPT), and the
leakage-safe train-on-synthetic/test-on-real protocol.

**What is NOT trusted.**

1. **All headline accuracy numbers** (fault 0.794, subtype 0.913, speed
   0.614/0.837) — they are synthetic-on-synthetic (SimForge validating
   SimForge-trained models).
2. **Blind speed estimation on real data** — it FAILED on every real dataset
   tested (CWRU 0.0, MFPT/SEU 0.2–0.3, wind turbine 0.12).
3. **Real-data "capture" PASSes** — they were scored at the *true* speed from
   metadata, not the system's own blind estimate (audit M1).
4. **SimForge realism** — never measured. The planned sim-vs-real
   discriminator gate was never built (audit O8). Assume the synthetic data
   has flaws until the discriminator says otherwise.
5. **Hand-tuned constants** — dozens of magic numbers were tuned after seeing
   real gate data (audit M6).

**Your mission.** Stand the system up on MAFAULDA (the founding target, never
yet gated), measure everything blind and against external baselines, find and
fix the synthetic data's flaws systematically, then improve the algorithms —
under the Phase 3 constitution, autonomously, iteration by iteration.

---

## 1. Non-negotiable ground rules (constitution digest)

These are the Phase 3 amendments (full text in `PHASE3.md`). Violating one
invalidates the iteration.

1. **A1 Blind means blind.** No real-data gate may use metadata speed, tacho,
   or filename speed except in a clearly labeled ORACLE diagnostic column.
   Promoted numbers come from `Cascade.analyze_record` / `monitor` end-to-end.
   MAFAULDA's tacho channel is for *judging* speed accuracy only — never an input.
2. **A2 Real-data vault.** Split every real dataset ONCE at catalog time by
   run-id hash: `dev-real` (60%, open) and `sealed-real` (40%, budget: 3
   evaluations for the whole phase, each one ledgered). Never re-freeze and
   re-score the same sealed gate (that killed Phase 2's credibility — audit M2).
3. **A3 Baseline panel.** Nothing about the signal path promotes unless it
   beats, blind, through the same adapters: B1 (SK/kurtogram + envelope
   spectrum at nominal defect orders), B2 (cyclic spectral coherence peak
   detection), B3 (ERM LightGBM/1-D CNN on standard features). Baselines are
   first-class code with tests.
4. **A4 Contract accounting.** `work/contracts.jsonl`, append-only: every
   pre-registered target gets a final verdict MET / MISSED / RETIRED(new-id).
   Misses stay missed. Reframing requires a NEW falsifiable contract.
5. **A5 Metric constitution.** Every number carries `population_id` and
   `metric_id`. No cross-population deltas in promotions or narratives.
6. **A6 Constant provenance.** New/changed thresholds go to a provenance
   table (value, unit, fit-source THEORY/SYNTH/DEV-REAL/HAND, date). HAND
   constants touched after real-data contact must be re-derived or refit on
   dev-real by a script.
7. **A7 Units.** SimForge v6 emits m/s² through a sensor model; inputs are
   unit-tagged; absolute features banned unless unit-tagged.
8. **A8 Serve-path metrics.** Evaluate through freeze → `analyze_record`,
   never in-memory OOF. Mechanism pin (pytest) required AT promotion.
9. **A9 Uncertainty floor.** Every real number ships with N and a 95% CI
   (Wilson / BCa bootstrap). N < 30 recordings ⇒ auto-label ANECDOTE.
10. **A10 Data before iteration.** Max 2 consecutive iterations without
    touching real data.
11. **A11 Matched false-positive gates.** Every capture/detection metric is
    reported next to its false-fire rate on healthy records. No FP gate ⇒ no PASS.
12. **A12 Two-key promotion.** The implementing agent never judges its own
    work: a separate judge subagent re-derives the verdict from result
    artifacts only. Every 5 iterations a replication subagent re-computes the
    headline table from frozen artifacts; non-reproducible numbers are retracted.
13. **A13 Honest branding.** "Quantum-inspired classical," never "quantum."
    Position against CSCoh/kurtogram/Andgram.

Standing rules from Phase 1–2 that remain: seed 20260709; mechanism test
first; α = 0.05/attempts promotion gate; kills are ledgered results, not
setbacks; never weaken a gate to pass it; never edit tests to make them pass;
`SELF_REF_MASK` for signal-derived phase variants; grouped splits by speed
band (plus the new run-id vault).

---

## 2. Stage 0 — Environment and repo verification (half a day)

```bash
git clone -b claude/vibration-grover-ensemble-research-4saxj6 \
  https://github.com/winkelermatthias/MachineLearningSamples-DeepLearningforPredictiveMaintenance.git
cd MachineLearningSamples-DeepLearningforPredictiveMaintenance/shor-cm

python3 -m venv venv                     # Python 3.10+
venv/bin/pip install -r requirements.txt
OMP_NUM_THREADS=1 venv/bin/python -m pytest -q tests/   # expect: all green (79)
OMP_NUM_THREADS=1 venv/bin/python scripts/99_smoke.py   # end-to-end synthetic smoke
```

Verification checklist (write `work/gates/P3_G0.json`):

- [ ] pytest green; record versions of numpy/scipy/sklearn/lightgbm.
- [ ] `numpy >= 2.0` (`theory.py` uses `np.trapezoid`; on numpy 1.x it crashes).
- [ ] `models/` contains `fault_lgbm.joblib`, `severity_lgbm.joblib`,
      `conf_isotonic.joblib`, `meta.json` — record their hashes; these are the
      frozen v5 artifacts, do not retrain them in Stages 1–3.
- [ ] Disk ≥ 60 GB free; note core count; use `WORKERS=<cores-1>`,
      `OMP_NUM_THREADS=1` for every script.
- [ ] Git identity + push access to the branch (commit/push every iteration).

**Known code landmines (from the code audit — check before relying on):**

- `shorcm/features.py` hardcodes `FS = 50_000` at module level. Fine for
  MAFAULDA (which is 50 kHz), silently wrong for anything else entering that
  legacy path. The cascade path passes fs explicitly — prefer it. Fix the
  hardcode in your first hygiene iteration.
- `shorcm/features.py` `BEARING_ORDERS` carries `TODO(A1)`: verify the
  bearing geometry orders against the MAFAULDA documentation (MB ER-10K:
  BPFO ≈ 2.998, BPFI ≈ 5.002, BSF ≈ 1.871, FTF ≈ 0.375 per rev — confirm
  against the dataset page before trusting any bearing feature).
- `phase_from_comb` (`shorcm/tacho.py`) uses `nperseg = 0.4·fs` (20k samples
  at 50 kHz); records shorter than ~1 s degrade or raise — guard it.
- Heavy redundant STFT/peak computation per record; if throughput hurts,
  memoize per-record spectra (there is a `<5 s/record` guardrail G-c).
- The isotonic confidence sentinel guard (`cascade.py` ~l.136) is
  load-bearing; do not "simplify" it away.

---

## 3. Stage 1 — Data: catalog, gates, vault splits (day 1)

### 3.1 MAFAULDA (already in a local folder — PRIORITY ONE)

Point the adapter at the local folder (do not reorganize the data; adapt the
adapter if the layout differs — data is read-only ground truth):

```
data/mafaulda/            # or symlink: ln -s /path/to/local/mafaulda data/mafaulda
  normal/                 # 49 runs expected
  imbalance/              # 333
  horizontal-misalignment/  # 197
  vertical-misalignment/    # 301
  underhang/              # 558   (bearing faults, underhang position)
  overhang/               # 513   (bearing faults, overhang position)
```

G1 catalog gate (script it; write `work/gates/P3_G1_mafaulda.json`):

- [ ] 1951 total runs (or document the true count vs the published one —
      treat published counts as expected, not gospel).
- [ ] 8 channels everywhere: tachometer, underhang accel (axial/radial/
      tangential), overhang accel (axial/radial/tangential), microphone.
      50 kHz, 5 s (250k samples). Flag any deviant file, don't silently drop.
- [ ] Speed sanity: tacho-derived rate within [700, 3600] rpm; fraction of
      runs with |rate − nominal|/nominal > 0.15 is < 2%.
- [ ] Severity metadata parsed (imbalance grams: 6–35 g; misalignment mm;
      bearing fault types: cage/outer race/ball for each position).
- [ ] **Vault split (A2):** hash run-ids → 60% dev-real / 40% sealed-real,
      stratified by class and speed band. Write the split file once, commit
      it, never touch it again. Also mark speed-band groups for CV.

### 3.2 Re-fetch the other real datasets (no proxy locally, ~2.5 GB)

```bash
cd data
git clone --depth 1 https://github.com/s-whynot/CWRU-dataset.git cwru
git clone --depth 1 https://github.com/mathworks/WindTurbineHighSpeedBearingPrognosis-Data.git windturbine
# MFPT: mathworks/RollingElementBearingFaultDiagnosis-Data (train_data/ + test_data/)
# SEU:  cathysiyu/Mechanical-datasets → data/seu/{bearingset|gearset}_<name>.csv
```

Give each the same treatment: catalog gate + dev/sealed split file.

### 3.3 Expansion datasets (fetch opportunistically, they unlock Stage ≥6 claims)

| Dataset | Why | Where |
|---|---|---|
| Paderborn KAt | REAL (run-in) damages + artificial; motor current channel; the honest generalization test | Paderborn university page (free registration) |
| XJTU-SY | 15 run-to-failure bearings — severity/RUL with N ≫ 1 | GitHub mirrors |
| NASA IMS | classic run-to-failure, 3 campaigns | NASA PCoE archive |
| FEMTO PRONOSTIA | run-to-failure, PHM12 challenge protocol | NASA PCoE / FEMTO |
| HUST, JNU, UORED-VAFCLS | breadth for the DG harness | GitHub |
| DG-PHM harness | published cross-domain baselines to compare against | github.com/CHAOZHAO-1/DG-PHM |

---

## 4. Stage 2 — The honest baseline table (before touching our stack)

Build `shorcm/baselines.py` + `scripts/100_baselines.py`. Each baseline runs
blind, through the adapters, on dev-real of every dataset. This table is the
reference point for every future claim, and it also directly measures how much
headroom our stack has.

- **B1 — Classical envelope analyzer (the 1990s bar).** Spectral kurtosis /
  kurtogram band selection (implement the fast kurtogram; keep our 4-band
  max-kurtosis as a comparison point) → band-pass → Hilbert envelope →
  envelope spectrum → peak test at nominal defect orders (BPFO/BPFI/BSF/FTF ±
  tolerance) with a noise-floor significance test → decision rule. Report
  per-class capture AND false-fire on normals (A11).
- **B2 — Cyclic spectral coherence.** Fast-SC (Antoni's fast algorithm) →
  CSCoh map → integrate coherence along spectral axis at candidate cyclic
  frequencies → same peak test. This is the field's reference tool (audit O1).
- **B3 — ERM ML baseline.** LightGBM + a small 1-D CNN on standard features
  (time stats, band RMS, envelope-spectrum bins, order spectra when speed
  known nominally), trained per DG-PHM protocol on source datasets,
  recording-level splits. This tells you what boring ML achieves.
- **B0 — Chance + prior** (always report).

Pre-register (A4) before running: "our frozen v5 cascade beats B1 on MAFAULDA
dev-real 6-way macro-F1, blind" — whatever the verdict, ledger it.

---

## 5. Stage 3 — MAFAULDA reverse-validation gate on frozen v5 (the founding experiment)

API-only, no refitting, blind (A1). Run the frozen cascade over MAFAULDA
dev-real, all four phase variants where applicable (tacho→oracle column only,
onex-Hilbert, harmonic-comb, nominal-filename-speed as a prior-only variant):

Report per class × speed band, with CIs (A9):

1. **O1 speed:** top-1/top-3 octave-corrected hit rate vs tacho truth;
   calibration (ECE) of the confidence head; abstention behavior.
2. **O2 patterns:** ledger capture of expected orders per fault class, WITH
   matched false-fire on normals (A11), at the system's own speed estimate
   (and separately at oracle speed — labeled ORACLE — to decompose front-end
   vs back-end failure, the decomposition Phase 2 never did).
3. **O3 fault:** 6-way macro-F1 (normal, imbalance, h-misalignment,
   v-misalignment, underhang, overhang) for rules arm, ML arm, and (new)
   their disagreement rate; abstention-coverage curve.
4. **O4 severity:** Spearman within class (imbalance grams; misalignment mm;
   bearing severity if defined).
5. **Sim-to-real gap taxonomy:** for every failure, classify: front-end
   (speed), feature-shift (units/scale), physics-missing (SimForge lacks the
   phenomenon), or label-mismatch. This taxonomy drives Stage 4/5.

Compare everything against the Stage-2 baseline table. Expected honest outcome
based on Phase 2 history: envelope/bearing decent, blind speed poor, severity
weak. Do not massage; ledger it. **This single experiment — frozen v5 vs
baselines on MAFAULDA, blind, with CIs — is the most valuable artifact the
program has never produced.**

---

## 6. Stage 4 — Synthetic-data flaw diagnosis (measure, don't assume)

The founding suspicion of Phase 3: **SimForge itself may be flawed.** Measure
it three ways before fixing anything:

1. **Discriminator gate (finally build it — PHASE2 §, audit O8).** Train a
   gradient-boosted classifier (and a small CNN on log-spectrograms) to
   distinguish SimForge records from dev-real records, per dataset,
   healthy-vs-healthy and per fault class. Realism target: AUC in **0.5–0.8**.
   AUC ≈ 1.0 = simulator trivially distinguishable (it will be, initially).
   Crucially, report **per-feature-family attribution** (SHAP on the
   discriminator): which families give it away — noise floor shape? envelope
   kurtosis? harmonic amplitude ratios? phase coherence? unit scale? This
   ranks the SimForge v6 backlog by evidence instead of taste.
2. **Reverse validation.** Train the same LightGBM architecture on dev-real
   MAFAULDA (recording-level splits), test on SimForge. If real-trained models
   fail on synthetic, the simulator's class-conditional structure is wrong,
   not just its noise floor.
3. **Marginal matching.** Compare distributions (per class, per channel) of
   the actual `LEDGER_FEATURES_V2` + pattern + envelope features between
   SimForge and dev-real: KS distance and quantile plots per feature. Features
   whose marginals don't overlap are features the frozen models learned wrong.

Deliverable: `experiments/simgap/findings.json` + a ranked flaw list with
evidence. Only then start Stage 5.

---

## 7. Stage 5 — SimForge v6: the synthetic-data improvement program

Rules: SimForge may be calibrated on **dev-real healthy records only** (noise
floors, resonance statistics, unit scales). It is NEVER fitted to sealed-real,
and never fitted to real *faulty* class data (that would collapse the
train-synthetic/test-real claim). Every v6 change ships with a mechanism test
and re-runs the discriminator gate to show AUC moving toward the 0.5–0.8 band.

Backlog, roughly priority-ordered (pick by Stage-4 evidence):

**Physical fidelity**

1. **Units + sensor model (A7).** Emit m/s². Model: accelerometer sensitivity
   + mounting resonance (random 8–15 kHz per channel), single-pole high-pass
   (~1 Hz), anti-alias filter, ADC quantization/clipping, sensor noise floor
   in µg/√Hz. Calibrate the overall scale distribution to dev-real healthy RMS.
2. **Transmission path.** Replace the single band-pass with 2–5 modal
   resonances per channel (random frequencies/damping from a fleet
   distribution), channel-correlated but not identical; optional
   **measured-FRF mode**: estimate smoothed FRF magnitudes from dev-real
   healthy spectra (cepstral smoothing) and filter synthetic sources through
   them (healthy-data-only calibration, allowed).
3. **Non-Gaussian, non-stationary noise.** Heavy-tailed bursts (alpha-stable
   or Bernoulli-Gaussian impulses), flow/cavitation-like colored bursts,
   slow noise-floor drift within a record, mains interference with realistic
   harmonic decay. Rationale: every blind band-selector (and our envelope
   channel) is known to be fooled by sporadic non-fault impulses — train
   against them.
4. **Bearing realism upgrades.** (a) Per-impact interval jitter (1–2% random
   slip *per event*, not a deterministic drifted tone — this is what makes
   real bearing lines smear); (b) load-zone amplitude modulation with random
   entry/exit angles; (c) extended-spall double impacts (entry/exit); (d)
   incipient-stage rendering: micro-impacts below the raw noise floor,
   visible only after prewhitening/envelope — gives the early-warning
   training signal Phase 2 lacked; (e) fault-size parameter mapped
   consistently to severity labels.
5. **Speed profiles.** In-record ramps, dwell, coast-down; load steps that
   shift both amplitude and f_e; speed wander with realistic spectra (not
   pure sinusoid) — audit F6.
6. **Load-varying VFD electrical content** (audit F5): f_e tracks load,
   slip-frequency sidebands (induction machines), rotor-bar passing
   signatures, PWM carrier drift. This is a paired simulator+detector blind
   spot that produces false bearing calls in the field.
7. **Gear realism.** Transmission-error variation tooth-to-tooth, hunting
   tooth frequency, ghost components, distributed wear vs local tooth fault
   (different sideband-fan signatures), backlash nonlinearity.
8. **Nonlinear looseness/rub.** Sub/inter-harmonics with cycle-to-cycle
   variability (currently too clean/deterministic), truncation
   (clipping-like) waveforms from rubs.
9. **Multi-fault machines** (audit F9): sample 2-fault combinations
   (imbalance+misalignment, bearing+looseness); emit multi-label ground truth
   — prepares the multi-label heads (idea 14 below).
10. **Microphone channel** for MAFAULDA parity: acoustic path = different
    transfer function + room noise; a cheap extra modality the cascade
    currently ignores.
11. **Sensor-fault classes for hygiene training:** dropouts, bias drift,
    clipping, detached-sensor (all-noise) — so the hygiene layer learns to
    say "sensor problem, not machine problem" (audit F2, S-seams).
12. **Fleet/domain randomization curriculum.** Explicit distributions over
    machine parameters wide enough to cover the real fleets (checked via the
    discriminator per dataset), with a curriculum flag (easy/hard SNR) for
    training schedules.

**Process fidelity**

13. **Sim-gap regression tests.** Every real-data failure becomes a synthetic
    replication: e.g., "CWRU has no 1× line" → generate machines with
    suppressed 1×, require blind speed to degrade gracefully to a wide
    posterior instead of a confident wrong answer. The synthetic corpus
    becomes a growing test suite of real-world pathologies.
14. **Hybrid augmentation (careful, powerful).** Additively mix synthetic
    fault signatures into dev-real *healthy* recordings (real noise + real
    transmission path + synthetic fault). Label leakage check: mixing must be
    energy-calibrated so the classifier can't detect "was mixed" itself —
    verify with a mixed-vs-real discriminator. Never with sealed-real.

---

## 8. Stage 6+ — Algorithm thrusts (each is an iteration family; A3 gates all)

1. **T1 State-space blind speed tracker (the front-end fix — top priority).**
   Bayesian filter over (shaft phase, log-speed, drift). Measurements:
   peak-Shor rational candidates, comb-STFT ridges, envelope defect-comb
   spacings; prior: SpeedBelief posterior + nameplate. Key requirement:
   graceful degradation — output an honest octave posterior with width, never
   a confident point when evidence is thin (that is what failed on CWRU).
   Success: MAFAULDA no-tacho top-1 ≥ 0.8 within 5% (it has a real 1× line,
   unlike CWRU); CWRU either ≥ 0.6 or a MET structural-impossibility contract.
2. **T2 Cyclostationary core + CF-rationalized CSCoh.** Implement fast-SC.
   Novelty to test: apply continued-fraction snapping on the *cyclic
   frequency* axis, where bearing slip families live; price in the false-snap
   rate from `theory.py`. This is where "Shor" earns its name or retires (E3).
3. **T3 Prewhitening + modern band selection.** Cepstral prewhitening and
   MED/MOMEDA before envelope; fast kurtogram (then composite indices) over
   the 4 fixed bands. Expect raw-spectrum capture to recover and incipient
   detection to move earlier on run-to-failure data.
4. **T4 Conformal deployment layer.** Split-conformal prediction sets on the
   fault head; risk-controlled abstention targeting a bounded false-alarm
   rate on healthy fleets; drift detection (ADWIN-class) on ledger invariants
   + hygiene stats in the monitor. Calibration on dev-real only.
5. **T5 Fault-arm fusion.** Stacked meta-classifier over {rules verdict +
   margins, LGBM probas, ledger scalars, sheet-contradiction flags, hygiene
   flags}; must beat max(rules, ML) blind on ≥ 2 datasets or dies.
6. **T6 Simulation-based inference (new flagship idea).** SimForge is a
   likelihood-free simulator — use it as one. Train a neural posterior
   estimator (NPE/SNPE) mapping observed features (or order spectra) →
   posterior over (fault type, severity, speed, machine params). This turns
   diagnosis into calibrated Bayesian inversion of the physics model,
   unifies O1–O4 in one head, and gives "analysis-by-synthesis" verification
   for free: resynthesize the MAP machine and check residual spectra. Gate it
   against B3 and the LightGBM arm like everything else.
7. **T7 SSL pretraining moonshot.** Masked-spectrogram (and masked
   order-spectrum) encoder pretrained on 100k–1M SimForge v6 machines;
   evaluate zero-shot (linear probe, no target fine-tuning) on
   MAFAULDA/CWRU/Paderborn. Nobody has an unbounded labeled physics corpus;
   the best empirical effort (VibFM) has ~400 h. If the probe beats B3, this
   is a publishable result on its own.
8. **T8 Further ideas backlog** (pick up when a thrust parks): test-time
   adaptation (TARD-style, source-free) for the monitor; CS-SHAP-style
   cyclic-domain attributions for the report layer; angular-resampling +
   order-domain CNN; matched-filter banks generated from SimForge per
   hypothesis; multi-label heads (multi-fault); quantile-regression severity
   with monotonicity constraints; RUL heads on XJTU/IMS/FEMTO; mic+vibration
   fusion on MAFAULDA; motor-current fusion on Paderborn (MCSA); optimal
   transport alignment of order spectra across domains; deep ensembles with
   seed/feature diversity for epistemic uncertainty; active-learning loop
   that requests the most informative real labels from Matthias.

---

## 9. The autonomous loop — exact mechanics

**Roles (subagents):** conductor (you) · hypothesis-writer · implementer ·
baseline-runner · **judge** (two-key, A12) · adversary (standing veto:
leakage grep, constant-provenance diff, FP gates, unit checks) · data-steward
(A10) · replicator (every 5 iterations) · reporter.

**Every iteration, strictly in order:**

1. `PREREG` commit — hypothesis, numeric contracts (append to
   `work/contracts.jsonl`), metric_id + population_id, mechanism-test plan.
   Written BEFORE any experiment code runs.
2. Mechanism test (pytest) red → green.
3. Implementation. New constants → provenance table (A6).
4. Dev evaluation: serve-path only (freeze → `analyze_record`), against the
   baseline panel, blind, with CIs.
5. Adversary attack pass.
6. Judge re-derives the verdict from artifacts alone; promotion gate at
   α = 0.05/attempts; ledger via `shorcm.metrics.log_attempt`; append the
   iteration section to `REPORT_SYNTHETIC_PROGRAM.md`; update contracts.
7. `git commit` + `git push` to
   `claude/vibration-grover-ensemble-research-4saxj6` (retry with backoff on
   network errors). Every iteration is pushed — the branch is the lab notebook.

**Budgets & stops:** sealed-real: 3 evaluations total per dataset for the
phase — spend them only at major freezes (suggested: after Stage 3 baseline
comparison; after the front-end fix; at phase end). Two consecutive
non-promoting iterations on a thrust → park the thrust, pick the next. Audit
backlog must be empty before any sealed-real evaluation. If blocked on data
or credentials → write the blocker to `work/blockers.md`, continue on another
thrust, tell Matthias in the session summary.

**Compute discipline:** `OMP_NUM_THREADS=1`, `WORKERS=<cores-1>`; long scripts
must be resumable; feature extraction cached per (record, extractor-version).

---

## 10. Pre-registered Phase 3 exit criteria (already in contracts — do not edit)

- **E1** Blind end-to-end 6-way macro-F1 on MAFAULDA sealed-real beats the
  best baseline (B1/B2/B3) by a CI-separated margin.
- **E2** CWRU blind speed ≥ 0.6 top-1, or a MET structural-impossibility
  contract with its refutation experiment actually run.
- **E3** CF-rationalized cyclic analysis (T2) beats B2 on ≥ 2 real datasets,
  else Shor branding is retired program-wide.
- **E4** Conformal layer demonstrates target false-alarm control on healthy
  dev-real fleets, N ≥ 30, CI reported.
- **E5** Contracts tally published, zero retroactive edits, replication agent
  reproduces the final headline table from frozen artifacts.

---

## 11. Suggested calendar (adapt, don't worship)

| Block | Content |
|---|---|
| Iteration 33 | Stage 0 + Stage 1 (env, gates, vault splits) + A2/A4/A5/A6/A9 plumbing with tests |
| Iter 34–35 | Stage 2 baseline panel (B1, B2, B3) on MAFAULDA + CWRU/MFPT/SEU/WT dev-real |
| Iter 36 | Stage 3: frozen v5 vs baselines on MAFAULDA dev-real, blind, full report |
| Iter 37 | Stage 4: discriminator gate + reverse validation + marginal matching |
| Iter 38–41 | Stage 5: SimForge v6 items ranked by Stage-4 evidence; retrain → v6 models; re-gate |
| Iter 42–45 | T1 speed tracker vs baselines; first sealed-real spend when it wins on dev |
| Iter 46+ | T2 cyclostationary, T3 prewhitening, T4 conformal, then T5–T8 by promotion record |

---

## 12. Kickoff prompt (paste into the local Claude Code session)

> Read shor-cm/HANDOFF_PHASE3_LOCAL.md fully, then PHASE3.md,
> REVIEW_EXTERNAL_AUDIT.md, and the tail of work/loop_ledger.jsonl. You are
> the conductor of Phase 3, resuming at iteration 33 on this machine (venv at
> ./venv, data under ./data, MAFAULDA already present locally). Verify G0
> (pytest green, numpy>=2, frozen v5 model hashes recorded), then execute the
> stages of HANDOFF_PHASE3_LOCAL.md in order under the Phase 3 constitution:
> blind end-to-end real gates only, dev/sealed vault split committed once,
> external baseline panel before any claim, contracts.jsonl accounting,
> serve-path metrics, CIs on every real number, two-key promotion with a
> separate judge subagent, adversary veto, ledger every attempt, append every
> iteration to REPORT_SYNTHETIC_PROGRAM.md, commit and push every iteration
> to branch claude/vibration-grover-ensemble-research-4saxj6. PRIORITY ONE is
> the MAFAULDA program: catalog gate, vault split, baseline panel, then the
> frozen-v5 reverse-validation gate — blind, tacho as oracle judgment only.
> Then the sim-gap diagnosis (discriminator gate) and SimForge v6 fixes ranked
> by its evidence, then thrust T1 (state-space speed tracker). Never fit
> SimForge or any calibration to sealed-real or to real faulty-class data.
> Park a thrust after two non-promoting iterations and continue with the next.
> Stop and summarize for Matthias only when blocked on all thrusts or a
> sealed-real budget decision is needed.

---

## 13. Final orientation

The single most important cultural instruction, inherited from the audit:
**a FAIL measured honestly is worth more than a PASS measured generously.**
Phase 2's genuinely good work (envelope channel, governance, theory layer) is
trusted today precisely because its failures were ledgered. Phase 3 exists to
extend that honesty to the two places it lapsed: real-data blindness and
external baselines. Measure first, fix second, brand last.
