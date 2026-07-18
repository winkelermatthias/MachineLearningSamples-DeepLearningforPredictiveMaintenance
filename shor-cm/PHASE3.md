# PHASE 3 — next-generation autonomous research constitution

Supersedes the Wave-6 loop (`EXPANSION.md`) and extends `PHASE2.md`.
Motivated point-by-point by `REVIEW_EXTERNAL_AUDIT.md` (M1–M13, O1–O12).
Everything in Phase 2 that is not amended here still applies (mechanism
test first, pre-registered judgments, α = 0.05/attempts promotion,
append-only ledger, kills as results, adapters-only real data,
seed 20260709).

The Phase 3 question, one paragraph: Phase 1–2 built a well-governed
machine that validates itself on its own simulator and fails blind on
real data. Phase 3 inverts the center of gravity: **every claim is an
end-to-end blind claim on real data, beaten against external baselines,
with distribution-free uncertainty** — and the simulator graduates from
"training set" to two new roles it is actually suited for: a realism-gated
physics prior and an unbounded pretraining corpus.

---

## 1. Constitutional amendments (structural fixes)

### A1 — End-to-end-only real gates (fixes M1)
No real-data gate may condition on metadata speed, tacho, or filename
nominal speed, except in a clearly-labeled ORACLE diagnostic column.
Every promoted real-data number comes from `Cascade.analyze_record` /
`monitor` blind. Reports must show the composed chain
(speed → pattern → fault → severity) jointly; a stage may be reported
in isolation only next to its composed number.

### A2 — Real-data vault (fixes M2)
Each real dataset is split ONCE, at catalog time, by run-id hash:
`dev-real` (open, tune/refit freely) and `sealed-real` (≥ 40%,
budget: 3 evaluations per phase, every evaluation ledgered including
the ones we wish we hadn't spent). Re-freezing a model and re-scoring
the same sealed gate is a constitution violation, not a v-bump.
The version that touches sealed-real is final for that budget slot.

### A3 — Mandatory external baseline panel (fixes M3)
A claim about the signal path promotes only if it beats, through the
same adapters and the same blind protocol:
  B1 classical envelope analysis: SK/kurtogram band selection +
     envelope spectrum peaks at nominal defect orders;
  B2 cyclic spectral coherence (fast-SC) peak detection;
  B3 ERM LightGBM/1-D-CNN on standard features (DG-PHM-style).
Baselines are first-class code with their own tests, run by a separate
agent from the proposer. "Beats a 1990s envelope analyzer, blind" is
the minimum bar for any Shor-branded component.

### A4 — Contract accounting (fixes M4)
`work/contracts.jsonl`, append-only: every pre-registered numeric
target with its eventual verdict — MET / MISSED / RETIRED(new-id).
A missed contract stays MISSED forever. Reframing ("structural limit")
is allowed only by opening a NEW contract containing a falsifiable
prediction plus the experiment that would refute it, linked to the
missed one. Reports must show the running MET/MISSED tally at the top.

### A5 — Metric constitution (fixes M5)
Every scoreboard number carries `population_id` (generator version +
config hash) and `metric_id` (frozen definition). Cross-population or
cross-definition deltas are forbidden in promotion decisions and in
headline narratives. Trend plots may span populations only with a
visible population-change marker.

### A6 — Constant-provenance rule (fixes M6)
Every numeric threshold in `shorcm/` moves to a single provenance table
(`shorcm/constants.py`): value, unit, fit-source (THEORY / SYNTH /
DEV-REAL / HAND), and date. HAND constants touched after any real-data
contact are flagged by a CI test and must be either re-derived from
theory/synth or refit on dev-real by a script. The adversary's standing
audit includes: diff constants against real-gate dates.

### A7 — Physical-units contract (fixes M7, O9)
SimForge v6 emits m/s² with a documented sensor model; `hygiene.py`
converts and unit-tags all inputs; absolute features are banned unless
unit-tagged. A units round-trip test pins the contract.

### A8 — Serve-path metrics only (fixes M8)
The promotion metric IS the certification metric: all dev evaluation
runs through the frozen-artifact API path (train → freeze → score via
`analyze_record`), never through in-memory OOF. Mechanism pins are
required at promotion time, not at audit time — an unpinned mechanism
cannot promote.

### A9 — Uncertainty floor (fixes M9)
Every real-data number ships with N and a 95% CI (Wilson or BCa
bootstrap, in `metrics.py`). Claims from N < 30 recordings are
auto-labeled ANECDOTE by the report generator. Run-to-failure timing
claims require ≥ 3 independent sequences before leaving ANECDOTE.

### A10 — Data before iteration (fixes M10)
No more than 2 consecutive iterations may run without touching real
data. If a priority dataset is blocked, iteration 1 of the block is
spent on the unblock path (mirror, local handoff, manual fetch request
to Matthias) — not on more synthetic optimization.

### A11 — Adversarial capture metrics (fixes M11)
Every capture/detection metric must report, in the same table, its
false-fire rate on healthy/baseline records and its expected
false-snap rate under the program's own theory (SYNTHVAL C3). A capture
gate without a matched false-positive gate cannot PASS.

### A12 — Two-key promotion (hardens the loop)
The implementing agent never judges its own work. A separate judge
agent re-computes the promotion decision from result artifacts alone
(no access to the implementer's narrative). Disagreement = no
promotion + ledger entry. Every 5 iterations, a replication agent
re-derives the current headline table from frozen artifacts only;
non-reproducible numbers are retracted in the ledger.

### A13 — Honest branding (fixes M13)
All documents and code comments say "quantum-inspired classical".
Claims of novelty are positioned against CSCoh/kurtogram/Andgram, not
against quantum computing. The Simon/QPE/DJ/QEC backlog
(`EXPANSION.md`) is parked behind gate: "peak-Shor beats B2 blind on
two real datasets."

---

## 2. Research thrusts (the science, priority order)

### T1 — Fix the front-end: joint state-space speed tracker (O3; attacks M1 root cause)
Reformulate blind speed as Bayesian filtering over instantaneous shaft
phase: state = (phase, log-speed, drift), measurements = peak-Shor
rational candidates + comb-STFT ridges + envelope defect-comb spacing,
prior = SpeedBelief posterior + nameplate. Deliverable: a tracker that
degrades gracefully to "octave posterior with honest width" instead of
confidently wrong point estimates. Real gates: CWRU C1 ≥ 0.6 blind or a
falsifiable structural-impossibility contract (A4); WT W4 ≥ 0.6;
MAFAULDA no-tacho variants.

### T2 — Cyclostationary core (O1)
Implement fast-SC/CSCoh as a first-class stage. Novel contribution to
test: **CF-rationalized cyclic coherence** — apply peak-Shor
continued-fraction snapping to the cyclic-frequency axis (where slip
families live) instead of the raw spectrum, with the false-snap theory
priced in. Benchmark head-to-head vs B1/B2 and vs the current
coherent/incoherent split. This is where "Shor" either earns its name
or is retired with a ledger entry.

### T3 — Prewhitening + modern band selection (O2, O4)
Cepstral prewhitening and MED/MOMEDA ahead of envelope; kurtogram
(then Andgram-style composite index) replacing the 4 fixed bands.
Expected effect: raw-spectrum J3 recovers without the envelope rescue;
incipient-fault capture on run-to-failure sets moves earlier.

### T4 — Conformal deployment layer (O5, O10)
Split-conformal prediction sets over the fault head and
risk-controlled abstention (target: bounded false-alarm rate on
healthy fleets — the metric operators actually buy). Add drift
detection on the monitor's feature stream (ADWIN-class on ledger
invariants + hygiene stats). Calibration sets come from dev-real only.

### T5 — SimForge as pretraining corpus: vibration SSL encoder (O7)
Masked-spectrogram (or masked order-spectrum) encoder pretrained on
100k–1M SimForge v6 machines, evaluated ZERO-SHOT (linear probe, no
target fine-tuning — DG not DA) on Paderborn/XJTU/CWRU/MAFAULDA.
Gate first (A3): must beat B3-ERM trained on handcrafted features.
This is the highest-upside bet: nobody has an unbounded *labeled*
physics corpus; VibFM has ~400 empirical hours.

### T6 — Learned fault-arm fusion (M12, O11)
Stacked meta-classifier over {rules verdict + margins, LGBM probas,
ledger scalars, sheet-contradiction flags, hygiene flags}, trained on
synth, calibrated on dev-real, wrapped in T4 conformal sets. Kill
criterion: must beat max(rules, ML) blind on two real datasets.

### T7 — SimForge v6 realism program (O8, O9)
(a) Build the PHASE2 discriminator gate at last: train a classifier
sim-vs-dev-real; realism target = AUC in 0.5–0.8 band per dataset,
reported per feature family so gaps are actionable.
(b) Close flagged gaps: load-varying VFD lines, in-record speed ramps,
multi-fault machines, multi-resonance transmission paths (2–5 modes +
mounting resonance), burst/non-Gaussian noise.
SimForge is still never fitted to sealed-real.

### T8 — Benchmark expansion (O6)
Priority: MAFAULDA (founding target, PRIORITY ONE the moment data
lands), Paderborn KAt, XJTU-SY, IMS, FEMTO PRONOSTIA, HUST; port the
DG-PHM harness so results are comparable to the published DG
literature. Each dataset gets: catalog → dev/sealed split (A2) →
baseline panel run (A3) → then and only then our stack.

---

## 3. The loop, mechanized

Roles (subagents): conductor · hypothesis-writer · implementer ·
baseline-runner · judge (two-key, A12) · adversary (standing veto) ·
data-steward (A10) · replicator (every 5 iters) · reporter.

Each iteration, strictly:
1. `PREREG` commit: hypothesis, contracts (A4), metric_id/population_id
   (A5), mechanism-test plan — before any experiment code runs.
2. Mechanism test written and red→green (pytest).
3. Implementation; constants land in the provenance table (A6).
4. Dev evaluation through the serve path (A8), against the baseline
   panel (A3), with CIs (A9), blind (A1).
5. Adversary attack (leakage grep, constant-provenance diff,
   false-positive gates A11, unit checks A7).
6. Judge re-derives verdict from artifacts (A12); promotion gate with
   shrinking α; ledger + contracts + report appended; commit + push.
Stop rules: 2 consecutive non-promoting iterations per thrust parks
the thrust; sealed-real budget exhaustion ends the phase's claims;
audit backlog must be empty before any sealed-real evaluation.

Phase-3 exit criteria (pre-registered here, A4-tracked):
- E1: blind end-to-end fault macro-F1 on MAFAULDA sealed-real beats
  baseline panel best by a CI-separated margin.
- E2: CWRU blind speed ≥ 0.6 top-1, or a MET structural-impossibility
  contract with its refutation experiment run.
- E3: peak-Shor (as CF-rationalized CSCoh, T2) beats B2 on ≥ 2 real
  datasets, else the Shor branding is retired program-wide.
- E4: conformal layer demonstrates target false-alarm control on a
  healthy fleet (dev-real held-out), N ≥ 30, CI reported.
- E5: contracts tally published with zero retroactive edits.

## 4. Iteration-33 kickoff order
1. A2/A4/A5/A6/A9 plumbing (one iteration, mechanical, fully testable).
2. Baseline panel B1/B2/B3 on CWRU + MFPT + SEU + WT, blind (this also
   produces the first honest table for the record).
3. T1 tracker spike vs those baselines.
4. Data-steward unblocks MAFAULDA/Paderborn/XJTU in parallel (A10).
