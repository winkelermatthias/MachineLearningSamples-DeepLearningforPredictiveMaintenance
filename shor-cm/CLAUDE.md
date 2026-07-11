# SHOR-CM AutoLab

Autonomous experiment: Shor-inspired coherent order analysis for machinery
fault diagnosis, tested on MAFAULDA, including no-tacho variants.

You (Claude Code) are the CONDUCTOR. Subagents (`.claude/agents/`) do the
work. You never skip a gate. All numeric claims come from result files.

## Execution environment

All python runs inside Docker for reproducibility:

```bash
docker compose up -d
docker compose exec lab python scripts/XX_*.py
```

If Docker is unavailable, fall back to a local venv from requirements.txt
and record the substitution in `work/deviations.md`.

Data lives in `./data` (gitignored). Peak disk ~45 GB during extraction,
~12 GB steady state. Check free disk before wave 1.

## Waves and gates. Run strictly in order.

### Wave 0, infra (you, no subagent)
- `docker compose build && docker compose up -d`
- `docker compose exec lab python -c "import numpy,scipy,pandas,sklearn,statsmodels"`
- Check disk: need 50 GB free. Write `work/gates/G0.json` with versions.

### Wave 1, data (subagent: data-acquirer)
- Runs `scripts/00_download.py` then `scripts/01_catalog.py`.
- G1 gate (you verify): catalog exists; class counts roughly
  normal 49, horizontal-misalignment 197, vertical-misalignment 301,
  imbalance 333, underhang 558, overhang 513 (verify against the dataset
  page; treat published counts as expected, not gospel); 8 channels
  everywhere; runs with |rate_hz - f_nom|/f_nom > 0.15 are < 2%.
- If the mirror is down: STOP, report URLs tried. Never substitute data.

### Wave 2, signal lib (subagent: signal-engineer)
- Library and tests already exist and pass. The agent's job here is only
  needed if a later wave finds a defect: it fixes `shorcm/` and re-runs
  `pytest -q tests/`. G2 = all tests green. 02_features.py enforces this
  gate itself and will refuse to run otherwise.

### Wave 3, features then experiments
- First: `scripts/02_features.py` (parallel, ~30-60 min for 1951 runs x 4
  variants on 8 cores; resumable, safe to rerun).
- Then run THREE subagents in parallel via the Task tool: h1-runner,
  h2-runner, h3-runner. Each runs its script, inspects output, writes a
  short interpretation appended to its summary.md. They are read-only on
  /data and /features.
- G3: each experiments/hX/ has results.parquet + summary.md.

### Wave 4, adversary (subagent: adversary)
- `scripts/06_adversary.py`. Exit code 1 = CRITICAL findings = report
  blocked. If the tacho_quality confound fires: rerun H1 with
  tacho_quality added as a feature to the HZ baseline AND with runs
  matched on quality deciles; document whether RATIO still wins.

### Wave 5, report (subagent: reporter)
- `scripts/07_report.py`, then the agent reviews the HTML, fills the
  recommendation line in report/decision_memo.md (SHIP / ITERATE / KILL
  with one sentence), verifies every number traces to a file.

## Retry policy
Max 2 retries per wave. On second failure, stop and summarize blockers
for Matthias. Never weaken a gate to pass it. Never edit tests to make
them pass; fix the library.

## Non-negotiables
- Seed 20260709 everywhere.
- Grouped splits by speed band only (shorcm/splits.py). Random row splits
  are leakage and the adversary will catch you.
- SELF_REF_MASK must be applied for signal-derived phase variants.
- Report numbers only from parquet/json outputs.

## The scientific question, one paragraph for context
Shor's algorithm = period finding + phase-coherent interference +
continued-fraction rationalization. Mapped to machines: shaft phase is the
modular clock; 5-rev blocks summed complex-coherently make shaft-locked
components (imbalance, misalignment, gears) interfere constructively while
bearing tones (1-2% slip, not phase-locked) average away; the
coherent/incoherent ratio is therefore a physics-grounded classifier
feature; continued fractions snap noisy peak orders onto exact rationals
p/q (0.4031 -> 2/5 cage, 0.503 -> 1/2 looseness) while true bearing orders
(2.998x) refuse to snap, which is itself the signature. The no-tacho
variants (onex, comb, nominal filename speed) test whether this survives
without a keyphasor, which is the deployment-relevant case for
MachineDoctor.

## Wave 6, improvement loop (after wave 5, optional, budget-gated)

Additive extension, see EXPANSION.md for the full constitution. Summary:
- North star: 7-way macro-F1, best NO-TACHO variant, extreme-speed
  hold-out, sealed vault. Dev iterations never touch vault runs.
- Loop: metrics-keeper picks backlog item -> signal-engineer implements
  + mechanism test (pytest green mandatory) -> runner evaluates on dev ->
  metrics-keeper applies promotion_gate (alpha shrinks with ledger
  length) -> adversary spot-audit -> vault eval at most every 5
  iterations, 3 total.
- First loop item is already implemented: scripts/08_h4_amplify.py
  (Grover slip-scan + PPA). Run it as the loop's iteration 1.
- Stop conditions: two flat vault evals, dev-vault overfitting gap, or
  budget. A negative result still produces a report.

## Phase 2 (see PHASE2.md, additive)

Objectives become a blind cascade: O1 speed top-3 with calibrated
confidence, O2 pattern energy ledger, O3 fault mode with abstention, O4
ordinal severity. Data policy INVERTS: train on SimForge synthetic, tune
and calibrate on MAFAULDA dev, final answers from the vault only.
Augmented runs inherit their source run's vault flag and speed-band
group. SimForge is never fitted to MAFAULDA; realism is gated by the
adversary (discriminator band 0.5-0.8, marginal matching, reverse
validation floors). New agent: simforge-engineer.
