# HANDOFF — continue shor-cm on a local PC with Claude Code

State as of iteration 32 (branch
`claude/autonomous-research-synthetic-data-bq07j6`, commit `6e2390a`):
79 tests green, models frozen at v5 (certified: speed top-1/top-3
0.614/0.837, ECE 0.038 ≤ 0.05 MET, blind 6-way fault 0.794,
misalignment subtype 0.913, tracking FA 0.044). Four real datasets
gated (MFPT, SEU, CWRU, wind-turbine run-to-failure). Full narrative:
`REPORT_SYNTHETIC_PROGRAM.md`. Every attempt: `work/loop_ledger.jsonl`.
System audit: `work/audit_iter28.md`. Loop constitution: `CLAUDE.md`
Wave 6 + `EXPANSION.md`; blind-cascade objectives: `PHASE2.md`.

## 1. Local setup (one time, ~10 min)

```bash
# EITHER unpack the delivered zip into a folder, OR clone fresh:
git clone -b claude/autonomous-research-synthetic-data-bq07j6 \
  https://github.com/winkelermatthias/MachineLearningSamples-DeepLearningforPredictiveMaintenance.git
cd MachineLearningSamples-DeepLearningforPredictiveMaintenance/shor-cm

python3 -m venv venv                  # python 3.10+
venv/bin/pip install -r requirements.txt
OMP_NUM_THREADS=1 venv/bin/python -m pytest -q tests/   # expect: all green
```

## 2. Re-fetch the real datasets (data/ is gitignored, ~2.5 GB)

```bash
mkdir -p data && cd data
git clone --depth 1 https://github.com/s-whynot/CWRU-dataset.git cwru
git clone --depth 1 \
  https://github.com/mathworks/WindTurbineHighSpeedBearingPrognosis-Data.git windturbine
mkdir mfpt && cd mfpt   # 20 files; names in scripts/73_real_transfer.py
# MFPT via mathworks/RollingElementBearingFaultDiagnosis-Data
# (train_data/ + test_data/ .mat files); SEU via
# cathysiyu/Mechanical-datasets gearbox/{bearingset,gearset}/*.csv
# -> data/seu/{bearingset|gearset}_<name>.csv  (see adapters.py)
cd ../..
# MAFAULDA (the founding target, never yet gated): download from your
# Google Drive folder to data/mafaulda/ preserving the per-class
# layout (normal/, imbalance/, horizontal-misalignment/, ...).
# On a local PC there is NO proxy restriction - the original mirror
# also works: http://www02.smt.ufrj.br/~offshore/mfs/page_01.html
```

## 3. Kick off the autonomous session

From `shor-cm/`, start Claude Code and paste this prompt:

> Read CLAUDE.md, PHASE2.md, EXPANSION.md, HANDOFF_LOCAL.md,
> REPORT_SYNTHETIC_PROGRAM.md and the tail of work/loop_ledger.jsonl.
> You are resuming the autonomous research loop at iteration 33 on
> this machine (local execution, venv at ./venv, data under ./data).
> Verify G2 first (pytest green). Then proceed fully autonomously,
> long-term iterative, under the standing constitution: mechanism
> test first, pre-registered judgments before any run, promotion
> gates with alpha = 0.05/attempts, ledger every attempt via
> shorcm.metrics.log_attempt, append each iteration to
> REPORT_SYNTHETIC_PROGRAM.md, commit and push each iteration to the
> branch claude/autonomous-research-synthetic-data-bq07j6, stop after
> two consecutive non-promoting iterations and report. PRIORITY ONE:
> if data/mafaulda exists, run the MAFAULDA program — catalog against
> the G1 expected class counts, then the reverse-validation transfer
> gate on the FROZEN v5 models (API-only, no refitting, tacho column
> for judgment only, never as input), including the no-tacho
> variants; report per-class results and the sim-to-real gaps found,
> then fix gaps by improving SimForge/algorithms (never by fitting to
> MAFAULDA — PHASE2 data policy). Otherwise continue the backlog at
> the end of REPORT_SYNTHETIC_PROGRAM.md.

Notes for the local session:
- Everything runs with `OMP_NUM_THREADS=1 venv/bin/python scripts/...`
  and `WORKERS=<cores-1>`; scripts are resumable where long.
- The frozen deployment surface is `Cascade.load("models")` +
  `analyze_record` / `monitor` (see `shorcm/cascade.py` docstring);
  real data enters ONLY through `shorcm/adapters.py`.
- Never weaken a gate to pass it; never edit tests to make them pass;
  kills and fails are ledgered results, not setbacks.
- The seeded-benchmark contract: no new rng draws inside existing
  sample/synth paths (derive new randomness from existing fields —
  see the impulsive-bearing mode for the pattern).
