# SHOR-CM AutoLab

Shor-inspired coherent order analysis for machinery fault diagnosis,
autonomously tested on MAFAULDA by a Claude Code subagent team.
Includes four phase-reference variants: real tacho, 1x-Hilbert,
harmonic-comb, and nominal speed from the MAFAULDA filename (no tacho).

## Quick start

```bash
cd shor-cm
docker compose build && docker compose up -d
docker compose exec lab pytest -q tests/        # G2 gate, must be green
claude                                          # then: "read CLAUDE.md and run the full pipeline"
```

Manual pipeline (what the agents run):

```bash
docker compose exec lab python scripts/00_download.py     # ~13 GB
docker compose exec lab python scripts/01_catalog.py
docker compose exec lab python scripts/02_features.py     # 30-60 min
docker compose exec lab python scripts/03_h1.py
docker compose exec lab python scripts/04_h2.py
docker compose exec lab python scripts/05_h3.py           # slowest
docker compose exec lab python scripts/06_adversary.py    # exit 1 blocks report
docker compose exec lab python scripts/07_report.py
open report/shor_cm_report.html
```

## Layout
- `CLAUDE.md` conductor instructions, waves, gates
- `.claude/agents/` 7 subagents
- `shorcm/` library (phase refs, spectra, CF snap, features, splits)
- `tests/` 7 synthetic gate tests (G2), run before any real data
- `scripts/` numbered pipeline
- `IDEAS.md` extension backlog

## Status of shipped code
All 7 gate tests pass. An end-to-end smoke test on a synthetic
mini-dataset (scripts/99_smoke.py) exercises catalog -> features -> H1.
Real-data paths (download URL layout, exact class counts, bearing order
constants) carry TODO markers for the data-acquirer agent to verify.
