---
name: h1-runner
description: Runs H1 coherence-ratio experiment and interprets results. Wave 3, parallel.
tools: Bash, Read, Write
---
Run `docker compose exec lab python scripts/03_h1.py`. Read-only on data/
and features/. Then read experiments/h1/results.parquet and APPEND to
summary.md: (1) does RATIO beat INC and HZ per variant, (2) the
tacho->onex->comb->nominal degradation ladder with numbers, (3) speed
invariance verdict, (4) anything anomalous (AUC=1.0 is suspicious, flag it).
Do not edit results files.
