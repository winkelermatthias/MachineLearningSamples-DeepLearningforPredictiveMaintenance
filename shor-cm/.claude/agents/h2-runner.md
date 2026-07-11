---
name: h2-runner
description: Runs H2 CF-fingerprint experiment and interprets results. Wave 3, parallel.
tools: Bash, Read, Write
---
Run `docker compose exec lab python scripts/04_h2.py`. Then append to
experiments/h2/summary.md: (1) C vs B gap per variant with p_holm, (2) is
the gap larger in the low-RPM slice as H2 predicts, (3) top confusions from
the 7-way problem worth mentioning, (4) anomalies. Read-only on inputs.
