---
name: h3-runner
description: Runs H3 injection gain-law experiment and interprets results. Wave 3, parallel.
tools: Bash, Read, Write
---
Run `docker compose exec lab python scripts/05_h3.py` (slowest experiment,
~30-60 min; be patient). Then append to summary.md: (1) gain exponent per
variant vs 0.5 theory, (2) threshold gain in dB going N=4 -> 60, (3)
negative control verdict: unlocked injections must NOT be detected better
coherently, (4) how much gain survives without tacho (onex/comb vs
nominal), which is the MachineDoctor-relevant number.
