# Deviations

- 2026-07-11: Docker unavailable in the cloud sandbox; all runs executed in a local venv (python 3.11.15, numpy/scipy/pandas/sklearn/lightgbm from requirements.txt), per CLAUDE.md fallback rule.
- MAFAULDA real-data waves (1,2) not run in this session: this session executes the SYNTHVAL + Phase-2 synthetic program only. Real-data transfer measurement is the next scheduled wave.
- 2026-07-11: MAFAULDA mirror unreachable from this environment: proxy returns 403 host_not_allowed for www02.smt.ufrj.br (URL tried: http://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/normal.zip). Real-data waves blocked until the host is added to the network egress allowlist. No substitute data used, per CLAUDE.md.
