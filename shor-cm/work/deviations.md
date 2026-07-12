# Deviations

- 2026-07-11: Docker unavailable in the cloud sandbox; all runs executed in a local venv (python 3.11.15, numpy/scipy/pandas/sklearn/lightgbm from requirements.txt), per CLAUDE.md fallback rule.
- MAFAULDA real-data waves (1,2) not run in this session: this session executes the SYNTHVAL + Phase-2 synthetic program only. Real-data transfer measurement is the next scheduled wave.
- 2026-07-11: MAFAULDA mirror unreachable from this environment: proxy returns 403 host_not_allowed for www02.smt.ufrj.br (URL tried: http://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/normal.zip). Real-data waves blocked until the host is added to the network egress allowlist. No substitute data used, per CLAUDE.md.

## 2026-07-12 — real-data acquisition (iteration 23)
- MAFAULDA (www02.smt.ufrj.br): STILL blocked, proxy 403 host_not_allowed
  (re-probed today, http and https). Never substituted per CLAUDE.md; the
  transfer gate below uses OTHER real datasets, clearly labeled.
- Reachable real mirrors via raw.githubusercontent.com (only open host):
  - MFPT bearing set: mathworks/RollingElementBearingFaultDiagnosis-Data
    (CC BY-NC-SA 4.0, owner Eric Bechhoefer) — 20 records, 25 Hz shaft,
    NICE bearing, known BPFO/BPFI orders. -> data/mfpt/ (gitignored)
  - SEU DDS gearbox: cathysiyu/Mechanical-datasets — 20 files (bearingset
    + gearset, 20/30 Hz), fs 5120. -> data/seu/ (gitignored)
    QUIRK: internal Title of bearingset ball_20_0 reads "outer_20_0";
    filenames treated as canonical labels, titles recorded per row.
- CWRU mirrors probed (s-whynot, yyxyz, XiongMeijing, Xiaohan-Chen,
  zhangxiaoli73): no raw-accessible .mat paths found; not pursued today.
- Relos MCP: DROPPED per Matthias 2026-07-12.
