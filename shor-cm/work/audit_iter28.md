# ADVERSARY AUDIT — iteration 28 ("big look at everything")

Scope: ledger (49 entries through it27/speed_belief), shorcm/{patterns,peakshor,
envelope,speedbelief,cascade,tracker,simforge_v2,adapters,simforge_corpus}.py,
tests/ (T1–T57; T24/T42/T55 absent). Read-only audit; no code touched.
Ranking = impact x likelihood, highest first.

---

## 1. UNPINNED MECHANISMS
Behaviors measured or fixed per the ledger with NO regression test (grep-verified
against tests/*.py).

### U1. `v4_feature_order_defect` (it47, verdict FIXED) — THE recorded near-miss, still unpinned
- File/behavior: `cascade.py:118` builds the serve vector as
  `[led.get(f) for f in SC.LEDGER_FEATURES_V2]` + optional PF_COLS + PF_ENV_COLS;
  training order lives only in `models/meta.json["fault_model"]["features"]`.
  Nothing at load or test time asserts the two agree. The exact class of bug that
  collapsed held-out acc 0.815→0.667 can recur on any refreeze or any edit to
  `LEDGER_FEATURES_V2`/`PF_COLS`/`PF_ENV_COLS`.
- Note: `tests/test_tracker.py:207` (T45) runs `Cascade()` WITH NO frozen models —
  the ML feature path is never executed by any test.
- Test sketch: `Cascade.load("models/")`; assert
  `meta["fault_model"]["features"] == SC.LEDGER_FEATURES_V2`, `features_pf == PT.PF_COLS`,
  `features_pfe == PT.PF_ENV_COLS`, and `fault_model.n_features_ == len(all three)`;
  plus one golden-vector: `analyze_record` on a fixed-seed V2 run reproduces a stored
  `fault_ml` + probability vector within 1e-6.

### U2. Degenerate-candidate confidence (it27 `confidence_isotonic_on_margin`)
- File/behavior: `cascade.py:83-90`. `margin = s[0] - s[1]` with sentinel `-99.0`;
  a drowned record (all-NaN candidates → the null dict with `score: -9`) or any
  single-candidate list yields margin ≈ 90 → isotonic clamps to its MAX fitted
  probability → `speed_confidence ≈ 1.0`, `abstain_speed = False` while
  `speed_hz = NaN`. Calibration contract (ECE 0.026 certified) silently violated
  on exactly the garbage records deployment will bring.
- Test sketch: `analyze_record(white_noise, fs)` with the frozen calibrator →
  assert `abstain_speed` is True or `speed_confidence < 0.5` when
  `len(speed_candidates) < 2` or hz is NaN.

### U3. Frame-arbitration 3x/(1/3)x extension + nameplate prior (it46, "implemented_pending_synthetic_gate")
- File/behavior: `peakshor.py:358` (mlt loop now includes 3.0, 1/3) and
  `peakshor.py:299-301,325-331` (f_nom candidate + soft octave bonus, weight 2.0
  measured ineffective on CWRU). T40 pins only the 2x/0.5x looseness case.
  Nothing pins: a 3rd-harmonic-lattice machine (CWRU normal physics) arbitrated back
  to f0, nor that f_nom enters as candidate, nor the bonus weight's sign/scale.
- Test sketch: synthetic peak set {3f0, 6f0, 9f0} + sheet/passage or meta f_nom≈f0 →
  `estimate_speed_sheet` top-1 within 1% of f0, not 3f0.

### U4. Adaptive alarm gates (it15 `adaptive_alarm_gates`, PROMOTED)
- File/behavior: `tracker.py:227` and `patterns.py:56`
  (`gate = max(3.0, 3*10*std(y[:5]))`). Production path `cascade.py:192` calls
  `trends(adaptive=True)`; grep "adaptive" in tests/ → zero hits. FA improvement
  0.023→0.012 rests on untested code.
- Test sketch: tight-baseline key alarms on a 4 dB rise; noisy-baseline
  (std 0.3 dec) key must NOT alarm on the same rise.

### U5. FIXEDHZ comb-mask concentration gate (it4 GUARDRAIL_FAIL root cause fix)
- File/behavior: `simforge_corpus.py:71-103` `_fixed_bases`: `conc_at(c) > 0.5`
  gate stops a severe bearing's smeared comb from masking its own evidence
  ("driver falling monotonically with severity"). No test touches `_fixed_bases`
  or the masked() path.
- Test sketch: severe tonal bearing run → `ledger()["uns_abs"]` must be
  monotone-nondecreasing vs severity on a 3-point paired sweep.

### U6. it34 isolation-hardening sub-mechanisms (4 of 6 unpinned)
T50 pins the shaft-coincidence cap + band_local overcredit. Still unpinned:
- (a) narrow dry-run 2x probe (`patterns.py:552-560`) — an unconditional wide 2x
  claim swallowing a NEIGHBORING bearing hump; sketch: two humps at o and ~2o±0.1,
  assert both come out as separate NEARRATs.
- (b) extended-margin merge only when RAW edges touch (`patterns.py:494-503`) —
  sketch: two diffuse tones separated by a cold valley → 2 patterns, not 1
  centerless pseudo-hump.
- (c) floor-relative (not e_tot-relative) anti-clutter gate (`patterns.py:536-540`) —
  sketch: small real tone at 0.2% of e_tot on a "big machine" record still emitted.
- (d) bearing (o,2o) pair vs second-shaft lattice coherence discriminator
  (`patterns.py:306-311`, `sp.coh(r) < 0.45` skip) — sketch: drifting pair at
  (3.1, 6.2) must NOT become HARM(3.1).

### U7. Envelope-mode gate inversions (it44, three measured mechanisms)
T54 pins end-to-end capture only. Unpinned individually:
- carrier floor 5.0→2.0 (`patterns.py:421`): BPFI 4.755 ±1x fan must come out as
  SIDEBAND(carrier≈4.755) in env mode — T54's `_env_near` accepts NEARRAT/HARM
  too, so the sideband-specific fix can regress silently.
- hybrid comb lock (`patterns.py:688`, trust comb only within 1.5% of f_hat):
  sketch: impulse-dominated record where the comb locks 10% off → envelope orders
  still within tolerance of the defect order.

### U8. pattern_features / pattern_features_env mapping (it38, it48 PROMOTED)
- File/behavior: `patterns.py:708-756, 953-976`. These ARE the serve-side inputs
  to the frozen v4 models; no test references `pattern_features`. A sign slip in
  `pf_harm1_2x1x` (log10 2x/1x) or a key rename silently skews the certified 0.8153.
- Test sketch: hand-built pats list → exact expected dict (e.g. HARM members
  {1:0.1, 2:0.01} → pf_harm1_2x1x == -1.0).

### U9. Severity head (it41 PROMOTED, deployed)
- File/behavior: `cascade.py:136-139` emits `severity_score` (clip 0..1) from the
  same feats vector. No test loads `severity_lgbm.joblib`; the smoke result
  (mild 0.488 vs severe 0.914) is not pinned; monotonicity guardrail exists only
  in scripts.
- Test sketch: frozen model on two fixed-seed runs (s=0.35 vs s=0.95, same
  machine) → score(severe) > score(mild) + 0.2.

### U10. Misalignment-subtype branch (it29 PROMOTED)
- File/behavior: `cascade.py:145-149` hardcodes `ax_ratio_2 > 0.61` while
  `models/meta.json` carries `subtype_threshold_ax_ratio_2: 0.61` — a refreeze
  that re-tunes the threshold changes meta but not behavior. T44 pins the physics
  (simforge_mc), not the cascade branch.
- Test sketch: assert cascade reads the threshold from meta (or that meta value
  == code constant); angular machine end-to-end → subtype "angular".

### U11. Monitor never applies FrameSelector warmup re-choices / never reframes
- File/behavior: `tracker.py:351` computes `warmup_choices` for retroactive
  re-framing; T37 applies them MANUALLY; `cascade.py:184` (MachineMonitor.feed)
  ignores them, so the first 5 records' tracker history is keyed under provisional
  top-1 frames. `GeneralTracker.reframe` (it27 unit-PROMOTED) is called by no
  production code path — only by T56.
- Test sketch: monitor fed 8 records where record-1 top-1 is an octave alias →
  assert the SHAFT_1 series is not contaminated / re-keyed after warmup.

---

## 2. CONTRACT SEAMS
Places where train/serve or module contracts can silently skew (the it47 LGBM
feature-order incident is the type specimen).

### S1. Positional, meta-flag-gated feature vector (highest risk, recurrence of it47)
`cascade.py:117-130`: three blocks concatenated, blocks 2-3 conditional on
`meta["fault_model"]["features_pf"/"features_pfe"]` truthiness. LightGBM is
positional; there is no length or name check at predict time. Extra twist:
`severity_model` (trained on "C_ledger+pf+pfe") consumes the SAME feats vector
gated by the FAULT model's flags — freeze the fault model without pfe and the
severity model silently gets a shorter/reordered vector. One `assert
len(feats) == model.n_features_` at `Cascade.load` would close the whole class.

### S2. Absolute-amplitude features assume SimForge's unit scale
`simforge_corpus.py:230-240`: a1/a2/a3, e_half, uns_abs, e_tot, floor, gmf_sb_energy
are ABSOLUTE amplitudes; `rules_from_ledger` thresholds are absolute (a1>0.5,
a2>0.35, e_half). Adapters (`adapters.py`) do no unit/scale normalization — Relos
historian samples arrive as-is (g? mm/s? ADC counts?). A gain factor of 10 flips
every rule and shifts the LGBM operating point; nothing detects it. Shares
(pf_*/pfe_*) are scale-free, the 24 ledger features are not.

### S3. Grid-hum mask deletes near-synchronous mains machines' shaft evidence
`peakshor.py:24,66` `_is_hum(tol=1.5)`: peaks within 1.5 Hz of 50/60/100/120 are
excluded from `pair_candidates` AND from `hz_structure_score`. A 2-pole mains
machine at slip ≤ ~2.5% runs 48.5–49.75 Hz (or 58.2–59.7): its 1x — and its 2x
vs 100/120 — is masked as hum. This is the single most common industrial machine
(2900/3500 RPM pumps). SimForge generates exactly this population
(`simforge_v2.py:82-84`, p=1 mains → f0 = lf*(1-slip)), so part of the corpus
speed miss rate (top1 0.586) is likely this seam; on real data it is systematic.

### S4. Hz-tolerance vs order-domain conversion at low speed
`simforge_corpus.py:123-124` `masked()`: ±1.5 Hz around k*base, k<6. At f_hat=5 Hz
that is ±0.3 ORDERS per line x 20+ lines — large stretches of the bearing band
(1.8–16) vanish from uns evidence on slow machines. Same constant, different
regime.

### S5. Two speed frames inside one monitor record
`cascade.py:182-189`: `analyze_record` diagnoses under `est[0]["hz"]` while
energies/tracking use `f_lock` from the FrameSelector. When they disagree (exactly
the octave-flip records the selector exists for), `fault_ml` and `alarms` in the
SAME output dict describe different frames; no flag records the divergence.

### S6. `mode="envelope"` / `env_mode` branches in decompose
`patterns.py:172-210,421,529,597`: three gates invert by flag. The flag rides in
from `decompose_envelope` only; any new caller passing an envelope without
`mode="envelope"` (or vice versa) reproduces the MFPT failure silently (FIXEDHZ
steals the defect rate). The raw/env contract is implicit in the caller, never
validated (e.g. an envelope signal is nonnegative-mean-removed — checkable).

### S7. Kinematic sheet is trusted, never validated
`patterns.py:325-401` seeded fans claim bounded energy on the sheet's say-so
(fused case claims up to 3.2*dmax around the mesh order); `peakshor.py:349-372`
octave arbitration re-frames on sheet_match margin 0.15. A wrong tooth count
(bad registry entry — PHASE2 explicitly lists this) fabricates SIDEBAND(seeded)
patterns and can flip the speed frame. No sheet-vs-evidence contradiction score
is emitted.

### S8. SEU adapter compresses time on dropouts
`adapters.py:109`: `x = x[np.isfinite(x)]` — NaN samples are DELETED, not filled,
shifting all subsequent phase and biasing fs by the dropout fraction. Fine for a
clean mirror, wrong as a pattern for field CSVs.

### S9. `GeneralTracker.reframe` re-keys only part of the vocabulary
`patterns.py:839-856`: HARM re-keyed only when base≈1.0 (a second-shaft
HARM(0.44) keeps its stale identity after a flip); HALFHARM and FIXEDHZ untouched
(HALFHARM's members genuinely move under a 2x correction — the half ladder
becomes a quarter ladder and will re-register as a NEW identity, breaking series
continuity that reframe exists to preserve).

### S10. PatternTracker f_ref locks to the first record's estimate
`tracker.py:178-180`: `f_ref = f_hat` of record 1; the omega² normalization of
SHAFT_1 (`tracker.py:203`) is then wrong by 4x forever if record 1 was an octave
error. PatternTracker has no reframe path at all (only GeneralTracker does).

### S11. Envelope phase subsampling vs decimate
`patterns.py:692` passes `ph[::q][:len(env)]` while `decimate(..., ftype="fir")`
imposes a group delay on env — a constant few-sample phase skew between reference
and signal. Small at current q; grows with fs (q=16 at 97.6 kHz WT data).

### S12. estimate_speed_shor vs estimate_speed_sheet output contracts differ
`peakshor.py:241` (no "score" key) vs `peakshor.py:375` (score present). Cascade
depends on "score" via `.get("score", -9.0)`; any caller swap to the plain
estimator degrades margins silently to the -9 sentinel.

---

## 3. EXPECTED FUTURE ISSUES (real deployment data)
Ranked by impact x likelihood; overlaps with §2 noted.

### F1. Velocity-vs-acceleration units / arbitrary gain (very high, ~certain)
Field sensors deliver mm/s velocity, g, or unscaled counts. (a) Every absolute
feature and rule threshold shifts (S2). (b) On velocity data the HF resonance
band barely exists → `envelope.py:34-40` max-kurtosis picks a noise band →
phantom NEARRAT in the envelope → false bearing calls. (c) ISO-anchored severity
(O4 plan) is unit-DEFINED — meaningless without unit metadata. Needed: a unit
contract in adapters + a spectral-slope sanity check (velocity spectra fall ~1/f
against acceleration).

### F2. NaN / dropouts / saturation in the waveform (high, very likely)
No NaN guard anywhere in `analyze_record` — one NaN propagates through rfft and
every output silently becomes NaN/garbage rather than "unknown". Dropout steps
and clipping flats are impulse trains: the envelope channel's kurtosis band
selector will LOCK ONTO them (they are the most leptokurtic thing in the record)
→ false impulsive-bearing evidence. Saturation additionally raises odd harmonics
of the dominant tone (fake misalignment/looseness flavor). Cheap gates: finite
check, clip fraction, dropout-run detector, before analysis.

### F3. Near-synchronous mains machines (high impact, high likelihood) — see S3
2-pole 50/60 Hz machines: shaft evidence deleted by the hum mask; additionally at
slip <1% and short records the wander is unresolvable (`patterns.py:190`
`_crystal` returns False below 3 Hz-bins of smear) so `shaft_coincident` exempts
the line — 1x vs hum genuinely undecidable and currently resolved silently, not
flagged. Expect systematic octave/speed failures and imbalance blindness on the
most common machine class.

### F4. Wrong/mislabeled sample rate & resampled/decimated inputs (high, likely)
Historian exports resample casually. Order-domain diagnosis survives (peaks and
f_hat scale together) but: reported speed_hz is wrong by the fs ratio; the
50/60/100/120 FIXEDHZ seeds land on wrong bins so real hum leaks into the pattern
ledger as crystal TONE/NEIGHBOR; envelope band candidates (`envelope.py:18-25`)
mismatch the true bandwidth. A "grid-hum-found-at-expected-Hz" check is a free
fs validator — its absence is itself a finding.

### F5. VFD electrical lines on driven equipment → false bearings (high, likely)
PHASE2 §2.2 specifies the jitter-sign (anti-correlation) discriminator for
ELEC-vs-NEARRAT; grep shows it is NOT implemented anywhere in shorcm/. A VFD
line under load-varying f_e is not crystal in Hz (defeats `_crystal`), not
shaft-coherent (defeats the coh gate), near-rational and drifting — it satisfies
every current bearing signature. SimForge only generates constant-f_e VFD
(`simforge_v2.py:88`), so the corpus cannot catch it: simulator blind spot +
detector blind spot aligned.

### F6. Speed ramps inside one record (medium-high, very likely at start/stop)
`phase_from_comb(prior_rel_sigma=0.008)` assumes near-constant speed; a ramp
inflates `wander_frac` (q84-q16 of f_inst, `patterns.py:74-76`) → FIXEDHZ claim
widths (1.5*wander*o) swallow neighbors, TONE windows balloon, and Hz-domain
peaks smear below the guard → candidate starvation (the exact failure mode the
ledger recorded as `ladder_ordering_repair`/candidate starvation). No ramp
detector, no per-record ramp flag.

### F7. Very short records / very low speed (medium, likely)
LO=3.0 Hz hard floor (`peakshor.py:25`) — slow agitators/kilns (<3 Hz) can never
get a candidate. <5 Hz with a 4 s record = <20 revs: block coherence (5-rev
blocks) has ≤4 blocks → ratio_b statistics collapse; snap_tol widens; `_crystal`
unresolvable. Trend layer needs 6 records before any alarm. Behavior at these
edges is undefined rather than degraded-with-flag.

### F8. Wrong kinematic sheet / stale registry (medium-high impact, medium likelihood) — see S7
Bad tooth count fabricates seeded SIDEBAND energy and can octave-flip the frame.
Sketchable guard: seeded-fan energy must exceed a null (sheet-permuted) baseline
before being trusted; emit `sheet_contradiction` when sheet_match(true sheet)
loses to the blind decomposition.

### F9. Multiple simultaneous faults (medium, common in reality)
`rules_from_ledger` is first-match precedence (looseness > gear > bearing >
misalign > imbalance) and the ML head is single-label 6-way softmax with a margin
abstain — a looseness+bearing machine reports looseness only, and the margin
between the two TRUE faults reads as "low confidence". The pattern ledger already
carries the multiplicity; only the O3 heads flatten it.

### F10. Aliasing near Nyquist (low-medium, occasional)
Mesh/PWM content folded by a non-anti-aliased decimation lands at stable
non-rational low orders: frequency-stable → escapes NEARRAT via coherence? No —
folded tones are NOT shaft-comb-coherent (folding scrambles phase relation to the
comb reference under wander) → classified drifting/unsnapped → false bearing.
Same signature-collision as F5.

### F11. 50-vs-60 Hz grid confusion (low)
Both families are seeded (`patterns.py:211`) and the general narrow-line detector
covers harmonics; residual risk is only the interaction with F4 (wrong fs moves
the seeds off the true lines).

### F12. DC offset / integrated (once- or twice-integrated) signals (low-medium)
Mean removal handles DC; integration's 1/f² floor tilt inflates low-order BAND/
FLOOR shares and drops pf_floor-relative gates' reference — mostly graceful, but
severity drivers (a1 in "velocity-ish" units) shift regime.

---

## Cross-cutting recommendation (one line each)
1. Add a frozen-artifact contract test (U1/S1): feature names+order+count checked
   at `Cascade.load`, golden prediction vector pinned.
2. Add an input-hygiene gate before `analyze_record` (F1/F2): finite, clip
   fraction, dropout runs, spectral-slope unit sniff — return "unknown" + reason.
3. Fix/pin the degenerate-margin overconfidence (U2) before any real deployment:
   it converts garbage input into certified-looking confidence.
4. Treat S3 (hum mask vs 2-pole mains machines) as iteration material: measurable
   on the existing corpus by conditioning speed top-1 on |f0 - grid| < 2 Hz.
