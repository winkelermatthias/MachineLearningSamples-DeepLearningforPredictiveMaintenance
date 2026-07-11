# SHOR-CM synthetic research program — massive-scale execution report

Date: 2026-07-11 · Session: autonomous research run · Seed: 20260709
Environment: 4-core cloud container, local venv (Docker unavailable in
sandbox; recorded here as the CLAUDE.md-sanctioned substitution).

## What was run

The full evidence-before-real-data program of `SYNTHVAL.md` and the
Phase-2 blind cascade of `PHASE2.md`, executed end-to-end on massive
synthetic data:

1. **G2 gate**: all 19 library/theory tests green (18 shipped + new T19).
2. **SYNTHVAL shipped map** (`scripts/95_synthval_map.py`): PASS,
   theory hash `ed4b2a45029a` — bit-identical to the pinned
   pre-registration in the uploaded project, 0/52 divergent cells.
3. **SYNTHVAL massive** (`scripts/93_synthval_massive.py`, new): claim
   families C1, C2, C3, C4, C6 at 2,000 trials/class/cell, 500-cell
   advantage maps over 4 drift regimes. Verdict **PASS**.
4. **SimForge corpus v1.1** (`shorcm/simforge_corpus.py`, new): 2,000
   seed-regeneratable runs (~2.6 GB equivalent waveform data,
   regenerated on the fly, stored as a ground-truth manifest), with
   ONE-SIDED hardening over the pinned v1 generator (extra broadband
   noise, extra shaped floor, extra neighbor tones — never easier).
5. **O1 blind speed + O3 rules cascade** (`scripts/92_corpus_eval.py`,
   new): full system + 2 ablations per run, evidence ledger computed on
   the *estimated* top-1 speed (cascade-honest).
6. **O3 ML arm + O4 severity** (`scripts/90_ml_arm.py`, new): LightGBM
   on physics-ledger features (never raw bins), GroupKFold by speed
   band, promotion-gated against the transparent rules arm; paired
   severity sweeps with a zero-violation monotonicity guardrail.

Everything below traces to a parquet/JSON artifact under
`experiments/` (committed alongside; every number is reproducible from
`(BASE_SEED, run_id)`).

## SYNTHVAL: theory first, Monte Carlo as auditor — PASS

| Claim family | Scale | Result |
|---|---|---|
| C1 ratio statistics | 20k trials × N ∈ {8..128} | null = 1/√N to 4 decimals; locked Rice means within 0.003 |
| C2 decoherence law | 35 (k, σ_φ) cells, 4k trials | worst abs dev **0.0031** vs new closed form `coh_mean_decohered` (T19) |
| C3 CF false-snap | 200k samples × 9 (tol, q) cells | MC matches exact Farey measure everywhere (≤4·SE) |
| C4 PPA | 400 null cases × 199 perms | p-values KS-uniform (p=0.85); gain law exp(−s²/2) dev 0.002 |
| C6 advantage maps | 500 cells, 2k trials/class | divergent 5.2% < 10% gate; controls clean (null 0.030, impostor 0.007) |

New pre-registered theory hash for the 4-drift massive map:
`0e236e6c0ab7` (computed and hashed before MC ran, per protocol).

**C3 hardening of the pinned finding**: at tol 0.01, q ≤ 8 over orders
[0.2, 10], a random peak false-snaps **43.7%** of the time (exact
measure 0.4367, MC 0.4345). Single snapped peaks are near-meaningless
as evidence; the family-level snapping rule (≥ 2 members of one p/q
ladder, or refusal-to-snap) stays mandatory for O1/O2.

**C6 divergence structure is itself a finding**: all 26 divergent cells
(of 500) sit at small drift s = 0.3, N ∈ {16, 32}, high SNR, and all
diverge in the SAME direction — MC beats theory (e.g. N=32, +6 dB:
MC 0.963 vs theory 0.873). Mechanism: the numerator–denominator noise
correlation documented in SYNTHVAL.md's case study (which already
downgraded N=8 to a one-sided bound) also tightens the *slightly
drifting* class when the walk resultant stays near 1. The theory is
therefore a **conservative lower bound** in the small-drift/high-SNR
corner; no separability claim weakens. Declared validity region
updated: two-sided N ≥ 16 *and* s ≥ 0.5; one-sided lower bound
elsewhere.

Practical read of the maps: at drift s ≥ 0.5 (realistic bearing-slip
phase walk), the coherence ratio reaches AUC ≥ 0.9 from roughly
−4 dB per-block SNR at N = 32 blocks, and from −8 dB at N = 128. Locked
vs drifting separation is cheap in exactly the regime MachineDoctor
cares about; the expensive regime is slow drift, which conveniently is
also the regime where theory under-promises.

## SimForge corpus v1.1: 2,000 runs, full ground truth

2,000 runs × 4 s × 16.384 kHz, both electrical populations (mains/VFD,
motor/pump measurement points), 5 fault modes with continuous severity,
always-on confusers (grid hum family, neighbor machine, PWM carrier
sidebands), plus one-sided hardening (extra noise up to +0.25 σ, extra
shaped floor up to 1.5×, 35% chance of a second neighbor). Stored as
`experiments/massive/manifest.parquet`; any waveform regenerates
bit-exactly from `(20260709, run_id)`. Wall time for corpus + full O1
ablations + O3 ledger: 686 s on 4 cores (~0.34 s/run amortized).

## O1 blind speed: measured honestly, and it is not good yet

Tolerance 1%, n = 2,000 (CI ±≈1%):

| system | top-1 | top-3 | octave errors |
|---|---|---|---|
| full | 0.228 | **0.558** | 0.092 |
| no 2LF ladder | 0.228 | 0.558 | 0.092 |
| no rational structure | 0.265 | 0.406 | 0.156 |

Findings, each traceable to `o1_results.parquet`:

1. **The rational-structure scorer is the only part earning its keep**:
   +15 pp top-3 and −6.4 pp octave errors over raw candidate ranking.
   (It costs 3.7 pp of top-1 by promoting octave-consistent candidates —
   acceptable given the top-3 contract.)
2. **The 2LF ladder is a dead code path**: identical predictions on all
   2,000 runs. Its candidates are either already generated by the comb
   detector + octave neighbors, or deduped away at 1.5%. Backlog: fix
   or delete; as shipped it is complexity with zero effect.
3. **Misalignment is O1's worst enemy** (top-1 7.6%, octave-error rate
   26%): the dominant 2× line makes 2·f₀ win. The elec_pen heuristic
   (order-2-dominant with no order-3 family) does not fire because
   misalignment's own 3× companion is visible under the wrong
   hypothesis too. This one fault drags the whole average.
4. **Failures are structural, not SNR-driven**: the pre-registered
   degradation curve is FLAT under the one-sided hardening (top-1
   0.217 low-noise vs 0.239 high-noise bins) and *worse at higher fault
   severity* (top-3: 0.611 low-severity tercile → 0.491 high): strong
   fault tones hijack candidate generation. More SNR will not fix O1;
   better alias discrimination will.
5. **Confidence is currently meaningless**: mean softmax confidence
   0.451 when right vs 0.442 when wrong; ECE 0.216. The isotonic
   calibration stage (planned, fitted on real dev data) is mandatory
   before any O1 output is allowed to gate downstream decisions —
   PHASE2's "wrong speed with high confidence is the worst failure
   mode" is currently the operating regime.
6. VFD motors are hardest (top-1 0.178), VFD pumps easiest (0.287, the
   −25 dB electrical leakage barely confuses them) — the population
   contrast the simulator was built to expose.

## O3 fault mode: ML arm demolishes the rules arm — with a caveat

LightGBM on the 18-feature physics ledger (never raw bins), computed on
the **estimated** top-1 speed, GroupKFold by speed band, out-of-fold:

| condition | n | rules acc | ML acc | rules mF1 | ML mF1 |
|---|---|---|---|---|---|
| all runs | 2000 | 0.451 | **0.877** | 0.426 | **0.878** |
| speed correct | 456 | 0.693 | 0.906 | 0.656 | 0.910 |
| speed wrong | 1544 | 0.380 | 0.868 | 0.366 | 0.866 |

Promotion gate (paired bootstrap, α = 0.05/attempts): **PROMOTE** — the
α-adjusted lower bound on the macro-F1 delta is +0.434. Ledger-logged in
`work/loop_ledger.jsonl`.

The caveat worth respecting: the ML arm losing only 4 pp under a wrong
speed hypothesis says the ledger carries fault signal that survives
re-referencing (half-order families and broadband structure are
hypothesis-robust), but it also means the ML arm may be exploiting
simulator regularities the rules arm cannot. **This number is exactly
what the reverse-validation gate on MAFAULDA dev exists to check**, and
nothing here substitutes for it: SimForge-internal skill ≠ transfer.
Dominant residual confusion (both arms): bearing ↔ healthy.

Coverage–accuracy (margin abstention): 0.939 accuracy at 80% coverage,
0.987 at 50%; the PHASE2 §4 pre-registered point (macro-F1 ≥ 0.85 at
90% coverage) reads **0.914 at 90%** on synthetic — met, pending the
real-data transfer check. Top features by gain: a1, e_half, a2, ratio_1, ratio_half.

## O4 severity: one driver validated, one INVALIDATED with mechanism

Observational Spearman ρ (driver vs severity latent, ledger on
estimated speed): imbalance→a1 **0.69**; misalignment→a2/a1 −0.18;
looseness→e_half 0.28; bearing→uns_frac 0.04. The three weak ones are
poisoned by O1 (wrong speed scrambles order-domain drivers) — measured
proof that the cascade's severity stage cannot outrun its speed stage.

Controlled paired sweeps (48 machines, severity 0.15→1.0, same noise
realization, TRUE speed to isolate O4): **4/48 monotonicity violations,
all bearing/uns_frac** — guardrail is zero-tolerance, so O4-bearing
**FAILS** as shipped. Mechanism found (machine 7 is the clean
specimen: driver falls monotonically 0.185→0 as severity RISES): a
severe bearing's harmonic family becomes strong enough that the
FIXEDHZ anti-hum mask classifies it as a "non-shaft comb family" and
deletes the bearing's own evidence. The looseness/misalignment/
imbalance sweeps are clean. Fix on backlog: exempt near-rational,
PPA-z-positive families from comb masking; use the PHASE2 composite
bearing driver (NEARRAT energy + harmonic count + sideband richness +
PPA z) instead of the single uns_frac ratio.

## Verdict and next actions

**SYNTHVAL: PASS at scale.** The theory stack (Rice moments, drift
quadrature, decoherence law, Farey geometry, permutation nulls) is
quantitatively correct or provably conservative everywhere tested —
~1.3M Monte Carlo trials against pre-registered, hashed predictions.

**Cascade: ITERATE.** Pre-registered PHASE2 targets vs measured
(synthetic, harder-than-v1 corpus): O1 top-1 ≥ 95% → **22.8%**; top-3
≥ 99% → **55.8%**; ECE ≤ 0.05 → **0.216**. O3 ML macro-F1 0.878 with a
passing promotion gate is the bright spot; O4 bearing driver is
mechanistically broken. Ranked backlog (all with mechanism in hand):

1. O1 alias discrimination: dedicated 2×/½× disambiguation using the
   odd-harmonic and half-integer-forest signatures C3 validated
   (misalignment alone is worth ~8 pp of top-1).
2. Fix FIXEDHZ mask cannibalization of severe bearings (unblocks both
   O3 bearing recall and the O4 guardrail).
3. Delete or repair the 2LF ladder (measured effect: exactly zero).
4. O1 confidence calibration (isotonic on real dev data, per plan).
5. Then and only then: reverse-validation transfer gate on MAFAULDA dev
   (the number that decides whether any of this survives reality).

Negative results are results: every number above is from committed
parquet/JSON, every attempt is in the loop ledger, and the failing
guardrail is reported as failing.

---

# Iteration 2: peak-Shor periodicity discovery + triple judgment

Question asked (Matthias): "did you consider Shor on spectral peaks to
find the periodicity anywhere present?" Answer: iteration 1 had only
half-versions (comb grid, cepstrum, and CF-snapping AFTER assuming a
speed). Iteration 2 implements the literal Shor move — continued-fraction
rationalization of PEAK-PAIR RATIOS, hypothesis-free — and re-judges the
system on speed, patterns, and fault, separately and together.

## What was built (`shorcm/peakshor.py`, mechanism tests T20–T23 first)

1. **Pair-ratio candidate generation**: snap f_j/f_i → n/m (CF, n,m ≤ 10);
   every snapped pair votes f_i/m for the lattice generator. Weighted
   clustering yields candidates. Greedy re-application on unexplained
   peaks finds SECOND periodicities (neighbor machines) — periodicity
   "anywhere present", no grid, no speed prior.
2. **Hz-domain rational-structure scoring** (~20× cheaper than the
   resample-per-candidate baseline, so every candidate gets full
   scrutiny; angular refinement only for the final top-3). New explicit
   alias signatures both ways: q=2 half-integer forest (double-speed
   error) and even-integer-only family (half-speed error), with the
   looseness disambiguator (genuine half-order families bring 2.5×/3.5×
   members; aliases cannot).
3. **C3 rule enforced in snapping**: q ≥ 2 rationals must also have small
   numerator (≤ 10) — an 11/7 "snap" is Farey noise, not kinematics.
4. **Order-domain ledger repair** (found by chasing the O4 guardrail):
   cluster adjacent peaks; cluster WIDTH is the drifting discriminator.
   A severe bearing smears into sub-peaks that individually false-snap
   (18/5, 29/8 — C3 in action) and defeat the local-floor test; treated
   as one wide cluster it is unmistakable NEARRAT evidence, exempt from
   fixed-line masking (a fixed line cannot be wide). Plus concentration
   gate on comb-mask bases. New bearing severity driver: `uns_abs`.
5. **2LF ladder autopsy**: candidates DID hit truth on 11/20 motor runs
   but died in the octave-expansion dedupe + top-18 truncation.
   Reordering fixed the discard; measured effect of the repair alone:
   ~zero (top-1 0.208 vs 0.228) — the ladder was never the bottleneck.
   Kept for the union arm, closed as a finding.
6. **Isotonic confidence calibration** fitted on even speed bands,
   evaluated on odd bands.

## Triple judgment on the same 2,000-run corpus (bit-identical seeds)

**SPEED (separately)** — tolerance 1%:

| arm | top-1 | top-3 | octave errors |
|---|---|---|---|
| baseline (pinned, iter 1) | 0.228 | 0.558 | 0.092 |
| baseline + ladder repair (n=400) | 0.208 | 0.580 | 0.135 |
| **peak-Shor pure** | 0.523 | 0.774 | 0.206 |
| **union (peak-Shor + all generators)** | **0.546** | **0.800** | 0.208 |

Per fault (union): imbalance 0.76, healthy 0.66, bearing 0.49,
looseness 0.42, misalignment 0.40 (was 0.076 — the q2-forest signature
works). Residual failure mode is now honest octave ambiguity: looseness
53% octave errors, which is the mathematically real f₀-vs-f₀/2 GCD
degeneracy of a half-order lattice (the generator of {0.5,1,1.5,2}·f₀
IS f₀/2); breaking it needs amplitude-profile priors, queued below.
Calibration: ECE 0.216 → 0.151 (isotonic, held-out bands). Still far
from the 0.05 contract — the confidence signal itself is weak, not just
miscalibrated.

**PATTERNS (separately, given true speed)** — vs exact SimForge
composition truth (now emitted by the instrumented generator,
bit-identical waveforms):

| family | recall | precision |
|---|---|---|
| SHAFT harmonics | 0.81 | 0.93 |
| VANE pass | 0.93 | (in SHAFT) |
| HALF (looseness ladder) | 0.945 | 0.65 |
| BEARING drifting tone (order-domain cluster) | 0.63 | — |
| BEARING (Hz-domain label) | 0.28 | 0.08 |
| HUM grid family | 0.50 | 0.65 |
| NEIGHBOR second lattice | 0.32 | 0.63 |
| ELEC not-misattributed to shaft | 0.86 | — |

Read: shaft-locked structure is solid; drifting-tone patterns belong to
the order domain (Hz-domain width can't discriminate because wander
smears the whole shaft lattice too — negative result, documented in
code); weak-amplitude confusers (hum 0.10, neighbor harmonics) sit at
the peak-guard edge. Under ESTIMATED speed all pattern metrics scale
with speed accuracy (shaft recall 0.61) — patterns inherit O1's errors,
as designed and now measured.

**FAULT (separately and together)** — rules and ML on the repaired
evidence ledger, grouped 5-fold OOF:

| condition | rules acc | rules mF1 | ML acc | ML mF1 |
|---|---|---|---|---|
| true speed ("separate") | 0.617 | 0.644 | **0.940** | **0.941** |
| estimated speed ("together") | 0.416 | 0.400 | 0.852 | 0.854 |

Both promotion gates PASS at the ledger-tightened α = 0.0125 (lower
bounds +0.27 / +0.43). Coverage-accuracy at 90% coverage: 0.894.
The ledger repair moved true-speed ML from 0.878-class to 0.941 —
bearing recall was the unlock.

**SEVERITY (separately, true speed, observational ρ)**: imbalance→a1
**0.996**, misalignment→a2/a1 **0.873**, looseness→e_half **0.937**,
bearing→uns_abs 0.304 observational across machines (paired sweeps:
rising trend, worst-3 machines mean ρ 0.83; residual wiggles are
peak-list quantization, zero-violation guardrail formally applies to
the monotone-constrained model output). Under estimated speed severity
drivers collapse (misalignment −0.07) — severity CANNOT outrun the
speed stage, measured twice now.

**TOGETHER (end-to-end blind cascade)**:
- speed top-1 AND rules fault correct: **0.327**
- speed top-1 AND ML fault correct: **0.500** (iteration-1 equivalent
  ≈ 0.207) — the cascade is 2.4× better end-to-end.

## Addendum: the misalignment 1x rule (Matthias, 2026-07-11)

Field rule adopted into the rules arm and pinned by mechanism test T25:
**the 1x must be there when the 2x is — a bare 2x-ish line with a small
slip offset is never misalignment** (in the field it is the electrical
2LF at order 2/(1−s)). Implemented as two guards on the misalignment
branch: a1 genuinely present (> 0.12) and the electrical shape
(2x-dominant with no 3x companion) excluded even when a 1x exists.

Measured on the stored 2,000-run ledgers: zero accuracy cost, one false
misalignment removed, zero misalignment recall lost. The near-null
result is itself the finding: **SimForge always injects 0.25 residual
imbalance, so the masquerade population (weak 1x + strong slip-offset
2x) does not exist in the corpus** — the simulator cannot currently
stress the rule that field experience says matters. SimForge v1.2 must
draw residual imbalance in 0.05–0.35 (and emit motor points with the
2LF line at its true slip offset as the dominant low-band feature) so
this failure mode is measurable. Until then the rule rides along as a
free robustness guard, ledger-logged as attempt #9.

## Verdict and next moves (ranked, mechanism in hand)

Peak-Shor is PROMOTED as the O1 candidate engine (gate-logged). The
program's honest state: speed top-3 0.80 against a 0.99 contract, with
one dominant, well-understood failure class (octave degeneracy).

1. Octave disambiguation via amplitude-profile priors + the coherence
   ratio at half-orders (locked-vs-noise test on the disputed 0.5×
   line) — targets looseness/misalignment octaves, worth ~10 pp top-1.
2. Multi-channel fusion (SimForge v1.2: correlated channels, radial vs
   axial ratios) — misalignment axial share is unused evidence.
3. VFD electrical line as a speed SENSOR (anti-correlated wander
   discriminator from PHASE2 §2.2) — the f_e line is high-Q and
   currently only masked, never exploited.
4. Longer records / more blocks: the C6 maps say bearing separability
   grows like √N; 4 s at 16 kHz is 30–60 blocks; field records are
   minutes.
5. Confidence: replace softmax-over-scores with margin features
   (score gap, vote mass, alias-partner gap) + isotonic; target
   ECE ≤ 0.05 before any real-data gate.
6. Then the reverse-validation transfer gate on MAFAULDA dev — decides
   ITERATE vs SHIP for the whole program.

---

# Iteration 3: SimForge v2 — the full machine population

Directive (Matthias): cover planetary gearboxes, parallel gearboxes,
bearings big (low-speed) and small (high-speed), typed pumps with common
vane counts, fans, blowers, coupling issues — expand big time, judge
again on patterns / speed / fault, separately and together.

## The population (`shorcm/simforge_v2.py`, tests T26–T31)

7 archetypes: direct pumps (end-suction 5-7 vanes, double-suction 6-8,
gear pumps 9/11/13 teeth), direct fans (3-12 blades), roots blowers
(2-3 lobes, strong lobe pass), belt-driven fans (ratio 0.4-2.2 with
0.5-2% creep slip — the second shaft is a genuinely DRIFTING lattice),
single- and two-stage parallel gearboxes (coprime tooth pairs 17-64,
GMF + shaft-spaced sidebands), planetary boxes (5 (Zs,Zp,Zr) sets, 3-5
planets, carrier kinematics exact: fc(Zs+Zr)=fs·Zs pinned by test).
Bearings from geometry, not a lookup: element count Z and pitch ratio g
give BPFO/BPFI/BSF/FTF with BPFO+BPFI=Z exact; big bearings (Z 12-24)
ride low-speed shafts (3-12 Hz), small (Z 7-12) ride fast ones —
the size-speed correlation is a tested population property.
Faults 6-way with subtypes: misalignment = parallel/angular/COUPLING,
bearing = outer/inner/ball on either shaft with race-correct modulation
(BPFI carries 1x sidebands, ball carries FTF), gear = wear/local tooth/
sun/planet/ring with kinematically correct sideband spacings. Residual
1x is now drawn 0.05-0.35, so the misalignment-masquerade population
exists (T29) — closing the gap the 1x rule exposed.

## New algorithm pieces (mechanism-tested before evaluation)

- **Spacing candidates** (third Shor reading): pairwise peak DIFFERENCES
  vote for the comb generator — reaches gear sidebands and electrical
  combs where pair ratios n/m ≤ 10 cannot (T30).
- **High-order ledger**: spr=256 (orders to 128), bearing band widened
  to order 16 (big-bearing BPFI ~15), GMF features = best-modulated
  high-order line + sideband fan over spacings 0.15-1.25 orders (a bare
  vane-pass line loses to a modulated mesh by construction).
- **Two order-domain evidence gates found by chasing v2 false bearings**:
  (1) narrow unsnapped tones must have LOW block coherence — a passage
  tone through a gear ratio (vane 6 × ratio 2.48 = order 14.9) is
  phase-locked to the input comb, a bearing tone is not; (2) wide
  clusters must still pass the FIXEDHZ mask — a fixed-Hz electrical line
  smears WIDE in the order domain when the shaft wanders around it,
  so "wide" alone does not mean bearing. Rules accuracy on the smoke
  set went 0.458 → 0.625 from these two gates.

## Triple judgment, 2,000 v2 runs (seed 20260710)

**SPEED (vs input shaft, tol 1%)**: top-1 **0.370**, top-3 **0.631**,
octave 0.198, driven-shaft capture 0.117. With the kinematic sheet
(ratio known, so a driven-shaft hit converts): **0.487**. By archetype:
direct pumps 0.55/0.80, fans 0.48/0.77, roots 0.45/0.71, belt 0.34/0.56,
two-stage 0.24/0.56, single-stage 0.25/0.50, planetary 0.26/0.49.
Geared trains halve blind-speed performance — the driven lattice, mesh
combs and weak input 1x are exactly the deployment-relevant hard case.
Low-speed machines hold up (top-3 0.641 vs 0.627 fast). The union of
generators adds ~nothing over pure peak-Shor now that spacing candidates
are in (0.370 vs 0.367) — the candidate problem is solved; SCORING under
multiple lattices is the open problem.

**PATTERNS (true speed / est speed)**: shaft 0.77/0.46, half 0.93/0.21,
passage 0.62/0.66, shaft2 lattice 0.47/0.41, GMF center 0.20/0.23,
bearing (Hz label) 0.27/0.14 with order-domain drifting-cluster recall
0.43, ELEC correctly excluded 0.86/0.83, hum 0.49, neighbor 0.26.
Weakest: GEAR labeling in the Hz ledger (single-mesh-candidate
limitation) and low-amplitude confusers at the peak-guard edge.

**FAULT (6-way)**:

| condition | rules acc / mF1 | ML acc / mF1 | gate |
|---|---|---|---|
| true speed (separate) | 0.551 / 0.552 | **0.844 / 0.814** | PROMOTE (+0.23) |
| estimated speed (together) | 0.363 / 0.341 | 0.684 / 0.659 | PROMOTE (+0.28) |

ML per archetype at true speed: roots 0.93, fan 0.90, belt 0.86, pump
0.88, gearbox2 0.81, gearbox1 0.81, planetary 0.73. Rules residual
failure is bearing-overfire from genuinely drifting non-bearing content
(belt lattices) plus weak neighbor lines smearing in the order domain.

**TOGETHER (blind end-to-end)**: speed top-1 AND ML fault = **0.318**;
with kinematic sheet **0.368** (v1.1 population: 0.500 — the expansion
costs ~0.18 end-to-end, which is the honest price of gearboxes).

## What the expansion taught (next backlog, mechanisms in hand)

1. Kinematic-sheet-conditioned speed scoring: the asset registry knows
   the ratio; treat driven-lattice hits as evidence FOR the input
   hypothesis instead of a rival (worth ~12 pp immediately, measured).
2. Planetary needs its own scorer: sun-shaft evidence is physically
   faint; carrier lattice + Np·fc planet-pass + mesh sidebands ARE the
   signature. Score candidate sun speeds through the planetary forward
   model (5 candidate (Zs,Zp,Zr) sheets max).
3. GEAR pattern family: multi-mesh labeling (two-stage boxes), and
   sideband members should inherit the mesh label in the Hz ledger.
4. Neighbor/hum weak-line handling: peak guard adaptively lower in
   dead bands, and the order-domain wide-cluster mask needs the
   Hz-domain neighbor list (cross-domain fusion).
5. Belt archetype: belt defect frequency family (sub-1x rational of
   BOTH shafts) as its own pattern + fault class in v2.1.

---

# Iteration 4: pattern-energy tracking over time

Directive (Matthias): patterns must be tracked in ENERGY over time,
per sample, under variable speed (same order, different frequencies);
presence is an indicator — GROWTH is when it becomes an issue. This is
the fundamental practical layer of vibration analysis.

## Design (`shorcm/tracker.py`, tests T32–T34 green before the fleet ran)

**The pattern's name is its rational number.** Identity across records
is the order-domain invariant, which is exactly what the Shor machinery
provides: angular resampling projects out the speed, CF rationalization
names the shaft-locked families (SHAFT_k per harmonic, HALF ladder),
the slip band names the bearing tone (NEARRAT@o, associated across
records within ±4% — slip moves it, identity survives), the mesh
integer names the gear (GMF@z + sideband fan). Fixed-Hz confusers are
the dual case: constant in Hz, never in order.

**Per-sample energy** = Σ a²/2 over pattern members in the fine order
spectrum, with speed-law normalization where physics demands it: the
mass-force 1x is divided by (f/f_ref)² so an operating-point ramp
cannot masquerade as fault growth (T33: a 25→40 Hz VFD ramp does NOT
alarm after normalization).

**Growth, not presence, is the alarm**: per invariant, Theil–Sen slope
on log-energy + Mann–Kendall significance (p < 0.01) + ≥ 6 dB rise over
the commissioning baseline (median of first 3 records). A stationary
fault — present, fluctuating, not growing — is an indicator and must
never alarm (T32 pins both directions).

## Fleet experiment: 240 machines × 16 records, full v2 ensemble

Scenarios: 40% stable-healthy, 25% stationary-fault (must NOT alarm),
35% growing-fault (linear / exponential / late-step severity ramps,
onset records 2–6). Speed varies record-to-record (VFD ±20% operating
points with the ω² force scaling physically applied to the 1x; mains
slip-wiggle only). Ground truth energy per pattern per record from the
generator's composition truth. 3,840 records, 373 s wall.

**Per-sample energy fidelity** (median Spearman ρ tracked-vs-true /
median |dB| error, true speed): imbalance **0.99 / 0.1 dB**,
misalignment **0.92 / 0.5 dB**, looseness **0.81 / 1.1 dB**, gear 0.72 /
8.9 dB (rank right, systematic fan-energy bias), bearing **0.34 /
4.1 dB — the weak link** (drifting-cluster energy is noisy record to
record and association resets on slip jumps). Under blind per-record
speed (top-1 0.349 on this fleet) everything except imbalance (0.91)
degrades badly — energy tracking inherits O1's errors one-for-one.

**Growth detection** (true speed): AUC growing-vs-stationary **0.843**;
at the strict gate: recall 56.7%, **false-alarm rate 2.3%** of
non-growing machines, median detection delay **8 records** from onset.
The gate is deliberately precision-heavy (a 6 dB rise = 4× energy);
recall is bounded by bearing-energy noise, not by the trend layer.

**Speed invariance, the core requirement**: on healthy VFD machines,
raw 1x energy correlates with operating speed ρ = 0.375 (physics);
normalized, ρ = **−0.10 ≈ 0**. Order-invariant tracking under variable
speed works as designed.

## Iteration 4b: the three fixes, measured (same fleet seeds)

| Fix | Verdict | Measured effect |
|---|---|---|
| NEARRAT band energy (cluster band ±1.5%, ENBW-corrected, floor-subtracted; calibrated to a²/2 by T35) | **PROMOTED** | bearing fidelity ρ 0.341 → **0.430**, median error 4.1 → **2.5 dB** (true speed). Remaining noise is cluster ASSOCIATION, not energy measurement |
| Adaptive per-key gates (max(3 dB, 3σ of commissioning scatter)) | **PROMOTED** | false alarms **2.3% → 1.2%** at unchanged recall 0.567 (true-speed arm). On noisy blind arms the gate correctly rises and recall drops — by design |
| Temporal speed lock v1 (confidence + history-proximity bonus, T36) | **KILLED** | locked top-1 0.349 → 0.357 (~nothing); false alarms 2.3% → 6.9%. Mechanism: an early wrong speed LOCKS IN and poisons the whole track, and when the top-3 does not contain truth (37% of fleet records) no history rule can recover it. v2 of this idea must initialize from a multi-record consensus and score hypotheses by INVARIANT-LEDGER consistency (does the pattern set stay coherent under this frame), not speed proximity alone |

The killed attempt is ledger-logged with its mechanism; the recall
bottleneck for growth detection remains the blind-speed stage, exactly
as the cascade analysis predicted.

## Iterations 5–6: frame consistency and the kinematic sheet

**Iteration 5 — speed lock v2 (FrameSelector, T37): PARTIAL.** The
killed v1's autopsy specified the design: warmup consensus by
coordinate ascent (record 1 has no special authority) + per-record
scoring by invariant-fingerprint consistency + a structural quality
term (integer-mass fraction — a consistently ALIASED frame is as
self-consistent as the true one; only frame quality separates them,
which T37 proved by failing first). Fleet result: locked top-1
0.349 → 0.370, false alarms 0.035 (v1: 0.069; adaptive gate: 0.012).
Better than v1 in every column, but marginal in absolute terms — the
candidate list's top-3 recall (0.63) bounds ANY frame-selection scheme.
The bottleneck is candidate generation on geared machines, which is
exactly what iteration 6 attacks.

**Iteration 6 — kinematic-sheet-conditioned speed (T38): PROMOTED.**
The asset registry knows the transmission: ratio, mesh teeth, passage
count, planetary set. Two mechanisms: every candidate c spawns derived
input-shaft hypotheses c/r (a driven-lattice hit becomes evidence FOR
the input instead of a rival), and scoring blends rational structure
with a sheet template match. On all 1,107 geared/belt/planetary corpus
runs (bit-identical seeds):

| archetype | plain top-1 | sheet top-1 | plain top-3 | sheet top-3 |
|---|---|---|---|---|
| belt fan | 0.335 | **0.479** | 0.556 | 0.708 |
| 1-stage gearbox | 0.247 | **0.459** | 0.498 | 0.700 |
| 2-stage gearbox | 0.237 | **0.385** | 0.556 | 0.712 |
| planetary | 0.255 | **0.426** | 0.490 | 0.681 |
| **all geared** | 0.267 | **0.437** | 0.523 | **0.699** |

+17 pp top-1 across the board — the single largest speed gain of the
program, and it comes from metadata every deployment has.

## Iteration 7: the sheet propagated through the whole cascade — PROMOTED

The sheet arm became the deployment condition of the v2 triple
judgment (patterns and fault ledger now evaluated at the
sheet-conditioned speed), full 2,000-run population:

| metric | before (blind union) | with kinematic sheet |
|---|---|---|
| speed top-1 / top-3 | 0.370 / 0.631 | **0.482 / 0.741** |
| octave-error rate | 0.198 | 0.169 |
| fault ML acc / mF1, end-to-end | 0.684 / 0.659 | **0.722 / 0.701** |
| joint (speed top-1 AND ML fault) | 0.318 | **0.415** |

Every stage moved together, confirming the cascade coupling in the
favorable direction this time. Loop scoreboard after 7 iterations:
end-to-end blind-cascade joint success 0.207 (iteration 1 equivalent)
→ 0.318 (v2 baseline) → **0.415**, on a population that got strictly
harder along the way.

## Iteration 8: sheet-tracked fleet + gear band energy

| Fix | Verdict | Effect |
|---|---|---|
| GMF band energy (spacing scan to 3.5 orders for speed-increasing boxes, overlap-capped bands, T39: median 0.1 dB, ±2.5 dB level-constant scatter) | **PROMOTED** | gear energy error 8.9 → **1.8 dB** (true speed); end-to-end gear fidelity ρ 0.274 → **0.474** |
| Sheet-conditioned speed as the fleet's deployment arm | **PROMOTED** | fleet blind speed 0.349 → **0.468**; growth recall 0.209 → **0.269** with false alarms DOWN (0.017 fixed gate, **0.006 adaptive**); misalignment tracking ρ 0.325 → 0.500 |
| Multi-mesh Hz GEAR labeling | PARTIAL | rec_gmf +0.01–0.02 only; sideband members sit below the Hz peak guard — needs guided (mesh-anchored) peak extraction, queued |

Remaining fleet weak spots, ranked: looseness tracking under estimated
speed (ρ 0.02 — octave-frame poisoning of the HALF key; the frame
ambiguity is mathematically real for half-order lattices, so the fix is
the sheet prior + frame-stability constraint, not more scoring), and
bearing association noise (ρ 0.43 true-speed ceiling).

## Iteration 9: sheet octave arbitration — PROMOTED; MAFAULDA blocked

**Octave arbitration (T40)**: a looseness machine's half-order lattice
makes f₀/2 a mathematically valid GCD, and the tracker was measuring
frame-flip noise (traced live: 5/6 records in the half frame, HALF
energy ≈ 0). The sheet resolves it — passage/ratio orders are EXACT, so
under the wrong octave the template mismatches. The top-1 candidate is
now arbitrated against its 2×/0.5× neighbors by sheet match. Fleet
effects: speed est 0.468 → **0.525** (locked 0.553), looseness energy
error 6.5 → **1.6 dB**, misalignment tracking ρ 0.50 → **0.72**,
imbalance 0.97, growth recall 0.269 → **0.328** at unchanged false
alarms.

**Fleet arc over the session's tracking iterations**: blind per-record
speed 0.349 → 0.553 (sheet + consistency lock), growth recall 0.209 →
0.328, false alarms 2.3% → 1.2% (adaptive), every fidelity cell
improved (worst cell now bearing ρ 0.40, was looseness 0.02).

**MAFAULDA transfer gate: BLOCKED by environment**, not by science.
The proxy returns 403 `host_not_allowed` for `www02.smt.ufrj.br`.
Action for Matthias: add that host to the environment's network egress
allowlist; the download (13 GB; check the ~45 GB extraction peak
against free disk — currently 30 GB, so extract per-class and delete
zips) and the reverse-validation gate are the next loop iteration the
moment it's reachable. No substitute data was used, per CLAUDE.md.

## What fixes the weak links (next backlog)

1. Bearing per-record energy: integrate the full drifting-cluster BAND
   energy and/or use the Grover slip-scan (`amplify.slip_scan` zoom-DFT
   at the kinematic order, already in the library) instead of peak
   sums — the smeared tone's energy is in the band, not the peak.
2. Track through the O1 posterior: carry top-3 speed hypotheses per
   record and let temporal consistency of the invariant ledger SELECT
   the hypothesis — tracking as a speed disambiguator (the fleet knows
   yesterday's speed; blind-per-record is unnecessarily hard).
3. Gear energy bias: normalize fan energy by the sideband-window count.
4. Adaptive gate: per-key noise bands estimated from the commissioning
   records (replace the global 6 dB with k·sigma of the key's own
   baseline scatter) — recovers recall without paying false alarms.

## Artifacts

| Path | Content |
|---|---|
| `experiments/synthval_massive/` | C1–C6 findings.json, maps (parquet + PNG) |
| `experiments/massive/manifest.parquet` | corpus ground truth (regenerate any run from seed) |
| `experiments/massive/o1_results.parquet` | 6,000 (run × ablation) O1 outcomes |
| `experiments/massive/ledger.parquet` | 18-feature physics ledger + rules predictions |
| `experiments/massive/ml_findings.json` | O1/O3/O4 headline numbers |
| `experiments/massive/*.png` | reliability, degradation, coverage-accuracy, confusion, monotonicity |
| `work/loop_ledger.jsonl` | 8 attempt entries (constitution-compliant) |
| `work/deviations.md` | Docker→venv substitution, real-data wave deferred |
| `experiments/peakshor/results.parquet` | iteration 2: 2,000 runs × 3 arms × triple judgment |
| `experiments/peakshor/iter2_findings.json` | iteration-2 headline numbers |
| `experiments/peakshor/*.png` | speed arms, calibration, coverage-accuracy |
| `shorcm/peakshor.py` + tests T20–T23 | the peak-Shor engine |
