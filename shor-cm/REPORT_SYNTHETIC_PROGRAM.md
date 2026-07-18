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

## Artifacts

| Path | Content |
|---|---|
| `experiments/synthval_massive/` | C1–C6 findings.json, maps (parquet + PNG) |
| `experiments/massive/manifest.parquet` | corpus ground truth (regenerate any run from seed) |
| `experiments/massive/o1_results.parquet` | 6,000 (run × ablation) O1 outcomes |
| `experiments/massive/ledger.parquet` | 18-feature physics ledger + rules predictions |
| `experiments/massive/ml_findings.json` | O1/O3/O4 headline numbers |
| `experiments/massive/*.png` | reliability, degradation, coverage-accuracy, confusion, monotonicity |
| `work/loop_ledger.jsonl` | 4 attempt entries (constitution-compliant) |
| `work/deviations.md` | Docker→venv substitution, real-data wave deferred |
