# SYNTHVAL: mathematical synthetic validation program

Purpose: evidence of success established BEFORE real data, with theory as
the primary object and Monte Carlo as its auditor. Additive; feeds the
wave-6 loop. Library: shorcm/theory.py. Gate tests: T14-T18. Experiment:
scripts/95_synthval_map.py, model: run it once per claim family.

## Protocol (what "autonomously, with quality" means here)

1. Every claim gets a closed-form or semi-analytic prediction in
   theory.py, unit-tested against limiting cases, BEFORE any experiment.
2. The theory map is computed first and hashed: pre-registration.
3. Monte Carlo runs with independent seeds; per-cell trial counts sized
   so MC standard error < half the tolerance band.
4. Agreement gate: divergence beyond the band is exit code 1 and a listed
   set of divergent cells: a mandatory investigation task, never a
   tolerance to widen silently. Validity regions may be declared, but a
   declared boundary must come with a mechanism and a one-sided bound
   that still holds outside it.
5. Negative controls run inside the same script: a same-vs-same contrast
   (AUC must sit at 0.5) and a broken impostor (phase-scrambled input
   must lose the advantage). A method that beats its own impostor by
   nothing is measuring nothing.
6. Artifacts: theory hash, results parquet, findings.json, map figure.
   All ledger-logged like any loop iteration.

## Claim families

- C1 ratio statistics: null E[r] = 1/sqrt(N); locked and drifting
  conditionals via Rice moments and the exact walk-resultant quadrature.
  Verified: T14, T15, T16.
- C2 decoherence law: a phase-reference error sigma_phi multiplies
  coherent amplitude by exp(-k^2 sigma_phi^2 / 2): Gaussian IN ORDER, so
  high orders die first under a bad reference. Verified: T18. This is the
  quantitative form of "the nominal variant's disease" and predicts
  exactly how much the no-tacho variants lose per order.
- C3 CF false-snap geometry: exact Farey-window union measure. FINDING,
  pinned in T17: at tol 0.01, q <= 8, a random peak in [0.2, 10] snaps
  ~1/3 of the time. Consequence adopted: a single snapped peak is weak
  evidence; families (>= 2 members of one p/q ladder) or refusal carry
  the diagnostic weight. O1's scorer and O2's ledger use family-level
  snapping only.
- C4 PPA: z-score is standard normal under exchangeability by
  construction (permutation null); gain vs walk step s follows
  exp(-s^2/2) per aligned pair. Covered by T10/T11; a KS uniformity test
  on null p-values joins the map script in the next iteration.
- C5 blind-speed identifiability: alias candidates produce denominator
  forests (q = 2 for double speed, even-integer-only for half speed);
  identifiability is a function of odd-harmonic content. Theory to be
  written in the same style before its experiment runs.
- C6 advantage maps: for each contrast that matters (locked vs drifting
  is the archetype), the (SNR, N, s) region where AUC >= 0.9, theory
  contour vs MC contour. The shipped map is the first instance.

## Case study: how the loop handled its own theory being wrong

The first map run DIVERGED (50% of cells outside band) with clean
negative controls, correctly localizing the fault to theory. Three
iterations followed, each driven by the divergence pattern, all inside
one session:
1. Moment-matched normal on the drift mixture: failed at small N; the
   mixture over walk resultants is right-skewed. Fix: integrate AUC over
   the latent instead of moment-matching. Divergence 50% -> 21%.
2. Delta-method variance without correlation: failed at high SNR; coh
   and inc share the in-phase noise projection (measured corr up to
   0.98), which cancels in the ratio. Second-order variance
   sigma^4/(2 A^4)(1/N - 1/N^2) matched MC to 5%. Divergence pattern
   moved, exposing the next flaw.
3. Regime blend keyed on per-block SNR: failed at low SNR x large N,
   because the numerator lives at POST-AVERAGING SNR, the entire point
   of coherent averaging. Fix: exact conditional, ratio as scaled Rice,
   AUC by deterministic quadrature. N >= 16 converged to +/- 0.03;
   N = 8 high-SNR remains a one-sided lower bound with the correlation
   mechanism documented and quantified (36x variance compression).
Final verdict PASS under a two-tier gate: two-sided in the declared
validity region, one-sided bound outside it. Total cost: minutes.

The point of the case study: the same protocol that validates the method
also debugged its own theory, and every intermediate wrong model is in
the history with the measurement that killed it. That is the quality bar
the autonomous runs are held to.

## Sizing and budget

Block-level simulation (complex Gaussians at one bin) makes cells cheap:
the shipped 52-cell map with 400 trials/class/cell plus controls runs in
about a minute on one core. Full claim-family sweeps (C1-C6, finer grids,
2000 trials) are hours on the lab box, embarrassingly parallel.
Waveform-level confirmation (through tacho extraction and resampling) is
reserved for spot checks of 5% of cells, since T1-T8 already pin the
waveform pipeline to the block-level model.
