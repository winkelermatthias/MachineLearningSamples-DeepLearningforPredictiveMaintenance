# EXPANSION: Grover and friends, plus the success loop

Additive to the existing spec. Nothing in waves 0-5 changes. This adds
hypotheses H4-H6, primitives in `shorcm/amplify.py` and
`shorcm/metrics.py` (tested, T9-T13), experiment `scripts/08_h4_amplify.py`,
a metrics-keeper agent, and wave 6: the autonomous improvement loop.

---

## 1. Grover's algorithm -> H4 (implemented)

Grover = an oracle marks target states, then repeated "rotation"
(inversion about the mean) amplifies the marked amplitude while the
background self-cancels. The quadratic quantum speedup does not transfer;
the amplification STRUCTURE does, three ways:

**1a. Slip scan (`slip_scan`).** Bearing tones sit near, not at, their
kinematic order (1-2% slip). Scan a demodulation frame over candidate
slips and take the amplitude curve: rotating the reference until the
marked component aligns. Peak location = mean slip (a load/health
indicator for free), peak sharpness = detection feature. Test T9: recovers
1.2% slip to <0.1pp in noise.

**1b. Predictive phase-aligned averaging (`ppa_zscore`).** The gap H1
leaves open: coherent averaging destroys bearing tones (phase
random-walk), power averaging keeps them but gains nothing. PPA aligns
each block by the previous block's measured phase: a real narrowband tone
has PREDICTABLE short-range phase and survives (toward ~N gain), noise
does not (~sqrt N). Guarded by a permutation null over block order, so the
output is a z-score, not a raw amplitude. Tests T10/T11: z>5 on drifting
tones, no false gain on noise. Together with H1 this completes a 2x2:
locked components amplified by coherent averaging, unlocked-but-real
components amplified by PPA, noise amplified by neither.

**1c. Template pursuit (`template_pursuit`).** Grover-as-matching-pursuit:
oracle = fault template (harmonic comb at rational base order, sideband
fan), iterate mark-best -> subtract -> repeat on the residual.
Floor-normalized log scoring so one huge line cannot impersonate a family
(same lesson as the comb capture guard). Feeds symbolic fingerprints,
complements CF snapping.

H4 experiment (script 08): does RATIO + {slip, sharpness, PPA-z} beat
RATIO alone on 7-way macro-F1, per phase variant, through the promotion
gate? Bonus physics check: slip vs severity monotonicity.

## 2. Other algorithms mapped (backlog, build via the loop)

**H5, Simon / hidden period on per-rev sequences.** Compute one scalar per
revolution (RMS, kurtosis, band energy), then run period finding on that
SEQUENCE of revolutions. Detects slow hidden periodicities that direct
spectra miss: cage modulation, pump interaction beats, planet-pass
patterns. Cheap: the sequence is ~100-300 numbers per run. Success: cage
detection threshold (dB) vs envelope-analysis baseline.

**H6, syndrome decoding (QEC) for sensor-vs-machine disambiguation.**
Error-correcting codes measure syndromes: consistency checks across
redundant views. Here: the same shaft-locked fault must appear on multiple
channels with stable relative phase and physically plausible amplitude
ratios; a signature on one channel violating cross-channel parity is a
SENSOR fault, not a machine fault. Test by corrupting one channel
synthetically (clipping, dropouts, gain drift) in healthy runs; success =
>90% correct sensor-vs-machine attribution. Directly monetizable at
NanoPrecise (false-alert reduction).

**QPE, multi-scale block ladder.** Phase estimation refines precision by
doubling evolution time. Classical port: coherence at block lengths 5, 10,
20, 40 revs read together like successive binary digits: locked components
stay coherent at every scale, slowly-drifting ones die at a
characteristic scale = a decoherence-time feature (drift rate readout).
Success: does the decoherence-scale feature separate looseness (locked
0.5x) from oil whirl (0.42-0.48x, drifting), the classic hard pair.
MAFAULDA lacks oil whirl, so validate the feature synthetically and on
field data later.

**Deutsch-Jozsa, one-shot triage.** A single cheap global statistic
(coherence entropy across the order grid + unsnapped energy fraction)
deciding healthy vs any-fault before running the full pipeline. Success:
>=99% recall at maximal rejection rate; value metric = fraction of compute
saved on the fleet. Feeds MachineDoctor edge budget.

**Zeno-gated sampling (firmware idea, not a MAFAULDA experiment).**
Frequent cheap triage statistics freeze state knowledge; full captures
trigger only on drift. Plugs into the 4-window adaptive upload model as a
principled trigger rule. Park in IDEAS until run-to-failure data exists.

**Explicitly skipped:** HHL (no sparse linear system here), QAOA/annealing
(template assignment is not the bottleneck), quantum walks (no graph
structure worth it). Recorded so the loop does not rediscover them.

---

## 3. HOW SUCCESS IS MEASURED (the loop's constitution)

The loop's failure mode is self-deception: iterate on a benchmark long
enough and numbers improve by selection, not discovery. Every rule below
exists to make improvement mean something.

### North star (NS)
Macro-F1, 7-way fault typing, BEST NO-TACHO variant (onex or comb),
extreme-speed hold-out, evaluated ON THE VAULT. This mirrors deployment:
MachineDoctor has an accelerometer and an RPM estimate, never a keyphasor,
and meets speeds it was not trained on. Tacho results are diagnostics;
no-tacho results are the product.

### Sealed vault (implemented, `vault_mask`, T12)
20% of runs sealed by run_id hash at catalog time: deterministic,
stateless, identical on every machine. Dev iterations never touch vault
runs (script 08 already filters). Vault evaluations are budgeted: 3 per
program (`request_vault_access`), each logged; the adversary audits the
counter. The vault answers "did we actually get better", at most 3 times.

### Promotion gate (implemented, `promotion_gate`, T13)
Candidate beats incumbent iff the paired-bootstrap delta on the dev NS is
positive at alpha = 0.05 / attempts, where attempts = ledger length. The
evidence bar RISES as the loop tries more things: attempt #1 needs a
nudge, attempt #50 needs an effect that survives alpha = 0.001. Plus all
guardrails:

- G-a tacho H1 AUC does not drop > 0.01 (no cannibalizing the easy case)
- G-b adversary suite passes
- G-c < 5 s CPU per 5 s recording (edge deployability)
- G-d the method ships >=1 synthetic mechanism test in tests/
- G-e no regression on previously promoted methods (pinned suite)

### Ledger (implemented, `log_attempt`)
EVERY attempt is appended, promoted or killed, with config hash. The
ledger is append-only; a deleted or rewritten line is a CRITICAL adversary
finding. Kills are as valuable as promotions: they are the multiple-
comparisons denominator and the map of dead ends.

### Iteration protocol (wave 6, repeat)
1. metrics-keeper picks the top backlog item by expected-info-gain / cost
   (H5, H6, QPE ladder, DJ triage, plus anything IDEAS.md accumulates).
2. signal-engineer implements behind the harness interface + mechanism
   test. `pytest -q` green or the iteration aborts.
3. Runner executes on DEV ONLY, computes paired deltas vs incumbent.
4. metrics-keeper applies promotion_gate, logs the attempt, updates the
   incumbent feature set if promoted.
5. adversary spot-audits (shuffle on the new features, ledger integrity,
   vault counter, guardrail G-e).
6. Every 5 iterations OR before a human checkpoint: one budgeted vault
   evaluation of the current incumbent. If dev gains do not show up on
   the vault (dev-vault gap > 2x the bootstrap CI width), the loop is
   overfitting: freeze, report, await human review.

### Program-level kill criteria
- Two consecutive vault evaluations with no NS improvement: stop, write
  the negative-result report (still a deliverable).
- NS(no-tacho) plateaus > 0.05 macro-F1 below NS(tacho): the method
  requires a keyphasor; pivot the program to the MAVEN pseudo-tacho
  variant or kill.
- Token/compute budget per iteration: 1 vault period must cost < the
  value of the information (metrics-keeper states expected gain BEFORE
  running; misses twice on the same idea family -> family blacklisted).

### What "breakthrough" means, in numbers
Pre-registered here, before any real data:
- Confirmed: NS(no-tacho, vault) >= 0.85 macro-F1 AND H3 gain exponent
  in [0.35, 0.65] for onex or comb AND slip estimates monotone with
  severity. Then this is a paper and a MachineDoctor feature.
- Interesting: NS(tacho) >= 0.85 but no-tacho <= 0.75: physics validated,
  deployment blocked on phase reference quality; MAVEN variant becomes
  the priority.
- Dead: RATIO never beats INC baselines outside the CI. Coherence adds
  nothing over standard order analysis on this data; publish the null.
