# HYBRID Program — real carriers + labeled synthetic fault injection

Phase 3 addendum (iteration 34+ candidate). Motivated by the iteration-33
evidence: contract P3-33-D1 MISSED — no forge reaches the realism band
(discriminator AUC 0.98+, giveaway = real records' spectral peak density
and band texture) — while D2 proved that *more realistic training data
transfers better*. The strongest possible realism is real data itself.

**The idea in one line:** take REAL healthy recordings as carriers (real
noise floor, real transmission path, real sensor chain, real spectral
density — realism by construction), inject SYNTHETIC fault signatures
with exactly known labels (type, subtype, severity-as-SNR, defect
frequencies), and train on the mixtures.

**Why it should help (mapped to measured weaknesses):**

| Iteration-33 finding | What hybrids provide |
|---|---|
| D1 MISSED: forge texture ≠ real texture | Carrier texture IS real; the discriminator gap collapses by construction for everything except the injected component |
| Incipient detection untested (v6 incipient mode is synthetic-only) | Severity = injection SNR, swept continuously down into the noise floor of a REAL floor — labeled early-fault curves |
| Severity heads weak on real data (CWRU Spearman ≤ 0) | Monotone injected-severity ladders on real carriers with ground truth |
| False alarms on healthy records (2/4–3/4) | Null-injection controls make "healthy real texture" an explicit training class |
| Blind speed 0.0 on CWRU | Carriers have measurable true speed (own tacho/1x/nameplate) → labeled REAL-signal speed training data for the T1 tracker |

---

## 1. Carrier inventory (dev-real ONLY, never sealed-real)

| Source | Carriers | Notes |
|---|---|---|
| MAFAULDA (local) | 49 normal runs × 5 s × 8 ch | The main event: multi-channel, tacho truth, speed sweep 700–3600 rpm |
| CWRU | 4 normals (~5 s @ 48 k) | Thin; window + re-inject |
| MFPT | 3 baselines (6 s) | Thin |
| SEU | 4 health files (long) | Gearbox carriers — carry real mesh tones |
| Wind turbine | days 1–8 (~6 s @ 97 k) | "Healthy-ish"; label risk: incipient degradation may already exist — use days 1–5 only, flag as carrier-grade-B |
| Later | Paderborn healthy (K001–K006, many 4 s runs), XJTU first 10% of life, IMS early hours | Orders of magnitude more carrier volume |

Rules: carriers are catalogued once with a `carrier_id`; all augmented
windows inherit the carrier's group for splitting (no carrier straddles
train/eval); real FAULTY records are never carriers and never train —
they stay evaluation-only.

## 2. Injection engine (`shorcm/hybrid.py`)

All injections are placed in the CARRIER's own physical frame:

1. **Estimate the carrier frame first**: shaft rate (tacho if present,
   else 1x/nameplate — this is carrier metadata, allowed: it's training
   data construction, not blind inference), RMS, spectral floor, resonant
   bands (kurtogram scan), existing shaft-lattice amplitudes.
2. **Bearing faults** (primary): v6's jittered impact-train generator
   (per-impact slip jitter, load-zone/cage modulation, extended spall,
   BPFO/BPFI/BSF from a sampled bearing geometry × carrier shaft rate),
   convolved with a resonance kernel **placed inside a band the carrier
   actually has** (choose among the carrier's measured resonant bands,
   randomized) — so the defect rides real structure, not an invented mode.
3. **Imbalance / misalignment / looseness — spectral surgery, not
   addition**: measure the carrier's own 1x/2x/3x lines (STFT-domain)
   and scale them (imbalance: 1x gain; misalignment: 2x,3x gain + axial
   channel gain where multi-channel; looseness: add half-harmonic comb
   phase-locked to the carrier's own 1x phase). This keeps the lattice
   consistent with the real speed wander instead of pasting a foreign
   tone next to the real one.
4. **Gear faults** (SEU/gearbox carriers only): sideband fans around the
   carrier's OWN measured mesh tones.
5. **Severity = calibrated SNR**: injection energy set relative to the
   carrier's band RMS, swept over a ladder (e.g. −30, −24, …, +6 dB) →
   continuous severity labels with physical meaning ("6 dB above the
   floor of THIS machine").
6. **Null-injection control (label-leakage killer)**: every carrier also
   passes through the identical processing chain with zero injection
   (same STFT round-trip, same resample/dtype ops). "Healthy" hybrids
   are null-processed carriers, so processing artifacts cannot become
   the fault label.

## 3. Leakage & validity gates (pre-register before any training)

- **H-G1 processing-artifact gate**: discriminator between raw carriers
  and null-injected carriers must be at chance (AUC ≤ 0.55) — else the
  chain leaks.
- **H-G2 realism gate**: discriminator real-faulty vs hybrid-faulty
  windows (per dataset, grouped by record). Target: AUC ≤ 0.8 (the band
  v2/v6 never reached; hybrids should land near it by construction).
  Attribution tells us what the injection still gets wrong.
- **H-G3 carrier-memorization gate**: all hybrid claims are evaluated
  CROSS-DATASET (train hybrids on carriers from datasets A, evaluate
  blind on real faulty records of dataset B whose carriers were never
  used). Same-dataset numbers are reported but labeled IN-DOMAIN.
- **H-G4 severity-semantics caveat (standing)**: injected severity is
  injection-SNR, not field damage size. Real fault growth also changes
  the carrier (friction, speed, load). Claims are phrased as
  "detectability ordering", never "damage size".

## 4. Training recipe (v7 candidate)

Corpus = mixture, ratio pre-registered (start 50/50):
- SimForge v6 (unbounded kinematic diversity — hybrids cannot cover
  gearbox archetypes, VFD populations, speed ramps),
- Hybrid corpus (realism anchor: real texture, labeled injections,
  null controls).
Curriculum option: pretrain on v6, fine-tune on hybrids (compare with
flat mixture; pre-register which is primary).
Heads: fault (6-way, null-hybrids strengthen the healthy class),
severity (injection-SNR regression per carrier), speed (T1 tracker gets
carrier-speed labels on REAL signals — its first real-signal training
data), envelope/band-selection (injected band is known → supervised
band selection becomes possible for the first time).

## 5. Pre-registered contracts (iteration 34, to `work/contracts.jsonl`)

- **P3-34-H1**: H-G1 passes (AUC ≤ 0.55) and H-G2 AUC ≤ 0.85 on ≥ 2
  datasets. (Realism by construction actually materializes.)
- **P3-34-H2**: v6+hybrid-trained model, blind, cross-dataset: bearing
  recall ≥ v6-only AND healthy false positives ≤ v6-only on every
  dataset (the D2 protocol, next rung).
- **P3-34-H3**: injected-severity Spearman ≥ 0.8 on held-out carriers,
  AND real CWRU fault-size ordering (007/014/021/028) Spearman > 0
  (v5 measured ≤ 0 — any positive ordering is progress; CI reported).
- **P3-34-H4**: incipient curve: minimum detectable injection SNR
  (recall ≥ 0.8 at FP ≤ 0.1 on null controls) improves ≥ 6 dB over the
  v6-only model on held-out carriers.

## 6. Execution order

1. `shorcm/hybrid.py` + mechanism tests T71+ (carrier catalog, frame
   estimation, injectors, null chain, SNR calibration).
2. `scripts/107_build_hybrid_corpus.py` (windows × injection ladder ×
   fault types; parquet with carrier_id groups).
3. `scripts/108_hybrid_gates.py` (H-G1/H-G2 discriminators).
4. `scripts/109_retrain_v7.py` (mixture + curriculum arms, serve-path
   OOF, freeze models_v7 on promotion).
5. Re-run `scripts/105_blind_gates.py` MODELS=models_v7 → contracts.
6. LOCAL SESSION: repeat with MAFAULDA's 49 normals as the primary
   carrier pool (multi-channel injection incl. axial-ratio misalignment
   signatures + mic channel) — this is where the program gets real
   statistical power. Cloud carriers total ~15 records (ANECDOTE-grade);
   treat the cloud run as the pipeline shakedown.

## 7. What hybrids will NOT fix (keep expectations honest)

- Blind speed on carriers without 1x content (CWRU): labeled data helps
  the T1 tracker learn, but physics-absent lines stay absent — the
  octave-posterior honesty requirement stands.
- Faults whose signature is a change OF the carrier (rubs, distributed
  wear, stiffness loss) — injection can only add/scale components.
  These stay SimForge-only.
- Multi-fault interaction physics; sensor faults (separate hygiene
  classes).
- Any sealed-real claim: hybrids are a TRAINING device; sealed-real
  evaluation remains untouched real recordings only.
