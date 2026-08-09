# v2 results: three real rigs, a 48-asset fleet, and a codec that earns its bits

v2 asks one question at three scales: how few bytes can carry an analyst-grade
spectrum, when the compressor is allowed to know what a spectrum is?

Everything below was measured by code in this repo (`make v2-all`). The real
data is the full CWRU 12 kHz drive-end matrix (60 files - v1 used 16), the
MFPT bearing set (20 files, two sample rates), and the SEU gearbox rig
(20 conditions at 5.12 kHz) - three bearing geometries, three rigs, sample
rates from 5.12 to 97.656 kHz, all from public GitHub mirrors
(`docs/DATA_SOURCES.md`). The synthetic side is a 48-asset, 90-day campaign:
8,640 acquisitions across nine machine classes with seeded fault
progressions, VFD setpoint jumps, ramps, and two failing sensors.

## 1. Headline: bytes per frame, same distortion contract as v1

Dual-rail frame (768-bin acceleration + 384-bin envelope, uint8 at 0.5 dB).
"raw" is the two-rail float32 Welch spectrum; "gz" is zlib-9 over the uint8
bins - the honesty baseline, a general-purpose compressor with none of the
codec's structure.

| | raw | u8 bins | gz | codec v1 | **codec v2** | v2 vs raw | v2 vs v1 |
|---|---|---|---|---|---|---|---|
| CWRU (332 frames, 64 states) | 16,392 | 1,152 | 980 | 522 | **251** | **65x** | 2.1x |
| MFPT (32 frames, 20 states) | 102,408 | 1,152 | 710 | 940 | **391** | **262x** | 2.4x |
| SEU bearing (80 frames) | 8,200 | 1,152 | 784 | 435 | **182** | **45x** | 2.4x |
| SEU gear (80 frames) | 8,200 | 1,152 | 792 | 417 | **176** | **47x** | 2.4x |
| Fleet, synthetic (8,640 frames) | 16,392 | 1,152 | - | 375 | **177** | **93x** | 2.1x |

The fleet at two acquisitions a day costs **354 B/day/sensor** - half of v1's
751, and v1's own 90-day radio arithmetic (4 sessions, connection setup
dominates) is untouched: fewer payload bytes only widen that margin.

What did NOT get cheaper: the distortion. The contract is v1's, verbatim -
transmitted bins exact to the 0.5 dB quantiser, suppressed bins bounded by
max(k*MAD, floor), and it is enforced by fuzz test (`test_codec2.py`: 7,200
frames including dropouts and saturation, encoder and decoder in bit lockstep,
zero divergence). On CWRU, verified patterns carry **99.2% of their energy
through the codec** (v1: 99.2% as well; median error 0.26-0.30 dB, 98.8% of
patterns within +/-3 dB), measured on the exact bins each pattern owns - see
section 7 for both the verification pass and why the metric is defined this
way. Every named line (1x, 2x, 3x, BPFO, BPFI, BSF, FTF) decodes at
0.0-0.5 dB median error, p95 within 3 dB, on all three rigs.

## 2. Where the 2.1-2.4x over v1 comes from

Three mechanisms, in descending order of yield:

**Entropy stage everywhere.** v1 spent whole bytes on symbols worth 2-4 bits:
raw uint8 anchor dumps, int8 band gains, 3-byte peak records, varint
residuals. v2 packs every layer through adaptive Rice coding, LOCO-I style -
the parameter is derived from a running mean on both sides, so it is never
transmitted. Rice, not arithmetic coding, deliberately: shift-and-mask on an
MCU, no multiplies, no tables, within a few percent of entropy on the
geometric-ish residuals actually observed.

**Zero-byte peak tables.** v1 transmitted 3 bytes per anchor peak (120 B per
anchor). But the decoder holds a bit-exact copy of the reference spectrum, so
it can run the same peak picker the encoder ran and get the same table.
Determinism replaces transmission. Anchors: 890 -> 463 B (acc), 482 -> 197 B
(env), measured on CWRU, and anchors are now lossless (first-difference +
Rice) where v1's already were raw.

**Reference choice.** v1 residuals always referenced the GOP anchor. v2
encodes against both the anchor and the previous decoded frame and keeps the
smaller, at one header bit. The chain reference compounds packet-loss blast
radius, so it is a config (`refs='anchor'` restores v1 semantics); it buys
3-8% on real rigs - real but the smallest of the three.

Per-layer ledger on CWRU residual frames (eval_breakdown): acceleration rail
84 B mean - gains 6 B (7%), tracked peaks 30 B (35%), active-bin list 55 B
(66%); envelope rail 43 B, where the peak layer dominates (56%) because so
few bins ever clear the dead zone. The next factor lives in the actives -
which is an alignment problem, hence section 4.

Economics note: with anchors 2.5x cheaper, v1's re-anchor trigger (residual >
0.60x anchor) fires too early. v2 uses 0.85, measured worth 3% on the fleet
and indistinguishable from 1.0.

## 3. Rate-distortion: the knobs, and what they actually buy

CWRU 8-file subset (2 normal + 6 fault states), both rails, 2 s windows,
sweeping the dead-zone:

| k*MAD | B/frame | line p95 | pattern energy |
|---|---|---|---|
| 1.5 (floor 2 dB) | 332.5 | 2.00 dB | 93.2% |
| 1.5 (floor 3 dB) | 285.9 | 2.47 dB | 92.6% |
| **2.5 (floor 3 dB, default)** | **221.2** | **3.00 dB** | **91.4%** |
| 4.0 | 184.1 | 3.00 dB | 88.5% |
| 6.0 | 174.6 | 3.50 dB | 89.0% |

The curve is shallow and saturates fast: above k=2.5 the floor term
dominates (all floor settings collapse to the same bytes) and k=6 buys 21%
bytes for a ~2.4-point energy loss. The default sits at the knee.

The sharpest rate lever remains what v1 found - acquisition length -
re-confirmed through the entropy stage, now with the anchor amortisation
confound removed by reporting residual frames alone:

| window | B/frame (all) | residual frames only | line p95 |
|---|---|---|---|
| 1 s | 299.1 | 243.1 | 3.50 dB |
| 2 s | 221.2 | 126.7 | 3.00 dB |
| 4 s | 261.1 | **75.1** | **1.50 dB** |

Residual cost falls 3.2x from 1 s to 4 s while accuracy improves - Welch
averaging smooths the chi-squared bin scatter the coder would otherwise pay
to encode, exactly v1's mechanism, still the biggest knob on the board. (The
all-frames column misleads at 4 s because the fixed-length files then hold
only 2-3 frames per chain, so anchors dominate the mean; on a device the
chain is months long and the residual column is the honest one.)

## 4. Deeper signal processing, cashed out in its own currency

Every v2 DSP element was measured in the currency it claims: SNR if it says
"detection", bytes if it says "alignment". Two claims survived, two failed,
and one broke even - all five are in the code and `eval_dsp.json`.

**Viterbi speed tracking: -56% bytes on a hostile asset.** A 120-acquisition
sequence, weak shaft comb, a fixed 47.6 Hz mount line as a rival. Per-frame
argmax teleports on 89 of 120 frames (p95 error 16.9%); one quadratic
transition penalty - a machine's speed cannot jump - removes every teleport
(p95 0.65% incl. wander) and the same codec drops from 315.6 to 138.0
B/frame. Misalignment is the single most expensive thing a spectrum codec
can be fed; smoothing the speed estimate is worth more than any coder knob.

**Dual-evidence speed estimation: the fleet tail, eliminated.** The envelope
comb is sharp but treacherous: on an outer-race fault the envelope contains a
BPFO-spaced comb and no shaft comb at all, and BPFO/3 (1.19x for a 6205)
lands inside the search span and wins with GOOD confidence - a confidence
gate cannot catch a confidently wrong estimator. v2 accepts the envelope
answer only when the acceleration harmonic comb corroborates it, and runs
that comb search on a full-length FFT (Welch's 2.93 Hz bins cannot resolve a
6.4 Hz rotor; the fleet's slow mixer sat at 35% error until this). Fleet
speed error: median 0.29%, p95 0.65%, and the gate's false alarms on healthy
assets went from 12/22 to **0/22** - most of what looked like gate
misbehaviour was misaligned spectra.

**Angular resampling: wins exactly where it should, and nowhere else.**
Welch in the angle domain (resample at uniform shaft angle, then averaged
periodograms) holds gear-mesh line SNR at 34-35 dB under 3% wander and
run-up ramps where the time-domain Welch path collapses to 24 dB. At <=1%
wander the two are equal - and on bytes the time-Welch path stays cheaper
even at 3%, because per-acquisition order binning already absorbs the mean
speed and the codec pays only for the smear. Conclusion: order tracking is a
detection tool for wander >=3% and transient regimes, not a compression
tool. (First implementation used a single full-length FFT and LOST 4x on
bytes from unaveraged bin scatter - the fix was averaging in angle, and the
lesson is v1's again: averages, not resolution, set the byte cost.)

**Kurtogram band selection: worth -4.7 dB where a tuned band exists, worth
everything where none does.** Against CWRU/MFPT fault recordings the
adaptive band gives up 4.7 dB of envelope line SNR to the hand-tuned
2-5 kHz band (the lines stay 15-20 dB clear - detectable either way). But
the fixed band is not merely suboptimal elsewhere, it is *unconstructible*
on the SEU rig: Nyquist is 2.56 kHz, so every SEU number in section 1 exists
only because band selection is data-driven. Two guards matter as much as
the score: a minimum bandwidth of Nyquist/16 (a modulated impact train needs
room for its sidebands; narrower bands show beautiful kurtosis while
amputating the modulation the envelope spectrum is for), and widest-band
tie-breaking.

**Cepstral prewhitening: rejected.** Flattening |X| before demodulation
costs 8.4 dB of envelope line SNR - it removes the resonance gain that
envelope analysis rides on. Negative result, kept out of the pipeline,
recorded here so it does not get re-invented.

## 5. The fleet campaign as a guardrail

48 assets, nine machine classes (v1's five plus belt drive, VFD pump,
reciprocating compressor, 6.4 Hz mixer), 40% with seeded fault progressions
(exponential, linear, sigmoid, intermittent), VFD setpoint jumps of +/-12%,
weekly load cycles, 5% ramp acquisitions, and two deliberately failing
sensors (clipping, dropouts).

The compression numbers above hold across every class (140-201 B/frame
v2best). The guardrail: the unmodified v1 gate, fed v2 extraction, detects
24/24 seeded mechanical faults (median latency 15 days into slow
progressions whose onset is defined at severity ~0) with zero false alarms
on 22 healthy assets over 90 days. A codec that wins bytes by starving
detection would be a bug with a good ratio; this one is not.

## 6. A v1 bug the deeper pipeline surfaced

v1's synthetic wander model normalised low-passed pink noise by its standard
deviation alone. A 0.08 Hz low-pass leaves an almost-constant signal whose
mean dwarfs its std, so "0.3% wander" silently became a random CONSTANT
speed offset of up to +/-15%: every v1 synthetic acquisition ran at a speed
other than the one requested. v1 never noticed because its estimator
searched 20-70 Hz blind and absorbed the offset; the moment v2's
nominal-guided estimator disagreed with the label, the bug had nowhere to
hide. Fixed in `synth2._speed_profile` (v1's `synth.py` is preserved
unmodified as the baseline). Two lessons: ground truth is only as true as
its own tests, and a second independent estimate is the cheapest test there
is.

## 7. Cloud-side comb verification, and what "recovery" should mean

The comb finder scores a candidate fundamental on the MAXIMUM inside a
tolerance window around each harmonic, and a window maximum is a low bar: a
neighbour's tail, a haystack edge, or plain noise all supply one. On busy
spectra the scorer over-reports - CWRU IR007's envelope returned four combs
of which one is physical. v2 adds a verification pass (`verify_combs`, on by
default in `extract_patterns`) that re-examines every claimed harmonic with
three harder questions: is the window argmax a REAL local peak of the
spectrum (not the window edge riding a neighbour's slope), is that peak IN
PLACE (within half the search tolerance of where the harmonic predicts), and
is it CARRYING energy above the floor on its own. A comb survives only if at
least half its claimed harmonics pass and those verified harmonics hold most
of the claimed energy; the fundamental is then re-fit through the verified
peak positions. Measured behaviour: the synthetic healthy motor keeps exactly
its true 1.00x shaft comb and loses three parasites; CWRU IR007's envelope
keeps the physical 0.98x inner-race comb and loses the 1.08x/0.90x/0.69x
ghosts; CWRU normal - whose low-frequency acceleration genuinely carries no
harmonic comb, per v1's own mount-line finding - loses all five proposals.

Verification also forced a definition question. Recovery was previously
measured by re-running discovery on the decoded spectrum and matching keys -
which quietly measured extractor STABILITY, not codec loss: a sub-multiple
collapse on the decoded side scored as "pattern lost" while every joule of
its energy sat intact in the same bins. Recovery is now measured directly:
decoded energy over the exact bins the verified pattern owns. By that honest
definition the codec preserves 99.2% of verified-pattern energy on CWRU at
0.3 dB median error - and the number is nearly identical for v1 and v2,
which is what "same distortion contract" always claimed.

## 8. Known limits

- The Rice coder is bit-exact and MCU-shaped (shifts, adds, one 16-bit
  escape) but not yet ported into `firmware/`; the C port is mechanical and
  unstarted.
- Chain references (`refs='best'`) trade packet-loss blast radius for 3-8%;
  the eval assumes lossless transport. On lossy links run `refs='anchor'`.
- SEU ships no per-file shaft-speed ground truth beyond the 20/30 Hz
  nominal; SEU speed accuracy is corroborated (dual evidence agrees) but not
  independently verified.
- The kurtogram band is chosen once per machine state and cached. A fault
  that moves the informative band mid-life (resonance shift) keeps the old
  band until re-anchor; per-GOP re-selection is unimplemented.
- Fleet detection latency is measured against onset at severity ~0 by
  definition, so "median 15 d" mostly measures how long progressions stay
  below any physical signature; it is not an estimator property.
- v1's 90-day radio/energy model was not re-run; v2 changes payload bytes
  only, which strengthens, not weakens, v1's session-dominated conclusion.
