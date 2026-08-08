# Spectral GOP Codec: edge compression with analyst-grade reconstruction

Version 0.1 skeleton. TODO markers indicate decisions requiring fleet data.

---

## 1. Cadence and frame structure

Two waveforms per day, 14-day anchor interval, so **GOP = 28 frames**: 1 anchor
(I-frame) + 27 residual frames (P-frames).

```
day  0    1    2    3   ...  13
     A p  p p  p p  p p ...  p p     A = anchor, p = residual
     |<--------- GOP = 28 --------->|
```

All residuals reference the anchor directly. No P-chains.

| Property | Anchor-referenced | P-chain |
|---|---|---|
| Packet loss blast radius | 1 frame | rest of GOP |
| Drift accumulation | none | compounds |
| Mean residual size | ~1.2x | 1.0x |
| Decoder state | anchor only | full history |

The 20% size penalty buys independent decodability. On NB-IoT / LTE-M with
2 to 8% packet loss this is not a close call.

### 1.1 Early anchor trigger

The device sends an anchor early when the residual stops being cheap. This is
self-regulating and handles run-ups, repairs, process changes, and
re-installations without any server-side logic.

```
if residual_bytes > 0.60 * anchor_bytes:      -> send anchor, reset GOP
if peak_match_rate < 0.50:                    -> send anchor, reset GOP
if global_gain outside [-12, +12] dB:         -> send anchor, reset GOP
if days_since_anchor >= 14:                   -> send anchor, reset GOP
```

TODO: calibrate the 0.60 threshold against fleet residual distributions.
Expect 5 to 12% of GOPs to terminate early on healthy assets, higher on
batch-process and VFD assets.

---

## 2. The residual vector (the "mask")

Work entirely in log amplitude, uint8, 0.5 dB per count. A multiplicative
change in linear amplitude becomes an additive offset, so reconstruction is:

```
S_k[b] = clamp(S_anchor[warp(b)] + Δ_k[b], 0, 255)
```

Δ_k is transmitted as four layers, decoded in order. Each layer is optional and
its presence is a header bit.

### Layer 0: global gain (1 byte)

Single int8, 0.5 dB per count. Captures load and speed-driven level shifts.
Estimated as the trimmed median of the per-bin difference over bins that are
above the anchor's noise floor.

### Layer 1: band gains (8 to 16 bytes)

int8 per band, on top of the global gain. Bands are octave or third-octave on
the order axis. Captures the broadband haystack, noise floor tilt, lubrication
state, and cavitation onset. This layer is what makes haystack events cheap:
a 2 to 8 kHz elevation costs 3 bytes instead of 400.

Band edges are fixed per machine class and never transmitted.

### Layer 2: tracked peak deltas (3 to 4 bytes per peak, 16 to 40 peaks)

The anchor carries a peak table. Each peak gets a stable ID for the life of the
GOP. Residual frames carry only changes:

```
uint6  peak_id
int7   Δamp    (0.5 dB/count, +/- 32 dB)
int7   Δfreq   (1/16 bin, +/- 4 bins)     [omitted when align_tier == D]
uint2  flags   (00 present, 01 vanished, 10 new, 11 reserved)
```

Peaks are matched to the anchor table by expected position after warping, with
tolerance widening as alignment confidence drops. A peak with |Δamp| < 1 dB and
|Δfreq| < 1/8 bin is not transmitted at all; the decoder assumes unchanged.

**This layer is the reason sidebands survive.** Sideband spacing is a
relationship between peak positions. Encoding peaks as tracked objects
preserves that relationship exactly; encoding them as bin-index deltas smears
it by the quantization step while leaving MSE looking healthy.

New peaks (flags=10) carry a full `(bin_idx uint16, amp uint8, width uint8)`
and are the primary early-warning signal. Never suppress them by threshold.

### Layer 3: residual bitmap (variable, typically 0 to 40 bytes)

Whatever remains after layers 0 to 2, thresholded at `k * MAD_baseline[b]`.
Encoded as run-length gaps plus zigzag varint values.

```
varint gap_to_next_active_bin
zigzag varint delta_half_db
```

Bins below threshold are transmitted as nothing. Their reconstruction error is
bounded by `k * MAD`, which is by construction below the asset's own
run-to-run variability. That is the formal guarantee behind "almost the
original": **error on suppressed bins is bounded by the machine's own noise;
error on transmitted bins is bounded by q/2 = 0.25 dB.**

### Byte budget

| Component | Bytes | Notes |
|---|---|---|
| Anchor | 780 to 850 | 768 bins uint8 + peak table + header |
| Residual, steady asset | 35 to 70 | layers 0 to 2 only |
| Residual, developing fault | 90 to 160 | layer 3 active |
| Residual, p99 | 240 | triggers early anchor above this |

GOP total, steady: 820 + 27 x 55 = **2,305 B**
Uncompressed equivalent: 28 x 25,600 = 716,800 B
**Ratio: ~310x**, ~380x after gateway-side zstd with a per-class dictionary.

---

## 3. Alignment ladder

Variable speed is ~10% of assets. Low-speed assets often present no usable
harmonic evidence at all. The ladder is confidence-gated with a hard bias
toward doing nothing.

| Tier | Condition | Action | Δfreq encoded |
|---|---|---|---|
| A | Tacho / VFD tag present | Order-resample from tag speed | yes |
| B | MAVEN conf >= 0.85 AND >= 3 harmonics detected | Order-resample from MAVEN | yes |
| C | MAVEN conf 0.50 to 0.85 | Log-f cross-correlation refinement, accept only if prominence > 3 dB and \|log α\| < 0.05 | yes, tolerance x2 |
| D | Everything else | **No warp.** Hz axis. Peak match tolerance +/- 2 bins | **no** |

### 3.1 Why log-frequency correlation

A speed change scales the frequency axis. Resampling to a log-frequency axis
converts that scaling into a pure translation, so a single 1D cross-correlation
against the anchor recovers α in one pass. Cheap enough for the MCU (one
resample plus one correlation over 256 log bins).

Accept the shift only if the correlation peak is *prominent*, not merely
maximal. A flat correlation surface means the spectrum has no periodic
structure to align on, which is exactly the low-speed case. Reject and fall to
tier D.

### 3.2 Why tier D suppresses frequency deltas

At 30 RPM the shaft order is 0.5 Hz. Resolving sidebands at 0.02 order needs
0.01 Hz bins, which needs 100 s of acquisition. Usually unavailable. Under
those conditions any Δfreq you compute is noise. Transmitting it costs bytes
*and* fabricates precision that the UI will then display as if it were real.
Suppress it, widen the peak match window, and flag the frame.

### 3.3 Provenance is part of the payload

Every frame header carries `align_tier` (2 bits) and `align_alpha` (int8, log
scale). The decoder is deterministic and the UI can show the analyst exactly
how much geometric confidence sits behind each row. Never silently warp.

---

## 4. Edge implementation constraints

Target: existing MachineDoctor MCU, no additional FFT, no float required.

### Persistent state

| Item | Bytes |
|---|---|
| Anchor spectrum (768 uint8) | 768 |
| Anchor peak table (40 x 5) | 200 |
| Per-bin MAD (768 uint8) | 768 |
| Band edge LUT | 32 |
| Header / GOP counter | 16 |
| **Total** | **~1.8 KB flash** |

### Working memory

Current spectrum (768 B) + log-f resample buffer (256 B) + output buffer
(256 B) ≈ **1.3 KB RAM peak**.

### Compute

Peak pick + subtract + threshold + RLE, all integer. Log-f correlation is the
expensive step at ~256 x 64 MACs and only runs on tier B/C candidates.
Budget: TODO, measure on target silicon. Estimate < 40 ms.

### Split of responsibilities

**Device performs semantic compression. Gateway performs entropy compression.**

Do not run zstd on the MCU. The device emits varint-packed layers; the gateway
applies zstd-19 with a per-machine-class trained dictionary, which is where the
last 1.3 to 1.6x comes from. Dictionary training is offline and versioned; the
dictionary ID travels in the ingest envelope, never on the device.

### Anchor scheduling

Anchor frames are ~15x the size of a residual. Schedule them into the good-RSSI
windows identified by the adaptive upload scheduler, and allow the anchor to
slip up to 36 hours to catch a favourable window. Never slip past 15 days.

---

## 5. Reconstruction and display

### 5.1 Decode path

```
S_k = clamp(S_anchor_warped + g_global + g_band[band(b)] + peaks_k + residual_k)
```

Server-side, materialize into the `spec_render` rail on ingest so the waterfall
pyramid never touches the codec.

### 5.2 Making it look uncompressed, honestly

Three techniques, in descending order of defensibility.

**Frequency-axis upsample (fully safe).** Render 768 bins into a 1200 px canvas
with monotone cubic interpolation. Peak amplitude and sub-bin offset were both
preserved, so no peak is invented or displaced. This is pure display.

**Noise floor texture (safe with labelling).** A reconstructed spectrum is
smoother than a real one below the threshold, and analysts read that smoothness
as "fake" even when the diagnosis is identical. Store per-band floor variance in
the anchor (16 extra bytes) and render texture drawn from those *measured*
statistics. The texture is synthetic but its statistics are real. Must be
toggleable, must be off in any measurement or export view.

**Temporal gap fill (do not do this).** Never interpolate a missing frame into
the waterfall. Draw the gap. The moment an analyst finds a fault onset in a row
that was invented, the entire product loses credibility.

### 5.3 UI affordances

- **Provenance ribbon** on the left edge of the waterfall: anchor rows solid,
  reconstructed rows hatched, alignment tier as a 4-colour dot, gaps as voids.
- **Residual view toggle.** Render Δ_k directly as a diff waterfall, diverging
  colormap centred at zero. This is free (you already stored it) and it is
  closer to what an analyst actually wants than the absolute spectrum.
- **Anchor pinning.** Let the user pick any anchor as the reference and
  re-render the whole window as deltas against it.

---

## 6. Experiment program

### 6.1 Loss taxonomy

MSE is the wrong metric and will lead the optimization somewhere useless. Gate
on diagnostic fidelity.

**M1 Sideband fidelity** (primary gate)
- spacing error, as % of true spacing, per modulated family
- sideband-to-carrier ratio error, dB
- sideband order recovered (did the 3rd pair survive?)

**M2 Harmonic fidelity**
- H2/H1 and H3/H1 ratio error, dB
- count of harmonics above floor recovered vs truth

**M3 Broadband / haystack**
- per-octave band RMS error, dB
- noise floor log-log slope and intercept error
- haystack IoU: overlap of regions flagged elevated above floor

**M4 Absolute severity**
- overall velocity RMS error against ISO 20816 band edges
- band-limited RMS error per standard band

**M5 Downstream agreement** (the metrics that actually decide ship / no ship)
- AttentionIndex delta
- HealthStatus class flip rate
- **VQ token agreement**: encode original and reconstructed through the VibFM
  tokenizer, measure per-position token match. If tokens agree, the foundation
  model is provably indifferent to the compression. This answers "training
  value preserved" directly rather than by proxy.

**M6 Analyst blind test**
- N=200 paired spectra, original vs reconstructed, analyst picks the real one.
- Target: 50%. This makes "looks like we didn't compress" falsifiable.

### 6.2 Corpora

| Corpus | Role |
|---|---|
| SimForge synthetic | Ground truth. Generate exact sideband spacings, harmonic sets, and haystacks, so M1 to M3 have a known answer. |
| MAFAULDA | Seeded faults, external validity, published comparability. |
| Fleet sample (11M spectra) | Realism. Stratified, see below. |

Stratify the fleet sample by archetype and report every metric per stratum:
constant-speed rotating, VFD, low-speed (<100 RPM), reciprocating, gearbox,
and a batch/intermittent-duty class. Compression ratio and failure mode differ
by more than an order of magnitude across these.

### 6.3 Adversarial set

Hand-built, must pass before any config ships:
1. Incipient BPFO, 6 dB above floor, no sidebands yet
2. Tight sidebands, 0.3 Hz spacing at 1500 RPM
3. Haystack appearing at GOP day 9, absent at anchor
4. Speed change of 15% at GOP day 4
5. Repair event at GOP day 6 (spectrum returns to a much older baseline)
6. Sensor loosening (broadband rise plus mount resonance shift)
7. Low-speed asset, 25 RPM, no usable harmonic evidence
8. Two assets with near-identical anchors but divergent faults

### 6.4 Sweep design

Do not grid this. Three stages.

**Stage 1: one-factor-at-a-time.** Hold everything at the conservative setting,
sweep one factor, find its individual knee and its metric sensitivity. Cheap,
and it tells you which factors matter enough to include in stage 2.

| Factor | Levels |
|---|---|
| dB quantization step | 0.25, 0.5, 1.0, 1.5 |
| Bin count | 1536, 768, 512, 384, 256 |
| Q profile crossover order | 10, 20, 30, 50 |
| Residual threshold k | 1.5, 2.0, 3.0, 4.0 |
| Peak table size | 64, 40, 24, 16 |
| Band gain count | 32, 16, 8, 4 |
| GOP length (days) | 7, 14, 28, 56 |
| Alignment policy | always / gated / never |

**Stage 2: Latin hypercube** over the 4 to 5 factors that showed real
sensitivity, 60 to 100 configs, around their stage-1 knees.

**Stage 3: Pareto scan.** Plot bytes-per-frame against a composite diagnostic
loss and take the knee. Report the full front, not just the pick, so the
decision is auditable when someone later asks why 0.5 dB and not 1.0.

Composite loss weighting is a judgement call, so publish it explicitly:
```
L = 0.35*M1 + 0.20*M2 + 0.15*M3 + 0.10*M4 + 0.20*(1 - M5_token_agreement)
```
TODO: revisit weights after stage 1 reveals which terms actually move.

### 6.5 Acceptance gates

| Gate | Criterion |
|---|---|
| G1 | Median reconstruction error < 1.0 dB, p99 < 3.0 dB |
| G2 | Sideband spacing error < 2% in > 99% of modulated families |
| G3 | New-peak recall > 99.5% at 6 dB above floor |
| G4 | HealthStatus class flip rate < 0.1% |
| G5 | VQ token agreement > 95% |
| G6 | Analyst blind test in [45%, 55%] |
| G7 | p95 residual < 150 B, p99 < 240 B |
| G8 | Edge: < 50 ms compute, < 2 KB RAM, < 2 KB persistent |
| G9 | Zero silent alignment failures: every tier-C/D frame flagged |

G3 is the one that will bite. Threshold-based residual coding naturally
suppresses small new peaks, which are exactly the incipient faults the product
exists to find. The mitigation is in the design (new peaks bypass the threshold
entirely) but it must be verified, not assumed.

### 6.6 Order of work

1. Build the metric harness against SimForge first, with a null codec (identity)
   to confirm every metric reads perfect on unmodified data.
2. Implement the reference codec in Python, no edge constraints, full float.
3. Stage 1 sweep on SimForge + MAFAULDA.
4. Add fleet strata, rerun the surviving configs.
5. Adversarial set.
6. Fixed-point port, verify bit-exact agreement with the reference decoder.
7. Edge profiling on target silicon.
8. Blind test with the CM team.
9. Shadow deploy: run the codec alongside full-fidelity upload on 20 assets for
   one GOP, compare reconstructed against actual.

Step 9 is non-negotiable. Everything before it is simulation of a system whose
inputs you control.
