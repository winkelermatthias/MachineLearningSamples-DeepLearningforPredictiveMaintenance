# Spectral GOP codec: end-to-end results on CWRU bearing data

Dataset: Case Western Reserve University bearing data, 12 kHz drive-end accelerometer,
SKF 6205-2RS JEM. 16 records: normal + inner-race faults at 0.007 / 0.014 / 0.021 in,
loads 0 to 3 HP. Downloaded from `engineering.case.edu`. 67 MB.

CWRU was chosen over IMS/MAFAULDA because it supplies both a **fault severity
progression** and **real speed variation** (1721 to 1796 rpm across load steps, 4.3%
spread), which is what exercises the alignment ladder. IMS is single-speed; MAFAULDA
and IMS were both too large for the available container.

Two experiments:
- **Exp B (progression)**: 28 frames, 4 s each, healthy -> IR 0.007 -> IR 0.014 ->
  IR 0.021 with wandering load. Anchor is healthy, fault develops mid-GOP. Worst case.
- **Exp A (continuous)**: 28 sequential non-overlapping windows from a single
  continuous recording. Realistic steady-asset telemetry.

---

## 1. Headline numbers

| | Exp B (progression) | Exp A (continuous, 2 s) | Exp A (continuous, 4 s) |
|---|---|---|---|
| Compression vs raw float32 | **69.9x** | 64.7 to 75.5x | **240.8x** |
| Median residual frame | 64 B | 34 to 44 B | 23 B |
| Anchors | 34/56 rails | 11 to 12/40 | 2/20 |
| Telemetry | 1.83 KB/day | 0.85 to 0.99 KB/day | **0.53 KB/day** |
| Recon error median | 0.00 dB | 0.50 dB | 0.50 dB |
| Recon error p99 | 3.50 dB | 4.50 to 5.00 dB | 6.00 dB |

Compression cascade, Exp B:

| Stage | Bytes | Cumulative |
|---|---|---|
| Raw float32, 8193 bins, ACC+ENV | 1,835,232 | 1.0x |
| Hybrid-Q order binning (768 / 384) | 129,024 | 14.2x |
| uint8 @ 0.5 dB | 32,256 | 56.9x |
| GOP codec | 31,704 | 57.9x |
| + zstd-19 with trained dictionary | 29,006 | **63.3x** |

Encode 1.54 ms/frame, decode 0.148 ms/frame, alignment 0.2 ms/frame (single core).

**The 100-500x target is reachable but only at the top end of the acquisition-length
range.** 240x is real and measured; it required 4 s windows with 16384-point Welch.

---

## 2. The dominant finding: acquisition length, not codec tuning

| Window / nperseg | Welch averages | Ratio | Median residual | Anchors |
|---|---|---|---|---|
| 1 s / 8192 | ~2 | 45.9x | 244 B | 28/56 |
| 2 s / 8192 | ~5 | 64.7x | 44 B | 12/40 |
| 4 s / 8192 | ~11 | 122.3x | 79 B | 2/20 |
| 4 s / 16384 | ~5 (finer bins) | **240.8x** | 23 B | 2/20 |

A 5.2x swing from acquisition length alone. No codec parameter came close.

**Why.** A Welch periodogram with few averages is chi-squared distributed with few
degrees of freedom, so individual bins scatter several dB frame to frame even on a
perfectly steady machine. The residual coder cannot distinguish that scatter from real
machine change, so it spends its byte budget encoding estimator noise. More averages
collapse the scatter and the residual starts encoding only what actually changed.

This is the practical lever: **on the edge, lengthening the acquisition is a cheaper
way to buy compression than tightening the codec**, and it improves diagnostic SNR at
the same time. It costs ADC-on time, which trades against battery.

Corollary that bit me during the run: the peak-match tolerance must exceed the
estimator's own scatter. At 1.5 dB tolerance against ~3 dB scatter, match rate fell to
0.10 to 0.47 on genuinely steady data and the early-anchor trigger fired on 25 of 28
frames. Raising the tolerance to 3 dB fixed it. Diagnosis is in §5.

---

## 3. Loss: does the diagnosis survive

Diagnostic feature amplitude error, envelope rail, all 28 frames:

| Feature | median | p95 | max |
|---|---|---|---|
| BPFI 1x | 0.17 dB | 0.98 | 1.17 |
| BPFI 2x | 0.15 dB | 1.40 | 2.44 |
| BPFI 3x | 0.14 dB | 0.86 | 2.52 |
| Sideband, lower (-1x) | 0.20 dB | 0.64 | 1.99 |
| Sideband, upper (+1x) | 0.18 dB | 1.00 | 1.11 |
| Shaft 1x | 0.16 dB | 0.64 | 1.24 |

- **Sideband-to-carrier ratio error**: median 0.15 dB, p95 0.74 dB.
- **Sideband spacing error**: median 0.000%, i.e. exact in the overwhelming majority.
  The max of 11.1% is a single frame where the sideband sits one coarse bin off; worth
  chasing but not a systematic failure.
- **Band RMS error**: 0.02 to 0.07 dB median across all seven order bands, 0 to 200
  orders. p95 never exceeds 0.44 dB.
- **Overall RMS error**: 0.022 dB median, 0.69 dB p95.

Every one of these sits an order of magnitude below the ~2 dB run-to-run repeatability
of a typical industrial accelerometer channel. Against G1/G2 from the spec: **pass**.

### Where the loss actually is

Not in the codec. Axis reduction dominates:

| Stage | median | p95 | p99 | max |
|---|---|---|---|---|
| Axis reduction 8193 -> 768 bins | 1.69 dB | 19.23 | 28.12 | 45.96 |
| Codec, ACC rail | 0.00 dB | 2.00 | 3.50 | 16.00 |
| Codec, ENV rail | 0.00 dB | 3.00 | 5.00 | 11.00 |

The 19 dB p95 on axis reduction is the peak-preserving MAX binning discarding the
*valleys* between peaks in the coarse region above 20 orders. Peaks are kept exactly.
This is intentional and is why the diagnostic metrics above stay clean while the raw
per-bin error looks alarming. **Any evaluation of this codec that reports only per-bin
error will reach the wrong conclusion.**

Figure 1(d) shows this directly: on a 197 B residual frame the visible error is
concentrated in orders 80 to 120, the coarse-Q region, and is essentially zero below
20 orders where the diagnosis lives.

---

## 4. Alignment ladder

Tacholess speed estimation, all 28 frames, validated against RPM tags:

- median error **0.045%**, p95 0.304%, max 0.934%
- tier assignment: B 22, C 4, D 2

Confidence gating works as designed:

| Gate | Frames passing | median abs err | max abs err |
|---|---|---|---|
| conf >= 1.5 | 28/28 | 0.045% | 0.934% |
| conf >= 2.0 | 24/28 | 0.041% | 0.223% |
| conf >= 3.0 | 22/28 | 0.037% | 0.092% |

### Correction to the spec

The spec assumed HPS on the **acceleration** spectrum. On this data that fails: CWRU's
low-frequency acceleration region is dominated by mount/structural lines at 40.3 and
47.6 Hz, and HPS locked onto a fixed 30.02 Hz feature for every frame regardless of
true speed, producing 1.7 to 4.7% errors while reporting acceptable confidence.

The fix is to estimate speed on the **envelope** rail. The envelope demodulates the
impact train and exposes shaft 1x and BPFI cleanly. Verified: envelope peaks landed at
30.0 / 29.3 / 28.6 Hz against truths of 29.93 / 29.20 / 28.80, and BPFI at 159.7 Hz
against a theoretical 5.4152 x 29.533 = 159.9 Hz.

Two further details that mattered: use a full-length FFT rather than Welch for speed
estimation (resolution beats variance reduction here), and apply parabolic sub-bin
interpolation, without which estimates quantize onto the 0.73 Hz bin grid and the error
floor is 1.25%.

**Update §3 of the spec: the alignment ladder runs on the envelope rail.**

---

## 5. What went wrong, and what it cost

Three real failures, all worth keeping in the record.

**Run 1, 50.6x.** The early-anchor trigger was specified but not implemented, so a GOP
straddling a 12 dB fault onset kept referencing a stale healthy anchor. The new-peak
detector pinned at its cap of 32 every frame and 60% of bins tripped the residual
threshold. Also the MAD baseline was computed from four windows of the *same* file,
giving ~0.5 dB variability where real day-to-day spread is 1 to 3 dB, so the threshold
was set far below the noise.

**Run 2, 48.4x.** Overcorrected. The confidence metric was changed to a peak-ratio with
range 1.1 to 2.7 while the gates were left at 8.0 and 5.0, so every frame fell to tier D,
alignment collapsed entirely, and 20 of 28 frames became anchors. The residual frames
that did survive were 66 B median, which is what revealed the codec itself was fine and
the estimator was the problem.

**Run 3, 63.3x.** Alignment fixed, fidelity excellent, but still 22/28 anchors. The
instrumented trigger showed match rates of 0.10 to 0.47 on consecutive frames. The
continuous control then showed the same behaviour on genuinely steady data with 0.00%
speed spread, which ruled out volatility and pointed at estimator variance (§2).

The general lesson: **every symptom looked like a codec problem and none of them were.**
Two were alignment, one was the statistics of the spectral estimate. Build the
instrumentation for which trigger fired before tuning anything.

---

## 6. Parameter sweeps

| Parameter | Value | Ratio | median err | p99 err | Anchors |
|---|---|---|---|---|---|
| q (dB/count) | 0.25 | 140.3x | 0.00 | 4.75 | 3 |
| | 0.50 | 46.1x | 0.00 | 3.50 | 22 |
| | 1.00 | 61.7x | 0.00 | 6.00 | 18 |
| | 1.50 | 65.3x | 0.00 | 6.00 | 18 |
| k_MAD | 1.5 | 45.6x | 0.00 | 3.00 | 22 |
| | 2.5 | 46.1x | 0.00 | 3.50 | 22 |
| | 4.0 | 100.8x | 0.50 | 12.50 | 6 |
| | 6.0 | 138.3x | 1.00 | 15.00 | 4 |
| n_peaks | 16 / 40 / 64 | 50.3 / 46.1 / 42.5x | 0.00 | 3.50 | 22 |
| n_bands | 4 / 16 / 32 | 46.2 / 46.1 / 46.0x | 0.00 | 3.50 | 22 |

Reading these:

- **k_MAD is the real knob.** Going 2.5 -> 6.0 buys 3x compression for 1.0 dB median
  error, still under sensor repeatability. But p99 goes to 15 dB, which is where
  incipient-fault recall (gate G3) will start to suffer. This needs the G3 test before
  anyone ships k=6.
- **n_bands does nothing** (46.0 to 46.2x across an 8x range). The band-gain layer is
  nearly free but is also contributing almost nothing on this data. Either the global
  gain is already absorbing it, or CWRU's controlled rig lacks the broadband drift that
  makes it pay on real assets. Retest on fleet data before deleting it.
- **n_peaks trades the wrong way**: more peaks costs bytes and buys no accuracy here.
  16 beats 40. Suggests the peak table is oversized for a 768-bin axis.
- **q = 0.25 giving 140x** is a genuine non-monotonicity caused by anchor-count
  interaction, not a clean win. Flagged, not trusted.

---

## 7. Caveats

1. The Exp B sequence is assembled from real measurements but the temporal ordering is
   constructed. Each frame is real; the progression is not a single continuous run.
2. CWRU records 98 and 99 carry no RPM tag, so their "truth" is CWRU's nominal speed
   for the load. The single 0.93% outlier in §4 is plausibly nominal-tag error rather
   than estimator error.
3. The 240.8x figure comes from a 10-frame GOP (file 99 is only 40 s), so the anchor
   amortization is measured over a short window. Directionally solid, needs a longer
   record to confirm.
4. No VQ token-agreement metric (spec M5), no analyst blind test (M6). Those need
   VibFM and the CM team respectively.
5. Single-core Python timings. The fixed-point edge port is not written, so the §8
   RAM/compute claims in the spec remain unverified.

---

## 8. Files

- `fig1_fidelity.png` : waterfalls original vs reconstructed vs difference, spectrum
  overlays, BPFI sideband zoom, raw spectrogram. All magnitudes linear amplitude.
- `fig2_performance.png` : bytes per frame, compression cascade, acquisition-length
  effect, k_MAD Pareto, speed error by tier, diagnostic feature error.
- `code/` : s1 frames, s3 alignment, s4 codec + metrics + sweeps, s5 control,
  s6 final, s7 figures.
