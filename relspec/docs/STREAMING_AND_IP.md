# Streaming spectral extraction: feasibility, architecture, and IP position

## 1. The constraint is not what it looks like

Frequency resolution is set by **coherent observation time**, not by how many
samples you hold at once. Those get conflated because the FFT happens to require
both. Separate them and the constraint dissolves.

A heterodyne integrator locked to the tracked shaft angle accumulates a single
order in three registers, for as long as you like:

```
theta  += f_shaft * dt                      (running shaft angle, revolutions)
A_k    += w[n] * x[n] * exp(-j*2*pi*k*theta)
```

Blocks are consumed and discarded. Nothing but `A_k` persists. Resolution grows
as 1/T while memory stays flat.

**And under speed variation this beats a long FFT rather than approximating it.**
A long FFT assumes a stationary frequency; when the shaft wanders, the energy of
a component sitting at a fixed *order* smears across many Hz bins. The
integrator mixes down by the tracked angle, so that component stays at DC no
matter what the shaft does.

---

## 2. Measured

CWRU record 209, imposed smooth speed wander, identical signal both ways.

| Record | Wander | Method | Peak mem | 1x width | 1x amp | 3x width | 3x amp |
|---|---|---|---|---|---|---|---|
| 4 s | 0% | batch | 887 KB | 0.0150 | 0.42 | 0.0150 | 0.31 |
| 4 s | 0% | **stream** | **65 KB** | **0.0100** | **0.95** | **0.0100** | **0.61** |
| 4 s | 3% | batch | 887 KB | 0.0225 | 0.34 | 0.0475 | 0.16 |
| 4 s | 3% | **stream** | **65 KB** | **0.0100** | **0.85** | **0.0125** | **0.42** |
| 10 s | 3% | batch | 1962 KB | 0.0300 | 0.20 | 0.0525 | 0.09 |
| 10 s | 3% | **stream** | **65 KB** | **0.0050** | **0.74** | **0.0075** | **0.31** |

At 10 s with 3% wander the third harmonic is **7x sharper and 3.4x stronger**
streaming than batch, on 30x less memory. The amplitude gap is the point: batch
does not just blur the peak, it loses the energy, which is exactly what breaks
sideband-to-carrier ratios and severity trending.

Memory scaling is the trivial part:

| Observed | Batch | Streaming | Ratio |
|---|---|---|---|
| 20 s | 3.9 MB | 65 KB | 61x |
| 5 min | 59 MB | 65 KB | 943x |
| 1 h | 842 MB | 65 KB | 13,348x |
| 24 h | 20 GB | 65 KB | 318,000x |

The 65 KB includes 2081 probe orders for the plot. A deployment tracks ~400
targeted orders and lands near 39 KB, which fits the RAM budget that the current
block-FFT design blows.

---

## 3. The real limit: phase coherence, not memory

The 20 s rows regress. That is not noise, it is the governing constraint, and it
sets the whole design.

If the speed estimate carries fractional error `e`, phase error at order `k`
after time `T` is `2*pi*k*e*f*T`. Coherent gain survives while that stays under
about `pi/2`:

```
T_coh(k)  =  1 / (4 * k * e * f)
```

With the measured `e = 0.05%` and `f = 29 Hz`: order 1 stays coherent for ~17 s,
order 3 for ~5.7 s, order 20 for ~0.9 s. That predicts the observed 20 s
degradation at 3x exactly.

Three consequences that shape the architecture:

1. **Coherent integration time must be chosen per order.** Low orders can
   integrate for minutes; high orders cannot. Resolution therefore *should* be
   allocated non-uniformly across the axis, which is the same hybrid-Q split
   already in the codec, now derived from first principles rather than picked.
2. **Beyond `T_coh`, switch to incoherent accumulation** of `|A_k|^2` over
   coherent segments. Resolution stops improving but variance keeps falling.
   This is the standard coherent/incoherent tradeoff from radar and pulsar
   search, applied per order.
3. **Better speed tracking directly buys resolution.** A closed-loop PLL that
   drives residual phase error to zero, rather than open-loop integration of a
   per-block estimate, extends `T_coh` by whatever it improves `e`.

---

## 4. Proposed streaming architecture

```
samples in (12 kHz, never stored)
  |
  +-- Welford moments                        O(1)   rms, kurtosis, crest
  |
  +-- shaft phase PLL  ------> theta(t)      O(1)   closed loop, not open
  |
  +-- TSA accumulator, one revolution buffer O(rev) synchronous rail
  |     running mean over M revolutions             order resolution 1/M
  |     residual = x - TSA(theta mod 1)     ------> non-synchronous content
  |
  +-- heterodyne bank, K targeted orders     O(K)   fine rail, 0..20 orders
  |     coherent to T_coh(k), incoherent beyond
  |
  +-- envelope: biquad -> |.| -> biquad -> decimate
  |     then its own heterodyne bank         O(K')  bearing rail
  |
  +-- block Welch |X|^2 running mean         O(B)   coarse rail, 20..200 orders
        (stochastic content: averaging is correct here, coherence is not)
```

Budget: TSA 8 KB, heterodyne banks ~10 KB, Welch accumulator 3 KB, one working
block 16 KB, filter state under 1 KB. **About 39 KB, independent of how long the
acquisition runs.**

Two structural points worth stating plainly.

**Different estimators for different halves of the axis.** Deterministic content
(shaft harmonics, gear mesh, their sidebands) wants coherent accumulation.
Stochastic content (broadband, resonance haystacks) wants incoherent averaging.
The current design uses one estimator for both and compromises on each.

**TSA separates the two rails for free and is inherently streaming.** One
revolution of buffer, a running mean over thousands of revolutions. The average
is the synchronous content at 1/M order resolution; the residual is the
non-synchronous content where bearing tones live. That is the classical
gearbox/bearing separation and it needs no stored waveform.

### What this unlocks beyond memory

Because the gating statistics are running quantities, the upload decision is
available *before* the acquisition finishes. That permits **early termination**:
acquire until a sequential test on the running features either clears the asset
or triggers, then stop. Given that acquisition is now 84% of the energy budget
(432 J of 513 J), stopping a healthy acquisition at 1 s instead of 4 s is a
larger saving than everything the radio work achieved.

---

## 5. IP position, honestly

I am not a patent attorney and this is not legal advice. What I can do is
separate what is old from what might not be.

### Clearly prior art, do not claim

| Element | Origin |
|---|---|
| Goertzel / single-bin recursive DFT | Goertzel 1958 |
| Time synchronous averaging | rotating machinery literature, 1970s |
| Order tracking, angular resampling | Fracture/Bruel & Kjaer, 1980s |
| Vold-Kalman order filtering | Vold & Leuridan 1993 |
| Welch averaging | Welch 1967 |
| Coherent/incoherent integration tradeoff | radar and radio astronomy, decades |
| Envelope demodulation for bearings | standard practice |
| CUSUM, SPRT | Page 1954, Wald 1945 |
| Streaming moments | Welford 1962 |

Every building block is decades old. Any claim over a block on its own will
fail. This matters: it is the most common way an application dies.

### Plausibly novel as a combination

The candidates I would ask an attorney to search, in descending order of how
defensible they look:

**A. Order-dependent coherent integration time.** Selecting per-order coherent
accumulation length from the *measured* shaft-speed uncertainty, via
`T_coh(k) = 1/(4 k e f)`, and switching to incoherent accumulation beyond it. So
frequency resolution is allocated across the order axis as a function of tracked
speed confidence, adaptively, at run time. I have not seen this stated as a
control law. It is also the thing with a clean measured effect (§3).

**B. Early-terminated acquisition under a sequential test.** Acquisition length
governed by a running sufficient statistic computed from the same streaming
accumulators used for the anomaly decision, so a healthy machine is sampled
briefly and a changing one is sampled long. The energy argument is quantified
and non-obvious: it inverts the usual fixed-duration acquisition.

**C. Encoding directly from streaming accumulators.** The residual codec
operating on accumulator state rather than on any stored or fully-formed
spectrum, with the anchor reference held in the same order-locked coordinates,
so compression, change detection and acquisition control share one state.

**D. Unsupervised operating-state gating driving all of the above.** The Otsu
histogram with its gap-validity test deciding not just transmission but whether
the accumulators integrate at all, so coherent integration only spans
same-state intervals. The validity test that refuses to invent an idle state on
a unimodal machine is a specific, describable mechanism.

Weakest of the four is C, which reads as an engineering consequence. Strongest is
probably A, because it is a rule with a formula, a measurable effect, and a
non-obvious inversion (more speed confidence buys more resolution, so improving
the tacho estimate improves the spectrum, which is not how anyone thinks about
tacholess methods).

### What I would actually do

1. **Do not file yet.** Run a proper search first: CPC G01H, G01M13/045,
   F16C19/52, and assignee searches on SKF, Schaeffler, Emerson/CSI, Bruel &
   Kjaer, Augury, Petasense, Erbessd. Most of this space is crowded.
2. **File a provisional** once the search is clean on at least one of A or B.
   Cheap, buys 12 months, and lets you keep publishing and testing.
3. **Get the measurement into the application.** Software and signal-processing
   applications live or die on a demonstrated, quantified, non-obvious effect.
   The 7x sharpness at 3.4x amplitude under drift, and the 13,348x memory ratio,
   are the kind of numbers that carry an application. Keep the harness that
   produced them.
4. **Decide patent vs trade secret deliberately.** A patent publishes the method.
   For something that runs inside a sealed sensor and is hard to reverse
   engineer from telemetry, trade secret can be the stronger position. A patent
   is worth more if the value is in licensing or in blocking a named competitor.
5. **Watch the disclosure clock.** Public use or disclosure starts a 12-month
   grace period in the US and can immediately bar filing in Europe. If any of
   this has been shown to customers or shipped, establish those dates now,
   before anything else.

Item 5 is the one that catches people. Everything else can be recovered from.

---

## 6. Caveats on the experiment

1. The speed wander is imposed by time-warping a real record, not measured on a
   genuinely variable-speed machine.
2. Speed tracking here is open loop, a per-block envelope estimate integrated
   into an angle. The PLL proposed in §4 is not implemented, so `T_coh` in
   practice should be better than measured.
3. TSA and the envelope heterodyne rail are designed but not built; only the
   heterodyne bank and the batch comparison were run.
4. The 39 KB figure is an accounting estimate, not a compiled measurement.
