# Cloud decode, diagnostics, and transmission economics

Everything below was produced by decoding the payloads the compiled C firmware
actually transmitted. No payload was simulated.

Energy model: session setup = 20 J, payload = 40 µJ/byte (so setup costs what
500 KB costs, per the stated ratio), acquisition = 1.2 J. Battery estimates
assume an 18.5 kJ usable pack.

---

## 1. Evidence that the compression is reversible

33 transmitted frames from the gradual-fault scenario, decoded cloud-side and
compared against the quantised spectrum the device held.

| | Gradual fault | Step fault | Incipient ramp |
|---|---|---|---|
| Frames decoded | 33 | 33 | 31 |
| Anchors / residuals | 25 / 8 | 22 / 11 | 11 / 20 |
| Mean payload | 741 B | 688 B | 528 B |
| **Bins reconstructed bit-exact** | **81.5%** | **74.7%** | 52.7% |
| Median error, all frames | 0.00 dB | 0.00 dB | 1.00 dB |
| Residual frames only, median | 1.00 dB | 1.00 dB | 1.00 dB |
| Residual frames only, p99 | 5.83 dB | 6.00 dB | 5.17 dB |
| Mean residual payload | 275 B | 283 B | 328 B |

A 259-byte residual frame rebuilds into a full 768-bin spectrum whose error
against the original is invisible at the peaks and confined to the low-amplitude
floor (figure 3a). Anchors decode bit-exact by construction; the interesting
number is the residual path, and there the median bin lands within one
quantiser step.

**Harmonic families survive the round trip completely.** Across the fault
scenarios, 60 of 60 families detected on the original were also detected on the
reconstruction, and BPFI sideband spacing and sideband-to-carrier ratio matched
to 0.00% and 0.00 dB. On the incipient scenario, where residuals dominate and
the signal is weak, agreement falls to 42 of 54 and sideband spacing diverges.
That is the honest boundary of the current codec.

---

## 2. Harmonic and sideband identification, cloud-side

Both are now cloud-side, which is the right split: the search over candidate
fundamentals and modulation spacings needs no extra bytes and has no MCU budget.

- **Harmonic families**: score each candidate fundamental by the geometric mean
  of its harmonic amplitudes over the local noise floor, so a family only
  reports if the whole comb stands up, not one loud line. Then reduce to the
  lowest plausible fundamental, because a comb at 12x also scores at 6x, 4x, 3x
  and 1x, and every real family is named by its fundamental.
- **Sideband families**: autocorrelate the log spectrum in a window around the
  carrier. Two bugs were found and fixed here: the tie-plateau in the
  autocorrelation was resolving to a sub-multiple (reporting 0.495 orders where
  the true modulation is ~1.0), and near-duplicate fundamentals were not being
  suppressed.

**The important negative result.** The families the detector actually finds sit
at 10 to 12 orders, which is structural resonance modulation, not bearing tones.
BPFI sideband spacing comes out at 0.917 orders in one scenario and 1.536 in
another. This is not a detector failure. **The firmware transmits only the
acceleration rail, and inner-race sidebands live in the envelope.** Earlier work
in this thread confirmed the envelope shows BPFI at 159.7 Hz against a
theoretical 159.9 Hz, while the acceleration spectrum's low-order region is
dominated by mount lines at 40.3 and 47.6 Hz. Fixing this is improvement #1.

---

## 3. Transmission economics

Connection cadence sweep, gradual-fault scenario, 90 days:

| Connect every | Sessions | Frames/session | KB | E acquire | E radio | E bytes | E total | Detect | **Delivered** |
|---|---|---|---|---|---|---|---|---|---|
| 1 d | 33 | 1.0 | 23.9 | 432 | 660 | 0.98 | 1093 J | 0.67 d | 0.92 d |
| 3 d | 30 | 1.1 | 23.0 | 432 | 600 | 0.94 | 1033 J | 0.67 d | 2.25 d |
| 7 d | 13 | 2.5 | 23.0 | 432 | 260 | 0.94 | 693 J | 0.67 d | 1.50 d |
| 14 d | 7 | 4.6 | 23.0 | 432 | 140 | 0.94 | 573 J | 0.67 d | 3.83 d |
| **21 d** | **4** | **6.8** | **18.7** | 432 | 80 | 0.76 | **513 J** | 0.67 d | **1.50 d** |
| 28 d | 4 | 8.0 | 23.0 | 432 | 80 | 0.94 | 513 J | 0.67 d | 8.50 d |

Payload bytes cost **0.76 J across the entire 90 days**. Radio setup costs 80 to
660 J. At your ratio, the byte count is a rounding error and the session count
is the only thing that matters. My earlier conclusion that batching barely helps
was an artifact of using a 5,300-byte-equivalent setup cost instead of 500,000.

### Should a step change break the schedule?

| Policy | Sessions | E total | Detect | Delivered |
|---|---|---|---|---|
| 3-day cadence, no burst (previous default) | 29 | 1013 J | 0.67 d | 2.25 d |
| 21-day cadence, no burst | 4 | 513 J | 0.67 d | 13.00 d |
| 21-day, burst on **every** change | 23 | 893 J | 0.67 d | 0.67 d |
| **21-day, burst on episode onset only** | **4** | **513 J** | **0.67 d** | **1.50 d** |

Bursting on every change during a progressing fault costs 19 extra sessions to
buy 0.83 days. Bursting only on the **onset of an episode** costs nothing
measurable and delivers in 1.5 days. This required a firmware fix: the step is
detected on the acquisition where the jump occurs, but the 2-of-3 persistence
delays the upload by one acquisition, by which time the jump is absorbed and the
event was being reclassified as drift. The step is now latched for the episode.

### Against the naive baseline

| | Sessions | Data | Energy | Battery |
|---|---|---|---|---|
| Upload every running acquisition, raw spectrum | 180 | 5,761 KB | 4,268 J | 1.1 y |
| Previous default (3-day cadence) | 29 | 23.0 KB | 1,013 J | 4.4 y |
| **Tuned (21-day cadence, onset burst)** | **4** | **18.7 KB** | **513 J** | **8.9 y** |

**45x fewer sessions, 309x less data, 88% less energy, and delivery latency
improves from 2.25 to 1.50 days.** The tuned policy beats the previous default
on every axis simultaneously.

---

## 4. Top ten improvements

Ranked by expected value, with the evidence each rests on.

**1. Transmit the envelope rail, not just acceleration.** §2 above. Bearing
diagnosis is in the envelope; the current payload cannot support it. The
envelope rail is 384 bins against 768, so the cost is roughly +300 B per anchor
and +100 B per residual. At 0.76 J of payload energy across 90 days, this is
free. Highest-value change in the list by a wide margin.

**2. Adopt the 21-day cadence with onset-only burst.** §3. Measured: 4 sessions
instead of 29, battery 4.4 → 8.9 years, delivery latency improved. Already
implemented and controllable; it is a default change, not new code.

**3. Attack acquisition energy, which now dominates.** At 21-day cadence,
acquisition is 432 J of 513 J total, or 84%. Radio is no longer the bottleneck.
An adaptive wake interval (12 h when quiet, 3 h while `in_change` is latched)
would roughly halve the dominant term while *improving* fault-onset resolution.
This is now the single biggest remaining lever and it did not exist before
batching was fixed.

**4. Cut the anchor share.** 25 anchors against 8 residuals in the gradual
scenario (figure 3g); anchors are 890 B against 275 B. The early-anchor trigger
fires whenever peak match rate drops, which during a progressing fault is
constantly. Hold two references, the healthy commissioning anchor and the
current one, and encode against whichever is closer. Expect residual share to
roughly invert.

**5. Persist state to NVM.** Nothing survives a power cycle: baseline, slow
reference, CUSUM accumulators, level histogram, and the codec anchor are all
lost. Every reset costs a 48-sample rediscovery, which at 6-hourly wake is 12
days of blindness. About 3.5 KB needs to persist.

**6. Cut RAM from 85.8 KB to roughly 45 KB.** Almost all of it is the 4096-point
FFT and envelope workspace. `EG_NFFT` at 2048, a real-input FFT rather than
complex, and a 4-slot queue get there. Without this it will not fit the target
silicon.

**7. Per-band CUSUM, not just per-feature.** Drift is currently detected on the
seven scalar features. A fault that raises one order band by 4 dB while leaving
overall RMS flat is invisible until it grows. The band deltas are already
computed and transmitted; running CUSUM on all 16 costs 128 B of state and
should detect band-localised degradation considerably earlier than the
27-day figure measured on the incipient scenario.

**8. Multi-state operating discovery.** The histogram currently splits into two
modes. Real assets run at several loads or speeds, and each deserves its own
baseline; a load change is currently either a false alarm or something the
absolute floor has to suppress. Otsu extends to K classes, or switch to a
1-D streaming mixture. The validity gates carry over unchanged.

**9. Let the operator set delivery latency, not cadence.** `conn_interval_s`,
`conn_burst_max` and `batch_max_age_s` interact in ways the sweep above shows
are not monotonic: 21 days delivers in 1.50 days while 28 days delivers in 8.50.
Nobody should be tuning that by hand. Expose a target delivery latency and have
the device derive the cadence.

**10. Sign the config and add a CRC.** `fallback_after_s` protects against a bad
config, but not against a malformed or hostile one. Also missing: staged rollout
by cohort, and an explicit time-sync path, since the sliding rate window assumes
a monotonic clock.

---

## 5. Caveats

1. Idle acquisitions remain synthesized; running acquisitions are real CWRU.
2. Energy constants are order-of-magnitude LTE-M figures pinned to your stated
   500 KB ratio, not measurements. The *ratios* between rows are the reliable
   part, not the absolute joules.
3. Battery life extrapolates 90 days to a year and ignores self-discharge,
   temperature, and quiescent draw. Treat as an upper bound.
4. Detection latencies on the CWRU severity steps are optimistic; the blended
   incipient scenario at 27 days is the honest figure.
5. Harmonic family labelling is limited by the acceleration-only payload, per §2.

## 6. Files

- `fig3_cloud_energy.png` — decode evidence, error distribution, energy
  breakdown, cadence sweep, burst policy, harmonic recovery over time.
- `code/s10_cloud.py` — payload decoder, harmonic and sideband family detection.
- `code/s11_energy.py` — energy model, cadence and burst sweeps, figure.
- `firmware/` — updated with step/drift classification, step latching,
  connection scheduling and onset-burst policy.
