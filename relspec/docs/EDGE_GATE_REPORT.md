# Edge transmission gate: implementation and evaluation

Firmware in C, driven end to end over seven 90-day scenarios built from real CWRU
bearing measurements. Every decision reported here was produced by the compiled
firmware, not by a Python model of it.

---

## 1. What the device does each wake cycle

```
wake (every 6 h)
  -> acquire 1 s @ 12 kHz
  -> features: acc rms, velocity rms (ISO 20816), acc kurtosis, crest,
               envelope rms / kurtosis / crest, 16 band levels, 8 tracked orders
  -> classify operating state   (learned locally, no commissioning)
       IDLE or TRANSIENT -> stop. No transmission. No baseline update.
  -> score change vs baseline   (robust z, p-mean, spectral mask, CUSUM drift)
  -> gate: change OR heartbeat, subject to persistence, dead time, rate cap
  -> encode (anchor or residual) -> queue
  -> open radio only when the batch is worth a session
```

Two upload conditions, as specified:
1. **Meaningful change** from baseline, and
2. **Heartbeat**: nothing sent for 3 days.

Both bounded by a hard sliding-window cap of 14 uploads per 7 days.

---

## 2. Headline results

Baseline configuration, 360 wake cycles over 90 days per scenario.

| Scenario | State acc | Idle suppressed | Uploads | Change | Heartbeat | Up/week | Busiest week | Sessions | Sent | Detect latency | False alarms | Energy saved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Continuous healthy | n/a | 0 | 30 | 0 | 29 | 2.33 | 3 | 30 | 16.2 KB | — | 0 | 96.7% |
| Duty healthy | 100% | 157 | 30 | 0 | 29 | 2.33 | 3 | 30 | 10.9 KB | — | 0 | 93.6% |
| Duty + step fault | 100% | 157 | 33 | 17 | 15 | 2.57 | 5 | 32 | 22.2 KB | **0.5 d** | 0 | 93.3% |
| Duty + gradual | 100% | 157 | 33 | 22 | 10 | 2.57 | 4 | 32 | 23.9 KB | **0.5 / 0.25 / 1.25 d** | 0 | 93.2% |
| Duty + load cycling | 100% | 157 | 30 | 0 | 29 | 2.33 | 3 | 30 | 15.7 KB | — | 0 | 93.5% |
| Irregular duty + fault | 100% | 139 | 31 | 16 | 14 | 2.41 | 4 | 31 | 18.9 KB | **0.25 d** | 0 | 94.2% |
| Duty + incipient ramp | 100% | 157 | 38 | 11 | 26 | 2.96 | 4 | 28 | 23.7 KB | 27.25 d | 0 | 93.5% |

- **Operating state classification is exact**: 157/157 idle and 156/156 running
  correct in every duty-cycled scenario, 0 errors. The remaining 47 acquisitions
  per scenario are labelled `unknown` during warmup, which is correct behaviour.
- **Zero false alarms** across all three no-fault scenarios, including the one
  where load changes every week.
- **Upload rate averages 2.33 to 2.96 per week** against a 14 cap. The busiest
  single week anywhere was 5. The cap is never the binding constraint under
  default settings, which is the right place for it to sit.
- **Detection latency is one to two acquisitions** (0.25 to 0.5 days) for step
  and severity-step faults.
- **93 to 97% energy reduction** against uploading every running acquisition as
  a raw spectrum. Energy model: 1.2 J per acquisition, 8.0 J per radio session,
  1.5 mJ per byte. These are assumptions, not measurements.

---

## 3. Idle discovery: how it learns, and the ablation

The device keeps a decaying 64-bin histogram of log level and recomputes Otsu's
threshold every 16 samples. Otsu is the right tool: O(bins), deterministic, no
tuning constants, and its separability figure doubles as a validity test.

Four conditions must all hold before the split is trusted: enough samples (48),
separability above 0.55, both modes at least 5% populated, and the two modes at
least 6 dB apart. The gap test is the one that matters. Otsu always returns *a*
threshold, including on a unimodal distribution.

**The continuously-running scenario correctly refuses to invent an idle state.**
Separability reads 1.00 but the modes are only ~1 dB apart, the gap test fails,
discovery stays invalid, and every acquisition is kept. That failure mode is
asymmetric and worth stating plainly: wrongly declaring RUNNING costs a little
battery, wrongly declaring IDLE silently discards the data the device exists to
collect.

### Ablation: idle gating disabled

| Scenario | State acc | Uploads | Change events | False alarms | Fault detected |
|---|---|---|---|---|---|
| Duty healthy | 50% | 35 | 9 | 9 | n/a |
| Duty + step fault | 50% | 52 | 38 | 38 | **no** |
| Duty + gradual | 50% | 55 | 44 | 44 | **no** |
| Irregular + fault | 57% | 50 | 29 | 29 | **no** |

This is the result that justifies the whole subsystem. With idle frames allowed
into the baseline, the learned spread widens to span both operating states. Every
start-up then reads as a large anomaly, the device burns 38 to 44 change uploads
on nothing, and **the real fault is never detected at all** because the baseline
is now wide enough to contain it. Upload volume roughly doubles while diagnostic
value goes to zero.

---

## 4. Change detection: what it took to work

The scalar features are tracked with a streaming robust baseline (sign-update
median plus EWMA absolute deviation, O(1) state per feature, no history buffer).
Robust z-scores are aggregated with a p-mean at p=3, which emphasises the largest
deviation without collapsing to a pure max.

Four independent brakes keep it from being twitchy:

1. **N-of-M persistence** (2 of 3) before a change can fire.
2. **Hysteresis**: enter at score 4.0, exit at 2.5.
3. **Dead time** of 12 h after a change upload, overridden only by escalation.
4. **Absolute significance floor**: the change must also be physically
   meaningful (0.8 mm/s velocity or 2.5 dB acceleration), not merely
   statistically unusual. Without this a very quiet machine with a tiny MAD trips
   on nothing.

Plus a fifth added during evaluation:

5. **Change fatigue.** A fault that is present and stable does not need
   re-reporting twice a day forever. After three change uploads without
   escalation, the dead time multiplies by six. This cut the gradual scenario
   from 71 uploads (60 change) to 33 (22 change) with no loss of detection: dense
   sampling at onset, sparse sampling once the condition is known.

### The failure that mattered most

The first working version detected step faults perfectly and was **completely
blind to gradual degradation**. On the blended incipient scenario, envelope
kurtosis rose 3.14 to 5.84 and acceleration RMS by 3.3 dB over 60 days, and the
anomaly score never moved off 1.0.

Cause: the adaptive baseline tracked the ramp and absorbed it. This is the
classic trade with adaptive references, and it is invisible in step-change tests.

Fix: a **second, slow reference** that freezes after 120 running samples, plus a
two-sided CUSUM against it. Small persistent shifts accumulate; the fast baseline
still handles load and temperature. First detection on the incipient ramp moved
from never to day 52.25.

CUSUM tuning, measured against detection and false alarms:

| k | h | Incipient detected | Healthy false alarms (3 scenarios) |
|---|---|---|---|
| 0.5 | 10 | day 53.25 | 0 / 0 / 1 |
| 1.0 | 10 | day 54.25 | 0 / 0 / 0 |
| 1.0 | 20 | day 58.25 | 0 / 0 / 0 |
| 1.0 | 35 | never | 0 / 0 / 0 |
| 1.5 | 20 | never | 0 / 0 / 0 |

Shipped at k=1.0, h=10.

A related bug is worth recording: I initially let the drift term bypass the
absolute significance floor. That produced 13, 9 and 10 false alarms in the three
healthy scenarios, because CUSUM will eventually reach any threshold on a
stationary signal with a slightly mis-centred reference. Requiring drift to also
clear the physical floor took all three to zero.

### Sensitivity sweep

On the blended incipient scenario (the seeded CWRU faults are 12 dB steps, far
too large to test sensitivity meaningfully):

| s_hi | Incipient uploads | Change events | Detection | Load-cycling false alarms |
|---|---|---|---|---|
| 2.0 | 36 | 25 | day 8.5 | 4 |
| 2.5 | 32 | 4 | day 26.5 | 1 |
| 3.0 | 33 | 4 | day 26.5 | 1 |
| 4.0 | 31 | 2 | day 29.25 | 0 |
| 5.5 | 32 | 2 | day 29.5 | 0 |

s_hi = 2.0 detects 21 days earlier at the cost of 4 false alarms on a machine
whose load simply changes weekly. That is the knob a fleet operator should own,
and it is exposed in the cloud config.

---

## 5. What is changing, in the frequency domain

Every upload header carries a compact change fingerprint, computed on device:

- `band_delta_db[16]` — signed change per order band, 0 to 200 orders
- `order_delta_db[8]` — signed change at 1x, 2x, 3x, BPFO, BPFI, BSF, FTF, 2xBPFI
- `mask_bands`, `mask_max_db` — how many bands breached the learned spectral mask
- `drift`, `drift_feat` — CUSUM level and which feature is drifting

About 26 bytes. It means the cloud knows which orders moved before it decodes a
single spectrum, and the band heatmap in the app is queryable across the fleet
without touching the codec. The axis is orders rather than Hz precisely so a
harmonic comb stays in the same band when the machine changes speed.

---

## 6. Batching: honest result

| batch frames | sessions/day | Uploads | Sessions | Frames/session | Energy | vs naive |
|---|---|---|---|---|---|---|
| 1, priority flush | 4 | 33 | 33 | 1.0 | 733 J | 93.2% |
| 4, priority flush | 2 | 33 | 32 | 1.0 | 725 J | 93.2% |
| 4, no priority flush | 2 | 33 | 30 | 1.1 | 707 J | 93.4% |
| 8, no priority flush | 1 | 33 | 30 | 1.1 | 707 J | 93.4% |
| 12, no priority flush | 1 | 33 | 30 | 1.1 | 707 J | 93.4% |

**Batching barely helps under these settings, and the reason is structural.** The
gate is already so selective that uploads arrive at roughly 2.5 per week, which is
slower than any sensible batch age limit. Frames never accumulate. Raising
`batch_max_frames` past 4 changes nothing.

Batching becomes worth its complexity in exactly two situations: a shorter wake
interval (hourly rather than 6-hourly), or an asset noisy enough to trigger
frequently. It is implemented and controllable, but on a well-behaved asset the
gate has already done the work batching was meant to do. I would not enable
aggressive batching by default.

---

## 7. Cloud control surface

All measured on the gradual-fault scenario.

| Push | Uploads | Change | Heartbeat | Sessions | Sent | Rate-limited | Detection |
|---|---|---|---|---|---|---|---|
| none (default) | 33 | 22 | 10 | 32 | 23.9 KB | 0 | 0.5 / 0.25 / 1.25 d |
| `cmd_reset_baseline` @ d40 | 35 | 20 | 14 | 33 | 25.1 KB | 0 | 0.5 / 0.5 / 0.25 d |
| `cmd_reset_discovery` @ d40 | 44 | 29 | 14 | 41 | 33.4 KB | 0 | 0.5 / 0.5 / 4.5 d |
| budget 14 → 6 | 27 | 22 | 4 | 27 | 21.7 KB | **43** | unchanged |
| budget 6, reserve 4 for change | 31 | 22 | 8 | 31 | 22.8 KB | **10** | unchanged |
| heartbeat 3 d → 7 d | 27 | 22 | 4 | 27 | 21.7 KB | 0 | unchanged |
| `cmd_force_anchor` @ d60 | 34 | 22 | 10 | 33 | 24.7 KB | 0 | unchanged |
| idle gating off | 55 | 44 | 10 | 54 | 41.6 KB | 0 | **never** |

Two things worth reading closely.

**The change reserve does what it is for.** Cutting the budget from 14 to 6
produced 43 rate-limit refusals; adding a 4-upload reserve for change events cut
that to 10 while keeping every change upload. Without the reserve, routine
heartbeat traffic competes with real events for the same budget. This is the
control I would make non-optional.

**`cmd_reset_discovery` is expensive.** It costs 11 extra uploads and delays the
third fault-stage detection from 1.25 to 4.5 days, because the device reverts to
`unknown` and stops suppressing idle frames until the histogram refills. Use it
after a genuine duty-cycle change, not as a routine remedy.

### The full shadow

| Group | Keys |
|---|---|
| Duty cycle | `wake_interval_s`, `heartbeat_s` |
| Budget | `upload_max_per_window`, `rate_window_s`, `upload_reserve_change`, `conn_max_per_day`, `conn_priority_extra` |
| Batching | `batch_max_frames`, `batch_max_bytes`, `batch_max_age_s`, `flush_on_change` |
| Sensitivity | `s_hi`, `s_lo`, `persist_m`, `persist_n`, `refractory_s`, `change_burst_max`, `backoff_mult`, `min_vel_delta`, `min_acc_delta_db` |
| Drift | `cusum_k`, `cusum_h`, `slow_freeze_n` |
| State discovery | `idle_gating_enable`, `discovery_min_n`, `discovery_min_sep`, `discovery_min_gap_db`, `state_hyst_db`, `state_persist_n` |
| Diagnostics | `bearing[8]` (pushable after a bearing change) |
| Safety | `fallback_after_s`, `debug_frames`, `version` / `applied_version` |
| Commands | `cmd_reset_baseline`, `cmd_reset_discovery`, `cmd_force_upload`, `cmd_force_anchor` |

### What else this needs, beyond what was asked

Four additions I made because the control surface is incomplete without them:

- **`fallback_after_s`.** If the device has not heard from the cloud in 30 days,
  it reverts to shipped defaults. A bad config push that suppresses all uploads
  is otherwise unrecoverable: the device stops talking, so you can never reach it
  to fix it. This is the single most important safety property in the file.
- **`upload_reserve_change`.** Quantified above.
- **`conn_priority_extra`.** An unbounded priority-flush path is how a noisy
  asset flattens its own battery. Change events may open extra sessions, but a
  bounded number.
- **`version` / `applied_version`.** Two-way acknowledgement. Without it you
  cannot tell a device that rejected a config from one that never received it.

Still missing, and worth adding before a real deployment: signed config with a
CRC, staged rollout by device cohort, an explicit time-sync path (the sliding
rate window assumes a monotonic clock), and per-device NVM persistence of the
baseline and histogram across power cycles. Nothing here survives a reset today.

---

## 8. Footprint and timing

| | |
|---|---|
| Static buffers (BSS) | 85.8 KB, dominated by the 4096-point FFT workspace |
| `ep_ctx_t` | 15.8 KB, of which the upload queue is 13.3 KB (12 slots) |
| `eg_ctx_t` | 2.1 KB |
| Discovery state | 188 B |
| Config shadow | 128 B |
| Code | ~15.7 KB text |

**This does not fit a small MCU as configured.** The 85.8 KB is almost entirely
the FFT and envelope workspace. Dropping `EG_NFFT` to 2048 roughly halves it, and
the queue is sized by `EP_Q_SLOTS`; at 4 slots it falls to 4.4 KB. A realistic
target build is around 45 to 50 KB RAM. That still needs a Cortex-M4F with
64 KB+, not an M0. I have not done the fixed-point port, so the claim that this
runs in the acquisition's own power budget is untested.

---

## 9. Caveats

1. Idle acquisitions are synthesized (sensor noise floor plus faint coupling at
   -30 dB). CWRU contains no stopped-machine data. The idle/running separation
   here is ~23 dB, which is generous. Real assets with light-load idle states
   will separate less cleanly and the gap test may correctly refuse to fire.
2. Running acquisitions reuse source records at random offsets. Each window is
   real measured data; the 90-day ordering is constructed.
3. Detection latencies for the CWRU seeded faults are optimistic because those
   faults are 12 dB steps. The blended incipient scenario is the more honest test
   and it gives 27 days.
4. Energy figures come from an assumed model, not measurement.
5. No NVM persistence, no clock-skew handling, no fixed-point port, no on-target
   profiling.

---

## 10. Files

- `edge_gate_app.html` — interactive evaluation. Seven scenarios, aligned strip
  chart, band-change heatmap, learned histogram, ablation panel.
- `firmware/` — `edge_gate.{h,c}` (DSP, features, codec, gating),
  `edge_policy.{h,c}` (state discovery, config shadow, queue), `sim_main.c`,
  `Makefile`.
- `code/` — `s8_scenarios.py` (scenario builder), `s9_eval.py` (evaluation).
