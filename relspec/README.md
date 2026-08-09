# relspec

Spectral condition monitoring end to end: synthetic generation, streaming
extraction, an edge transmission gate in C, a dual-domain residual codec, a
kinematic ontology in SQL, and pattern-energy tracking with a measured loss
ledger.

Built and measured over one session. Every number below was produced by the
code in this repo, on real CWRU bearing data or on synthetic data generated
here. Failures and negative results are recorded alongside the working parts,
because several of them changed the design.

---

## v2: the compression-first deepening

v2 keeps v1's frame format and distortion contract and rebuilds everything
around it: an entropy-coded codec, three real rigs instead of one, a fleet
three times the size, and signal processing measured in bytes. Full numbers
and the studies behind them: `docs/RESULTS_V2.md`; data provenance:
`docs/DATA_SOURCES.md`; run it all: `make v2-all`.

| | codec v1 | codec v2 | vs raw |
|---|---|---|---|
| CWRU, full 12k DE matrix (60 files, 332 frames) | 522 B/frame | **251 B/frame** | **65x** |
| MFPT (48.8 / 97.7 kHz) | 940 | **391** | **262x** |
| SEU gearbox (5.12 kHz - v1's fixed band cannot run here) | 435 / 417 | **182 / 176** | **45-47x** |
| Synthetic fleet, 48 assets x 90 d, 8,640 acqs | 375 | **177** (354 B/day/sensor) | **93x** |

Same dead-zone distortion contract as v1, enforced by a 7,200-frame fuzz
test; verified patterns carry 99.2% of their energy through the codec on
CWRU (0.3 dB median, measured on the bins each pattern owns - combs are
cloud-verified against real in-place peaks before they count); named lines
at 0.0-0.5 dB median; the unmodified v1 gate on v2 extraction detects 24/24
seeded fleet faults with 0/22 false alarms.

New in v2, each measured (`src/relspec/`): `codec2` adaptive-Rice entropy
stage, zero-byte peak tables, chain references; `dsp2` kurtogram band
selection, Viterbi speed tracking (-56% bytes on a hostile asset),
angle-domain Welch order spectra, spectral kurtosis, prewhitening (measured,
rejected); `pipeline2` rig-agnostic extraction with dual-evidence speed;
`datasets` CWRU/MFPT/SEU loaders; `synth2` nine machine classes, rotor-bar
and VFD physics, ramps, sensor pathologies - plus a v1 wander-model bug the
deeper estimator surfaced (RESULTS_V2 section 6).

---

## Quick start

```bash
cd src && python3 run_demo.py            # 16-asset synthetic fleet -> DuckDB
cd src && python3 run_patterns.py        # + pattern extraction and loss ledger
cd src && CWRU=/path/to/mat python3 cwru_eval.py     # full CWRU accuracy
cd src && CWRU=/path/to/mat python3 phase_anchor.py  # anchor / phase study
cd firmware && make                      # builds the edge simulator
```

CWRU `.mat` files: files 97-100, 105-108, 169-172, 209-212 from the Case
Western Reserve bearing data centre, 12 kHz drive end.

---

## Layout

```
db/                  schema. 001-003 portable (Postgres + DuckDB),
                     004 pattern store, 010 TimescaleDB hypertables and CAGGs
firmware/            edge_gate.{c,h}   DSP, features, codec, gating
                     edge_policy.{c,h} idle discovery, config shadow, batching
                     sim_main.c        harness that drives the real firmware
src/relspec/         synth      physics-based waveform generator
                     pipeline   extraction, codec, decoder, gate
                     patterns   combs, modulation, peaks, haystacks, energy
                     patternrun pattern extraction over stored spectra
                     fleet      fleet-scale run, pyramid, order trends
                     db         schema loader, fleet builder
src/                 run_demo, run_patterns, cwru_eval, phase_anchor, make_report
reports/             interactive HTML, self-contained
experiments/         the incremental studies, kept for provenance
docs/                written reports
```

---

## Headline results

| | |
|---|---|
| Compression, dual domain, synthetic | **54.7x**, 77 B/day/sensor |
| Compression, real CWRU, no entropy stage | 26.8x, 611 B/frame |
| Named line amplitude error (1x, 2x, 3x, BPFO, BPFI, BSF, FTF) | **0.0-0.5 dB** median |
| Pattern energy recovered through the codec | **91.5%** energy-weighted |
| Energy balance closure | 5.6e-17 |
| Radio sessions, 90 d | 4 (from 180 naive) |
| Battery estimate | **8.9 y** (from 1.1 y naive) |
| Idle/running classification | 157/157 and 156/156, zero errors |
| Detection latency, step fault | 0.25-0.5 d |
| False alarms across three healthy scenarios | 0 |
| Ontology view vs materialised | **130x**, and view cost is flat in fleet size |
| Streaming vs batch under 3% speed wander | 7x sharper, 3.4x stronger, 30x less memory |

---

## Reports

- `reports/method_and_accuracy.html` - the method dissected with the
  mathematics, full CWRU accuracy, and phase-free shaft-speed recovery
- `reports/dual_domain_report.html` - interactive waterfalls, measured against
  rebuilt against difference, both rails, scrubable
- `reports/anchor_and_phase.html` - anchor-mode normalisation, the
  order-locked vs frequency-locked test, and where phase does and does not help
- `reports/edge_gate_app.html` - 90-day transmission decisions across seven
  scenarios, aligned strip chart, band-change heatmap, idle-gating ablation

---

## Findings that changed the design

**Speed must be estimated on the envelope, not the acceleration spectrum.**
CWRU's low-frequency acceleration is dominated by mount lines near 40 and
47.6 Hz; harmonic search there locks onto a fixed feature regardless of true
speed, giving 1.7-4.7% error while reporting good confidence. On the envelope:
0.062% median.

**Acquisition length dominates compression, not codec tuning.** 1 s gives 46x,
4 s with a 16384-point Welch gives 241x. A 5.2x swing that no codec parameter
came close to. Short acquisitions have few Welch averages, so bins scatter
several dB on a steady machine and the residual coder spends its budget
encoding chi-squared noise.

**Idle frames must never enter the baseline.** With idle gating off, state
accuracy halves, 38-44 false change uploads fire, and the real fault is never
detected at all: the baseline widens to span both operating states and the
fault fits inside it.

**Session setup costs what 500 KB costs, so connections are the only thing
that matters.** Payload bytes cost 0.76 J across 90 days; radio setup 80-660 J.
Bursting on episode onset only, at a 21-day cadence, beat the previous default
on every axis at once - 4 sessions instead of 29, half the energy, and *faster*
delivery.

**Per-bin error is the wrong loss metric.** Peak-preserving MAX binning shows
19 dB p95 per-bin error while every diagnostic feature stays within 0.2 dB.
Loss is measured as pattern energy, with exclusive bin ownership so the balance
closes.

**Modulation belongs in the envelope domain.** Sideband search on the
acceleration rail failed across six design iterations; at order 8 a 0.25-order
spacing is 6 bins and the pair windows cannot be both wide enough to catch a
line and narrow enough not to overlap. Demodulation moves the same content to
baseband where it is an isolated peak. One detector, applied twice.

**The anchor need not be shaft speed, but it must move with it.** A mode with a
fixed ratio to the shaft works as a normalisation reference. On CWRU the ~12x
family is order-locked at ratio 12.053 with cv 0.0054, tighter than 1x. But the
lines near 2x and 3x are *frequency-locked* structural resonances: normalising
by one would actively misalign every spectrum.

**Phase is real but second-order.** The two-frame phase-vocoder estimator
resolves sub-bin for one extra FFT and no extra bytes, and phase is consumed on
the device rather than transmitted. It moves the anchor frequency by 0.15 bins,
which is 170x smaller than the anchor's own 22% order-stability variation. Sub-
bin precision is not the bottleneck; anchor identity is.

---

## Known limits

- Diagnosis is explicitly out of scope. An earlier attempt matched combs to the
  nearest kinematic line and produced findings on 14 of 16 assets including
  every healthy one; it detected presence, not abnormality. Removed.
- ~~Pattern observations average two per pattern... trending is starved.~~
  Closed: every acquisition now carries a 30-byte banded fingerprint
  (16 acc + 8 env dB bands + gate score/drift + vel RMS), stored and served
  by the health endpoint, so trends have per-acquisition resolution while
  full spectra stay gated.
- Envelope residual share is 35% mean. Much of an envelope is genuinely not
  patterned, but this is unconfirmed rather than explained.
- Firmware is 85.8 KB BSS as configured, mostly FFT workspace. NFFT 2048 and a
  4-slot queue reach roughly 45 KB. No fixed-point port, no NVM persistence, no
  on-target profiling.
- The streaming extractor (heterodyne bank, TSA, order-dependent coherent
  integration) is specified and its core validated standalone, but
  `pipeline.extract` still uses block Welch.
- Anchor selection is validated over only 3.9% of speed span, all CWRU offers.

---

## The full system (sensor → cloud → screen)

Everything above ships as one deployable project:

```
edge/firmware/   C99 sensor firmware: portable RSP/1 core (host-tested),
                 ESP32-S3 port skeleton, POSIX device simulator
edge/gateway/    gateway container: beacon discovery, stream reassembly,
                 SQLite store-and-forward, edge features, OPC UA server,
                 cloud uplink
edge/tests/      end-to-end chain test: C sim → gateway → real Postgres
platform/        cloud: FastAPI + TimescaleDB ingest/query API, passphrase
                 workspaces, MCP server, webapp
viz/             explorer + CMMS dashboard prototype (artifacts)
docker-compose.yml   one command for the whole stack: db, api, webapp,
                 MCP, gateway, two simulated sensor nodes
```

Hardware reference design, protocol spec, and gateway architecture:
`docs/EDGE_FIRMWARE.md`. Chain verification: `make -C edge/firmware test`
and `pytest edge/tests/`.
