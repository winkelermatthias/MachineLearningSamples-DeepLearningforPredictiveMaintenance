# relspec edge: firmware, gateway, and the full chain

This document covers the hardware selection, the firmware architecture,
the RSP/1 wire protocol, the gateway container, and how the whole chain
packages together with the cloud platform. Everything described is
implemented in `edge/` and exercised end-to-end by
`edge/tests/test_chain.py`: a compiled C device simulator streams
waveforms into the gateway, which buffers, feature-extracts, exposes
OPC UA readouts, and uplinks into the real platform (FastAPI + embedded
Postgres) — the test then reads the results back from both the cloud
API and an OPC UA client.

```
[sensor node firmware (C)] --RSP/1 UDP beacon + TCP stream--> [gateway container]
                                                              | SQLite spool (store & forward)
                                                              | edge features (RMS/peak/crest)
                                                              | OPC UA server  opc.tcp:4840
                                                              | HTTP status/control :8080
                                                              v
                                    [cloud: FastAPI + TimescaleDB]  --> webapp / MCP
```

---

## 1. Chipset selection

### 1.1 Sensing front-end

The physics sets the requirement: bearing envelope analysis needs the
resonant band (2–6 kHz typically, up to 10 kHz on small bearings), so
the sensor chain must be honestly flat well beyond 2 kHz — this is
where consumer MEMS parts fail silently.

| part | type | BW (±3 dB) | noise | range | interface | verdict |
|---|---|---|---|---|---|---|
| **TDK IIM-42352** | digital MEMS | **4 kHz** | 70 µg/√Hz | ±16 g | SPI 24 MHz, 2 KB FIFO | **wireless node default** — industrial vibration-specific, int16 native, FIFO tolerates radio jitter |
| **ADXL1002** | analog MEMS | **11 kHz** | 25 µg/√Hz | ±50 g | analog → ext. ADC | **wired node default** — full envelope band, needs ADS127L01 (24-bit, 512 kSPS) behind it |
| ADXL355 | digital MEMS | 1.5 kHz | 22.5 µg/√Hz | ±8 g | SPI | low-speed machinery (<600 RPM) only |
| ADcmXL3021 | module | 10 kHz | 26 µg/√Hz | ±50 g | SPI | excellent but costly, constrained supply |
| IEPE piezo + ADS1278 | piezo | 15 kHz+ | best | ±50 g | analog 4 mA loop | retrofit path: reuse installed accelerometers via an IEPE input board on the gateway |

Two reference nodes, one firmware:

- **RS-W (wireless)**: IIM-42352 @ 8 kHz ODR → 4 kHz usable band.
  Covers 1×–10× orders and the 2–4 kHz envelope band on 1–3 kRPM
  machines. Battery or 24 V loop powered.
- **RS-H (wired, high-bandwidth)**: ADXL1002 → ADS127L01 @ 25.6 kSPS →
  11 kHz band. Powered, Ethernet/WiFi. For gearboxes and high-speed
  spindles.

### 1.2 MCU

| MCU | cores/clock | RAM | radio | why / why not |
|---|---|---|---|---|
| **ESP32-S3-WROOM-1** | 2× LX7 @ 240 MHz | 512 KB + 8 MB PSRAM | WiFi b/g/n | **chosen** — PSRAM holds a 32 s sample ring (backpressure absorber), native WiFi streaming, mature IDF, $3 module |
| STM32U575 + nRF5340 | CM33 160 MHz + net core | 786 KB | BLE 5.3 | the coin-cell route-collector variant; BLE burst upload instead of streaming; phase 2 |
| RP2350 + external radio | 2× CM33 | 520 KB | none | cheap but radio integration is on you |
| i.MX RT1062 | CM7 600 MHz | 1 MB | none | overkill unless the node itself runs the codec |

Decision: **ESP32-S3** for both reference nodes. The 8 MB octal PSRAM
is the deciding feature — a big ring buffer converts WiFi latency
spikes and AP roaming into *buffered seconds instead of lost samples*,
which is exactly the failure mode that ruins spectral data. Power
figures (measured class numbers): 240 mA peak TX, ~40 mA sampling
with radio idle, 8 µA deep sleep → a duty-cycled RS-W node (2 s
acquisition every 10 min) averages ≈1.1 mA: two years on 2×AA
lithium, or indefinite on a 24 V loop tap.

Edge compute split, stated honestly: the node does **acquisition and
transport only**. The gateway computes instant features (RMS/peak/
crest) for OPC UA; the cloud runs the codec, gate, and pattern
pipeline, because those are stateful per-sensor and versioned there.
Running the codec on-node is possible (it is integer-friendly by
design) but buys nothing while the gateway exists.

## 2. Firmware architecture

Portable core (C99, zero allocation, no OS deps) + thin ports:

```
firmware/core/     rs_proto.[ch]  RSP/1 serialization + CRC-32  (host-tested)
                   rs_ring.[ch]   SPSC sample ring: DMA/ISR producer,
                                  task consumer, all-or-none push,
                                  drop counting (gap = restart window)
firmware/ports/
    posix/         device simulator: the same core driven by a synthetic
                   shaft+vane+BPFO-impact signal; full RSP/1 including
                   control handling. This binary is what CI streams.
    esp32s3/       ESP-IDF project skeleton: sample_task (SPI DMA →
                   ring, core 1), stream_task (ring → TCP), beacon_task;
                   Kconfig for device id / gateway / WiFi.
firmware/tests/    host unit tests (make test): CRC vectors, ring wrap +
                   overflow discipline, frame/beacon/ctrl roundtrips,
                   corruption rejection.
```

Design rules the code enforces:
- **All-or-none ring pushes** — a DMA block never lands half-in; a
  full ring counts drops instead of corrupting a window. The consumer
  aborts the acquisition on a gap: contiguity is sacred, latency is not.
- **The wire is bytes, not structs** — hand serialization with explicit
  little-endian put/get; identical C and Python implementations, both
  pinned by tests.
- **Devices hold no cloud credentials.** A stolen node yields a WiFi
  password, not plant data: only the gateway knows the workspace
  passphrase.

## 3. RSP/1 wire protocol

CRC-32 (IEEE) on every PDU. Little-endian throughout.

**Beacon** — UDP :47700, every 5 s, 40 bytes:
`magic 'RSB1' | ver | flags(streaming,fault) | fw_ver | dev_id[12] |
fs_hz | channels | battery_mv | uptime_s | drop_total | crc`

**Stream** — TCP :47701, device → gateway. Frame header 40 bytes:
`magic 'RSD1' | type | n | seq | dev_id[12] | t0_us | fs_hz |
scale_g(f32) | payload int16[n] | crc`

- `HELLO` opens a session; `DATA` carries ≤1024 samples with the
  first-sample timestamp; `ACQ_END` closes a window
  (payload: total_samples, dropped). The gateway discards incomplete
  windows loudly.
- int16 + per-frame scale is deliberately the cloud API's native
  encoding (`encoding:'int16', scale`) — the waveform is bit-identical
  from ADC to database.

**Control** — same TCP socket, gateway → device, 16 bytes:
`magic 'RSC1' | cmd | arg | crc` with START / STOP / SET_FS /
SET_ACQ_MS / IDENT (blink for a technician) / REBOOT.

## 4. Gateway container

`edge/gateway` (Python asyncio, ~600 lines):

| module | job |
|---|---|
| `ingestd` | beacon listener + TCP stream server; window reassembly; CRC and completeness enforcement |
| `spool` | SQLite store-and-forward: an acquisition is durable on gateway disk before any network step; zstd int16 blobs |
| `uplink` | drains the spool oldest-first (the cloud rejects out-of-order history); passphrase → bearer token; exponential backoff on outage — **cloud down = spool grows, nothing lost**; 4xx verdicts recorded, never block the queue |
| `edgeproc` | instant RMS / peak / crest per acquisition for local readout |
| `opcua_server` | asyncua server, namespace `urn:relspec:gateway` |
| `main` | orchestrator + tiny JSON HTTP: `GET /status`, `POST /ctrl {dev,cmd,arg}` |

**OPC UA namespace** (`opc.tcp://gateway:4840/relspec/`):

```
Objects/Gateway/    SpoolPending  SpoolSent  CloudOnline  DeviceCount
Objects/Devices/<dev_id>/
    SensorPath  Streaming  Connected  BatteryMv  DropTotal
    RmsG  PeakG  Crest  LastAcqTime  AcqCount
```

Values update on every completed acquisition and beacon; any SCADA /
historian subscribes like to any other OPC UA server. Device→sensor
mapping lives in `devices.json` (`{"RS-000A01": "Plant/Asset/Comp/S1"}`);
unmapped devices are ingested and buffered but marked `skipped` at
uplink until mapped — nothing silently vanishes.

## 5. Packaging: the big project

`relspec/docker-compose.yml` runs the entire system:

| service | image | role |
|---|---|---|
| db | timescale/timescaledb pg16 | hypertables, compression, CAGGs |
| api | platform/server | ingest pipeline, policies, cached reads |
| frontend | nginx + built webapp | dashboard (passphrase connect) |
| mcp | platform/mcpserver | Claude-driven migration/ingestion |
| **gateway** | edge/gateway | RSP/1 in, OPC UA + status out, uplink |
| **devicesim-a/b** | edge/firmware | two simulated sensor nodes streaming 2 s @ 8192 Hz every 2–3 min |

Bring-up: `cp platform/.env.example .env` (set `POSTGRES_PASSWORD`,
`RELSPEC_PEPPER`, `RS_PASSPHRASE`), create the workspace + hierarchy
once (webapp or MCP `ensure_hierarchy`), map devices, `docker compose
up --build`. Real hardware replaces the sim services by pointing nodes
at the gateway host.

## 6. Verification

- `make test` (firmware): CRC vectors, ring overflow/wrap discipline,
  frame/beacon/ctrl roundtrip + corruption rejection — compiled
  `-Wall -Wextra -Werror -Wconversion`.
- `pytest edge/tests/test_chain.py` (4 tests, ~9 s): full chain
  streaming to a real Postgres cloud; OPC UA client readback of device
  features and gateway gauges; control-channel START of an idle device
  flowing through to the cloud; spool ordering/marking discipline.
- Observed end-to-end in the chain test: the cloud's significance
  policy answering the gateway per acquisition
  (`first_waveform` → raw stored; `weekly_budget_exhausted` → codec
  frame only) — the storage economics work from the first byte a
  device sends.

## 7. Roadmap (not yet built)

- OTA: signed firmware images announced via control channel, A/B
  partitions on the S3.
- BLE route-collector node (STM32U5 + nRF5340) for hard-to-reach
  points; the gateway grows a BLE bridge.
- Gateway-local gate preview (reuse `relspec.pipeline2`) to trigger
  event-budget snapshots *before* the cloud round-trip.
- IEPE analog input board: 4-channel ADS1278 hat turning the gateway
  itself into a wired high-bandwidth node.
