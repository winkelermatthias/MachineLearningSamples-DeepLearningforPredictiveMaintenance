# relspec platform — plan

Dockerised service that turns raw time waveforms into the relspec
representation (features, dual-rail spectra, codec payloads, verified
patterns, gate events), stores it in TimescaleDB under a simple
passphrase-scoped workspace, serves the explorer webapp fast from cached
APIs, and exposes an MCP server so Claude can migrate a customer's existing
waveform database into it conversationally.

The pipeline itself is done and measured (RESULTS_V2): this plan is about
wrapping it in storage, transport, auth, and access. Design principle
throughout: the server owns every policy decision (significance, storage
cadence, codec state, track identity) — clients, including Claude via MCP,
only ever say "here is a waveform from this sensor at this time".

---

## 1. Topology

```
docker compose
├─ db         timescale/timescaledb:2.x-pg16     volume: pgdata
├─ api        FastAPI + uvicorn (python 3.12, numpy/scipy + relspec pkg)
├─ frontend   nginx: static explorer app + /api reverse proxy w/ micro-cache
└─ mcp        relspec MCP server (streamable-http), thin client of api
```

- No separate worker initially: ingest processing (extract2 + codec2 +
  verified patterns + gate) measures ~150-300 ms per 2 s waveform on one
  core — inline in the API request with a bounded process pool. A queue
  (arq/Redis) is a phase-5 option if sustained ingest exceeds ~10/s.
- No Redis for caching. Three cheaper layers (§6) built on TimescaleDB
  continuous aggregates + HTTP semantics do the job for this access
  pattern.
- TLS terminates at the frontend nginx (or an optional caddy in front for
  auto-certificates); api and db are not published on the host network.

## 2. Data model

Builds on `db/001-004` (kept: the kinematic ontology and forcing-frequency
views are exactly what diagnosis-adjacent queries need) with three changes:
every row gains a `workspace_id`, surrogate ids become ULIDs, and the
API-facing hierarchy is trimmed to the four levels the user asked for —
**plant → asset → component → sensor** — mapped onto the existing tables
(plant = site with an implicit org per workspace; area optional, default
"main").

New / reshaped tables (020_workspace.sql, 021_measurement.sql):

```sql
workspace   (workspace_id TEXT PK,           -- derived from passphrase, §5
             name TEXT, auth_hash TEXT,      -- argon2id verifier
             data_version BIGINT,            -- bumped on every write → ETags
             created_at, params JSONB)

plant       (workspace_id, plant_id ULID PK, name, tz, UNIQUE(ws, name))
asset       (workspace_id, asset_id PK, plant_id FK, tag, machine_type,
             criticality, UNIQUE(ws, plant_id, tag))
component   (workspace_id, component_id PK, asset_id FK, name, kind,
             shaft_ratio, UNIQUE(ws, asset_id, name))
             -- kinematics (bearing_instance, gear_mesh) attach here, optional
sensor      (workspace_id, sensor_id PK, component_id FK, code, position,
             units, fs_nominal, fr_nominal, UNIQUE(ws, component_id, code))

acquisition (HYPERTABLE on ts; workspace_id, acq_id, sensor_id, ts,
             fs, n_samples, client_ref,      -- idempotency key
             fr_est, tier, conf,
             vel_rms, acc_rms, env_rms, acc_kurt, acc_crest, env_kurt,
             gate_score, gate_drift, gate_decision, gate_kind,
             frame_kind, payload_bytes, waveform_stored BOOL, sig_reason)

spectrum_frame (acq_id, rail, payload BYTEA)       -- the codec2 bitstream
pattern        (acq_id, rail, track_id, kind, key, f0, spacing,
                energy, energy_dec, share, ver_harm, ver_claimed, ver_frac)
pattern_track  (workspace_id, sensor_id, rail, track_id, kind, key,
                first_seen, last_seen)             -- persisted TrackRegistry
codec_state    (sensor_id, rail, anchor BYTEA, prev BYTEA, mad BYTEA,
                abytes INT)                        -- encoder continuity
gate_state     (sensor_id, state JSONB)            -- Gate fields, resumable
waveform       (HYPERTABLE; acq_id, encoding, scale, data BYTEA)  -- zstd
event          (HYPERTABLE; workspace_id, sensor_id, ts, type, payload JSONB)
                -- gate changes, repairs, workspace lifecycle
```

Codec and gate state live in the DB so ingest is stateless across API
replicas and restarts — the sensor's GOP chain and baseline survive
anything except an explicit reset event.

### Continuous aggregates & materialized views (022_caggs.sql)

- `trend_daily`  (CAGG, 1 day buckets per sensor): avg/max/last of
  vel_rms, acc_rms, env_rms, max gate_score, count, sum payload_bytes.
  Refresh policy: every 30 min, lag 1 h. This is the strip-chart API.
- `trend_hourly` (CAGG, for zoomed views; retention 90 d).
- `pattern_energy_weekly` (CAGG over pattern joined via acq ts): per
  (sensor, rail, track) weekly sum energy / energy_dec — the long-horizon
  pattern trend without touching row data.
- `sensor_latest` (small MATERIALIZED VIEW, `REFRESH CONCURRENTLY` every
  60 s by pg_cron or api scheduler): one row per sensor — last ts, last
  levels, last gate state, open change events, bytes over 30 d. Feeds the
  fleet overview screen in one indexed read.

### Compression & retention (023_policies.sql)

- `acquisition`, `spectrum_frame`: Timescale native compression after 7 d
  (segment by sensor_id) — the payloads are already entropy-coded, but the
  scalar columns compress ~10x.
- `waveform`: compressed after 7 d, retention 2 y (configurable per
  workspace in `params`).
- `event`: kept forever (tiny).

## 3. Ingest pipeline & the significance policy

`POST /v1/waveforms` (or MCP submit) runs, per waveform:

1. Validate: fs 1-100 kHz, 0.25-60 s, units in {g, m/s2, mm/s}, converted
   to g; NFKC-trim ids; reject NaN/clipped>20%; idempotent on
   (sensor, client_ref) — replays return the original result.
2. `extract2` with the sensor's cached kurtogram band key and fr_nominal;
   dual-evidence speed; features.
3. Load codec + gate state; encode both rails (v2, refs='best'); gate
   decision; persist new state.
4. Verified pattern extraction on both rails (as in the explorer export);
   track ids resolved against `pattern_track`.
5. **Significance decision** — server-side, never the client's call:

   store the FULL waveform iff
   `first waveform ever for this sensor`
   OR (`gate_decision == up_change` AND `event budget not exhausted`)
   OR (`≥7 days since last stored waveform` — the weekly baseline);
   subject to the hard cap `waveform_budget_per_week` (default 1 baseline
   + 2 event-triggered; set `1 + 0` for the strict max-1/week reading).
   Every submission ALWAYS stores features + codec frames + patterns +
   gate — that is the whole point: the ~300 B representation is never
   rationed, only the ~35 KB raw waveform is.
6. Bump `workspace.data_version`; append `event` if gate fired.
7. Respond with the verdict: `{acq_id, waveform_stored, sig_reason,
   features, payload_bytes, patterns: n, gate: {...}}` so Claude (or any
   client) can report honestly what happened to each submission.

Batch endpoint accepts NDJSON streams (one waveform per line) with
per-line results; historical backfills MUST be submitted oldest-first per
sensor so baseline, codec chain, and weekly budget accrue in real order —
the MCP prompt enforces this.

Storage arithmetic (from measured numbers): features+frames+patterns
≈ 0.5 KB/acquisition; at 1/day that is ~180 KB/sensor/year. Weekly
waveform: 2 s @ 12 kHz int16+zstd ≈ 30-40 KB → ~1.8 MB/sensor/year.
10,000 sensors ≈ 20 GB/year before Timescale compression. Nothing here
needs sharding for a long time.

## 4. API surface (v1)

Auth on every route except workspace create/health (§5). All responses
JSON; arrays of u8 spectra as base64, exactly the explorer's format.

```
POST /v1/workspaces                 {passphrase, name?}    → {workspace_id, token}
GET  /v1/workspaces/me                                     → summary, data_version

POST /v1/hierarchy:ensure           {plants:[{name, assets:[{tag, components:
                                     [{name, kind, sensors:[{code, fs, units,
                                     fr_nominal, position}]}]}]}]}
                                    idempotent upsert by natural keys → ids
GET  /v1/hierarchy                                          → the tree

POST /v1/waveforms                  {sensor_path|sensor_id, ts, fs, units,
                                     encoding: float32|int16, scale?, data_b64,
                                     client_ref?, meta?}    → ingest verdict
POST /v1/waveforms:batch            NDJSON stream           → NDJSON verdicts

GET  /v1/fleet/overview                                     → sensor_latest rows
GET  /v1/sensors/{id}/trend?from&to&bucket=day|hour         → CAGG rows
GET  /v1/sensors/{id}/frames?from&to&limit&cursor           → acq metadata list
GET  /v1/acquisitions/{id}/spectra?rail=acc|env             → decoded u8 + patterns
                                     + ownership map (server decodes payload)
GET  /v1/sensors/{id}/waterfall?from&to&rail&max_cols=400   → assembled matrix
GET  /v1/sensors/{id}/tracks                                → pattern_track + weekly energy
GET  /v1/events?from&to&type                                → gate/lifecycle events
GET  /v1/acquisitions/{id}/waveform                         → raw samples when stored
GET  /health, GET /metrics (prometheus)
```

Limits: request body ≤ 32 MB; ≤ 20 ingest/s/workspace (429 + Retry-After);
waterfall ≤ 1,000 columns per call (cursor beyond that).

## 5. Passphrase workspaces — no accounts, one key

The passphrase IS the workspace: whoever holds it can write and read.

- Creation: `POST /v1/workspaces {passphrase}`. Server normalises (NFKC,
  trim, collapse whitespace), enforces ≥ 4 words or ≥ 20 chars, then:
  - `workspace_id = base58( HKDF-SHA256(passphrase, info="relspec-ws-id") )[:16]`
    — deterministic, so the same phrase always finds the same workspace,
    with no lookup table and nothing to enumerate;
  - `auth_hash = argon2id(passphrase, random salt, m=64MiB, t=3)` stored
    for verification (the HKDF id alone must never authenticate, or the id
    leaking would grant access).
- Requests authenticate either way:
  - `Authorization: Passphrase <b64(phrase)>` — verified via argon2
    (rate-limited: 5 failures/min/IP, constant-time compare), or
  - `Authorization: Bearer <token>` — HMAC token minted at create/verify,
    24 h expiry, so hot paths skip argon2. MCP caches the bearer.
- Server-side pepper (env `RELSPEC_PEPPER`) mixed into both derivations so
  a stolen database alone cannot be brute-forced offline at full speed.
- Honest limits, stated in the UI and MCP prompt: lose the phrase, lose
  the data (no recovery); anyone with the phrase has full read/write (no
  roles); this is scoping + authentication, not end-to-end encryption —
  at-rest encryption stays a disk/volume concern. Phrases are never
  stored, logged, or echoed back.

## 6. Caching — how the frontend stays fast

1. **The database does the heavy lifting**: trends and weekly pattern
   energy are CAGG reads (indexed, pre-aggregated); fleet overview is one
   MV read. No query the frontend issues ever scans raw hypertable rows
   except frame lists over bounded ranges.
2. **API layer**: per-workspace in-process TTL-LRU for decoded spectra and
   assembled waterfalls, keyed `(endpoint, params, data_version)` — a
   workspace's `data_version` bump on ingest invalidates implicitly.
   Acquisition-scoped resources are immutable → `Cache-Control:
   max-age=31536000, immutable`. Range queries → `ETag:
   "<data_version>"` + 30 s `max-age`, so the browser and nginx revalidate
   cheaply.
3. **nginx micro-cache** in the frontend container: 10 s proxy cache for
   GETs keyed by full URL + Authorization hash — absorbs dashboard fan-out
   (five strips ask for the same frames) and refresh storms.

## 7. Frontend

The explorer as built, ported from embedded-JSON to the API:

- sequence selector → plant/asset/component/sensor picker from
  `/hierarchy` + `/fleet/overview` (status chips: last seen, gate state,
  velocity).
- waterfall/strips fetch `/frames` + `/waterfall` windowed (default last
  90 frames, infinite scroll left); frame inspector fetches
  `/acquisitions/{id}/spectra` on selection; trends from `/trend`.
- passphrase prompt on first load → bearer in sessionStorage; everything
  else unchanged (both themes, 3-D, events, help — the help tab's live
  examples run off fetched frames).
- Full-waveform frames get a download affordance (`/waveform`).

## 8. MCP server — Claude as the ingestion analyst

Python `mcp` SDK, streamable-http in the container (stdio entry point for
local use). Thin: every tool is an API call plus good errors.

Tools:
```
relspec_create_workspace(passphrase, name?)         → workspace summary
relspec_connect(passphrase)                         → verifies, caches bearer
relspec_ensure_hierarchy(tree)                      → ids, per-node created/existing
relspec_get_hierarchy()
relspec_submit_waveform(sensor_path, ts, fs, units,
                        data_b64 | values[], client_ref?, meta?)
                                                    → full ingest verdict
relspec_submit_batch(ndjson_chunk)                  → per-line verdicts
relspec_get_status(sensor_path?)                    → fleet/sensor summary
relspec_get_trend(sensor_path, from, to)            → daily trend rows
```

Shipped MCP prompt (`ingest-migration`), the text Claude receives:

> You are migrating a customer's vibration time-waveform database into a
> relspec workspace. Work in this order and confirm each stage with the
> user before executing it.
> 1. INSPECT their database read-only. Find the tables holding waveforms,
>    sample rates, timestamps, units, and whatever identifies the
>    measurement point. Show the user what you found and how many
>    waveforms/sensors/date-range it covers.
> 2. PROPOSE the hierarchy mapping: plant → asset → component → sensor,
>    built from their naming. Never invent structure they don't have —
>    a flat plant with assets is fine. Show the tree; adjust until they
>    approve; then call relspec_ensure_hierarchy once.
> 3. MIGRATE oldest-first per sensor, in batches of ≤ 200. Order matters:
>    baselines, codec chains and weekly storage budgets accrue in time
>    order, and history submitted backwards is history wasted. Use a
>    stable client_ref (their primary key) so re-runs are idempotent.
> 4. Do NOT decide which waveforms deserve full storage — submit
>    everything in range; the server keeps every waveform's features,
>    spectra and patterns, and stores raw waveforms on its own
>    significance policy (weekly baseline + change events). Report the
>    verdicts honestly, including how many raw waveforms were kept.
> 5. VERIFY: after each sensor, call relspec_get_trend and compare row
>    counts and level ranges against their source. Summarise per-sensor:
>    submitted, ingested, duplicates skipped, waveforms stored, events
>    raised. Never report success you have not read back.
> The passphrase is the only key to this data: have the user choose it
> and store it themselves; never write it into files or logs.

## 9. Compose file sketch

```yaml
services:
  db:
    image: timescale/timescaledb:2.15-pg16
    environment: {POSTGRES_DB: relspec, POSTGRES_USER: relspec,
                  POSTGRES_PASSWORD_FILE: /run/secrets/pg}
    volumes: [pgdata:/var/lib/postgresql/data,
              ./db/init:/docker-entrypoint-initdb.d:ro]
    healthcheck: pg_isready
  api:
    build: ./server          # python:3.12-slim + numpy scipy zstandard
    environment: {DATABASE_URL: ..., RELSPEC_PEPPER_FILE: /run/secrets/pepper}
    depends_on: {db: {condition: service_healthy}}
    deploy: {resources: {limits: {memory: 2g}}}
  mcp:
    build: ./mcp             # talks to api over the compose network
    environment: {RELSPEC_API: "http://api:8000"}
    ports: ["8765:8765"]     # streamable-http MCP endpoint
  frontend:
    build: ./web             # nginx: static app + /api proxy + micro-cache
    ports: ["443:443", "80:80"]
volumes: {pgdata: {}}
```

Backups: nightly `pg_dump -Fc` sidecar to a mounted volume; `waveform`
excluded from the fast tier of restore docs (biggest, least critical).

## 10. Build phases

1. **MVP ingest + store** — compose, schema 020/021, workspace auth,
   `/waveforms` inline pipeline with codec/gate state persistence,
   `/hierarchy:ensure`, `trend_daily` CAGG. Exit test: replay the
   3-year synthetic sequence through the API oldest-first; verify frame
   bytes and gate events match the explorer export exactly (they run the
   same code, so any drift is a bug).
2. **Read APIs + frontend port** — spectra/waterfall/frames/overview
   endpoints with the caching layers; explorer running against them.
3. **MCP server + prompt** — tools above; end-to-end demo: Claude
   migrates a SQLite dump of CWRU windows into a fresh workspace.
4. **Policies & ops** — compression/retention jobs, `sensor_latest`
   refresh, rate limits, metrics, backup sidecar, load test (target: 50
   concurrent dashboard users on cached endpoints < 50 ms p95; sustained
   5 ingests/s single API replica).
5. **Later, explicitly out of MVP** — queue-based ingest for burst
   backfills, per-workspace at-rest encryption, role separation
   (read-only phrases), Timescale multi-node.

## 11. Risks & decided trade-offs

- **Passphrase-only auth is deliberately weak-by-design**: it trades
  account machinery for friction-free "Claude can just write". Mitigated
  by argon2+pepper, rate limiting, and honest messaging; not suitable for
  regulated data as-is.
- **Deterministic workspace ids** leak existence (same phrase → same id).
  Accepted: ids alone grant nothing.
- **Inline ingest** couples API latency to scipy; bounded pool + 32 MB cap
  keeps it sane, queue is the escape hatch.
- **Server-side significance** means a client cannot force-keep a
  waveform; a `meta.pin: true` request is honoured only within the weekly
  event budget — documented, since it will surprise someone.
- **Codec state in DB** makes replay/backfill order-sensitive per sensor;
  the batch API rejects out-of-order timestamps per sensor within a
  stream rather than silently degrading compression.
