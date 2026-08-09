# relspec platform

The relspec pipeline as a service: submit raw time waveforms, get back the
~300-byte spectral representation (features, dual-rail codec frames,
verified patterns, gate events) stored in TimescaleDB under a
passphrase-scoped workspace, served fast to the explorer webapp, with an
MCP server so Claude can migrate an existing waveform database
conversationally. Design rationale: `../docs/PLATFORM_PLAN.md`.

## Run it

Production shape (TimescaleDB + API + nginx frontend + MCP):

```bash
cd platform
cp .env.example .env          # set POSTGRES_PASSWORD and RELSPEC_PEPPER
docker compose up --build
# frontend http://localhost:8080   API :8080/v1   MCP http://localhost:8765
```

Zero-setup development (embedded Postgres via pgserver, no Docker):

```bash
pip install -r server/requirements.txt pgserver
python3 scripts/dev_server.py --port 8000
python3 scripts/seed_demo.py http://127.0.0.1:8000 "pick a long passphrase here"
# open http://127.0.0.1:8000/app/  and enter the same passphrase
```

## Tests

```bash
pip install pytest pgserver httpx
python3 -m pytest tests/          # real Postgres per test, ~15 s
```

The suite covers passphrase auth (derivation, verifier, tokens, throttle),
idempotent hierarchy, the ingest policy (first/weekly/event budget,
out-of-order rejection, duplicate replay, unit conversion), workspace
isolation, every read endpoint including ETag/304s, a 30-day fault
campaign through HTTP, and `test_parity.py` — the plan's exit test: the
service pipeline with its state round-tripping through serialisation
produces byte-identical codec payloads to the offline evaluation.

## Workspaces = passphrases

`POST /v1/workspaces {"passphrase": "..."}` derives the workspace id from
the phrase (HKDF + server pepper) and stores an scrypt verifier; the same
phrase always finds the same workspace. Authenticate with
`Authorization: Bearer <token>` (returned at connect) or
`Authorization: Passphrase <base64 phrase>`. The phrase is the only key:
losing it loses the data; holding it grants full read/write. Set a long
random `RELSPEC_PEPPER` and never change it after go-live.

## Significance policy

Every submission stores features + codec frames + verified patterns
(~0.5 KB). The raw waveform is stored only for: the sensor's first ever
waveform, the weekly baseline (>=7 days since last stored), or a gate
change event within the per-week event budget (default 0 extra = strict
max one per week; set workspace `params.event_budget_per_week` to raise).
Submit history oldest-first per sensor — out-of-order timestamps are
rejected because codec chains and baselines accrue in time order.

## MCP

`mcpserver/relspec_mcp/server.py` — tools: create/connect workspace,
ensure hierarchy, submit waveform/batch, status, trend; plus the
`ingest-migration` prompt that walks Claude through inspect → propose
mapping → confirm → migrate oldest-first → verify. Local stdio config:

```json
{"mcpServers": {"relspec": {
    "command": "python3",
    "args": ["-m", "relspec_mcp.server"],
    "cwd": "<repo>/relspec/platform/mcpserver",
    "env": {"RELSPEC_API": "http://127.0.0.1:8000"}}}}
```

## Layout

```
server/    FastAPI service; relspec_service/sql/ holds migrations
           (001-004 portable Postgres, 005 applied only when the
           timescaledb extension is present: hypertables, compression,
           retention, trend_daily as a continuous aggregate)
mcpserver/ MCP server (thin httpx client of the API)
web/       build.py renders the explorer template into an API-driven app;
           nginx.conf proxies /v1 with a 10 s per-workspace micro-cache
scripts/   dev_server.py (embedded PG + API), seed_demo.py
tests/     pytest suite against real Postgres (pgserver)
```

Known limits: Timescale-specific SQL (005) is exercised only in compose,
not in the pgserver suite; ingest runs inline (bounded by uvicorn workers)
— queue-based backfill is a planned phase; the passphrase model is
deliberately account-free (see the plan's trade-offs).
