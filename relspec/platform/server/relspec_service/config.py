"""Service configuration. Everything comes from the environment; every
value has a development default so `uvicorn relspec_service.app:app` runs
against a local database with zero ceremony."""
import os, sys, pathlib

DATABASE_URL = os.environ.get('DATABASE_URL', '')
# Server-side pepper: mixed into workspace-id derivation and verifier
# hashing so a stolen database cannot be brute-forced offline at full
# speed, and into bearer-token HMAC. MUST be set to a long random value in
# production; the default exists so tests and dev boot without ceremony.
PEPPER = os.environ.get('RELSPEC_PEPPER', 'dev-pepper-not-for-production')

# Path that makes `import relspec` work. In the Docker image the package is
# copied into site-packages and this is a no-op.
RELSPEC_SRC = os.environ.get(
    'RELSPEC_SRC',
    str(pathlib.Path(__file__).resolve().parents[3] / 'src'))
if RELSPEC_SRC and RELSPEC_SRC not in sys.path:
    sys.path.insert(0, RELSPEC_SRC)

TOKEN_TTL_S = int(os.environ.get('RELSPEC_TOKEN_TTL', 24*3600))
MAX_BODY_MB = int(os.environ.get('RELSPEC_MAX_BODY_MB', 32))

# Ingest bounds: generous, but bounded. 8 M samples ≈ 80 s at 200 kHz —
# anything larger is a mistake or an attack, refused before the DSP runs.
MAX_SAMPLES = int(os.environ.get('RELSPEC_MAX_SAMPLES', 8_000_000))
FS_MIN, FS_MAX = 1.0, 200_000.0
# Per-workspace ingest rate cap, waveforms per sliding minute (in-memory,
# per API replica). 429 + Retry-After beyond it.
INGEST_RATE_PER_MIN = int(os.environ.get('RELSPEC_INGEST_RATE_PER_MIN', 600))

# Significance policy defaults; overridable per workspace via params.
SIG_DEFAULTS = dict(
    baseline_days=7,          # scheduled full-waveform cadence
    event_budget_per_week=0,  # extra change-triggered stores ("max 1/week"
                              # by default; raise to keep event captures too)
)
