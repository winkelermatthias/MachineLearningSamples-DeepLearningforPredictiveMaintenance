-- Resumable bulk-ingest jobs. Items keep the full submitted body so a
-- job survives process restarts; per-item state is the checkpoint, and
-- client_ref idempotency in the ingest path makes resume exactly-once.

CREATE TABLE migration_job (
    job_id       TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    state        TEXT NOT NULL DEFAULT 'running',  -- running|paused|done|failed
    created      TIMESTAMPTZ NOT NULL DEFAULT now(),
    total        INT NOT NULL,
    done         INT NOT NULL DEFAULT 0,
    errors       INT NOT NULL DEFAULT 0
);

CREATE TABLE migration_item (
    job_id   TEXT NOT NULL,
    ordinal  INT  NOT NULL,
    sensor_ref TEXT NOT NULL,
    ts       TIMESTAMPTZ NOT NULL,
    body     JSONB NOT NULL,
    state    TEXT NOT NULL DEFAULT 'pending',      -- pending|done|error
    error    TEXT,
    PRIMARY KEY (job_id, ordinal)
);
CREATE INDEX migration_item_pending ON migration_item (job_id, state, ordinal);
