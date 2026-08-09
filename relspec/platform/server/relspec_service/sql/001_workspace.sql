-- Workspaces: the passphrase IS the tenancy. workspace_id is derived
-- deterministically from the passphrase (HKDF, see auth.py) so the same
-- phrase always finds the same workspace; auth_hash is an scrypt verifier
-- (the derived id must never authenticate by itself).
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename    TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE workspace (
    workspace_id  TEXT PRIMARY KEY,
    name          TEXT NOT NULL DEFAULT '',
    auth_hash     TEXT NOT NULL,
    -- bumped on every write; the ETag / cache key for everything read
    data_version  BIGINT NOT NULL DEFAULT 1,
    params        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
