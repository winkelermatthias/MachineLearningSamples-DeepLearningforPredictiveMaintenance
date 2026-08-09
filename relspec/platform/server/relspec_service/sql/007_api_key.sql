-- Derived API keys: scoped bearer credentials minted by a full-access
-- principal. The token itself embeds key_id + scope under HMAC (auth.py);
-- the row exists only for listing and revocation — no secret is stored.
CREATE TABLE api_key (
    key_id       TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL REFERENCES workspace(workspace_id),
    scope        TEXT NOT NULL CHECK (scope IN ('read','ingest','full')),
    label        TEXT NOT NULL DEFAULT '',
    created      TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked      BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX api_key_ws ON api_key(workspace_id);
