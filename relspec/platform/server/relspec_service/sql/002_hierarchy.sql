-- The API-facing hierarchy: plant -> asset -> component -> sensor.
-- Natural keys are unique per parent so :ensure is idempotent; surrogate
-- ids are opaque text (uuid4 hex) minted by the server.
CREATE TABLE plant (
    workspace_id  TEXT NOT NULL REFERENCES workspace(workspace_id),
    plant_id      TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    tz            TEXT NOT NULL DEFAULT 'UTC',
    UNIQUE (workspace_id, name)
);

CREATE TABLE asset (
    workspace_id  TEXT NOT NULL REFERENCES workspace(workspace_id),
    asset_id      TEXT PRIMARY KEY,
    plant_id      TEXT NOT NULL REFERENCES plant(plant_id),
    tag           TEXT NOT NULL,
    machine_type  TEXT NOT NULL DEFAULT '',
    criticality   INT  NOT NULL DEFAULT 3,
    UNIQUE (workspace_id, plant_id, tag)
);

CREATE TABLE component (
    workspace_id  TEXT NOT NULL REFERENCES workspace(workspace_id),
    component_id  TEXT PRIMARY KEY,
    asset_id      TEXT NOT NULL REFERENCES asset(asset_id),
    name          TEXT NOT NULL,
    kind          TEXT NOT NULL DEFAULT '',       -- motor / gearbox / pump ...
    shaft_ratio   DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    UNIQUE (workspace_id, asset_id, name)
);

CREATE TABLE sensor (
    workspace_id  TEXT NOT NULL REFERENCES workspace(workspace_id),
    sensor_id     TEXT PRIMARY KEY,
    component_id  TEXT NOT NULL REFERENCES component(component_id),
    code          TEXT NOT NULL,                  -- customer's channel id
    position      TEXT NOT NULL DEFAULT '',
    units         TEXT NOT NULL DEFAULT 'g',
    fs_nominal    DOUBLE PRECISION,
    fr_nominal    DOUBLE PRECISION,               -- nominal shaft speed, Hz
    -- demodulation band chosen by the kurtogram on first ingest and pinned:
    -- band identity is an asset property, not an acquisition property
    band_lo_hz    DOUBLE PRECISION,
    band_hi_hz    DOUBLE PRECISION,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, component_id, code)
);
CREATE INDEX sensor_ws_idx ON sensor(workspace_id);
