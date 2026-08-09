-- Health model + per-pattern z (DASHBOARD_DESIGN 2-3) and the per-
-- acquisition change fingerprint. Portable; no Timescale dependency.

ALTER TABLE acquisition ADD COLUMN health SMALLINT;      -- 0 ok 1 mon 2 alert 3 crit, NULL learning
ALTER TABLE acquisition ADD COLUMN health_drivers JSONB;
ALTER TABLE acquisition ADD COLUMN fingerprint BYTEA;    -- 30 B banded-energy + score, every frame

ALTER TABLE pattern ADD COLUMN z REAL;                   -- vs own baseline, NULL while learning

-- Streaming robust baseline per (sensor, rail, track): EW estimates of
-- median/absolute deviation over quiet frames only.
CREATE TABLE pattern_baseline (
    sensor_id  TEXT NOT NULL,
    rail       TEXT NOT NULL,
    track_id   INT  NOT NULL,
    n          INT  NOT NULL DEFAULT 0,
    med        REAL,
    mad        REAL,
    window_start TIMESTAMPTZ,
    last_seen  TIMESTAMPTZ,
    miss_streak INT NOT NULL DEFAULT 0,
    PRIMARY KEY (sensor_id, rail, track_id)
);
