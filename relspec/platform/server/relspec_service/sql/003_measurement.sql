-- Measurement side. In Timescale mode (005) acquisition / waveform / event
-- become hypertables; the DDL here is plain Postgres and stays valid.
CREATE TABLE acquisition (
    workspace_id  TEXT NOT NULL,
    acq_id        TEXT NOT NULL,
    sensor_id     TEXT NOT NULL REFERENCES sensor(sensor_id),
    ts            TIMESTAMPTZ NOT NULL,
    client_ref    TEXT,
    fs            DOUBLE PRECISION NOT NULL,
    n_samples     INT NOT NULL,
    fr_est        DOUBLE PRECISION,
    tier          INT,
    conf          DOUBLE PRECISION,
    vel_rms       DOUBLE PRECISION,
    acc_rms       DOUBLE PRECISION,
    env_rms       DOUBLE PRECISION,
    acc_kurt      DOUBLE PRECISION,
    acc_crest     DOUBLE PRECISION,
    gate_score    DOUBLE PRECISION,
    gate_drift    DOUBLE PRECISION,
    gate_decision TEXT,
    gate_kind     INT,
    frame_kind    TEXT,                        -- A / R / P (worst of two rails)
    payload_bytes INT,
    waveform_stored BOOLEAN NOT NULL DEFAULT FALSE,
    sig_reason    TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (acq_id, ts)                   -- ts in PK: hypertable-ready
);
CREATE UNIQUE INDEX acq_dedupe ON acquisition(sensor_id, client_ref)
    WHERE client_ref IS NOT NULL;
CREATE INDEX acq_sensor_ts ON acquisition(sensor_id, ts DESC);
CREATE INDEX acq_ws_ts ON acquisition(workspace_id, ts DESC);

-- Both rails per acquisition. payload is the codec2 bitstream (what a device
-- would have transmitted; its length is the compression ledger). o / dec /
-- own are the original, decoded and pattern-ownership u8 vectors stored for
-- serving: ~3.5 KB/frame keeps every read a pure assembly with no chain
-- replay, and TOAST compresses them well.
CREATE TABLE spectrum_frame (
    acq_id     TEXT NOT NULL,
    rail       TEXT NOT NULL CHECK (rail IN ('acc','env')),
    kind       TEXT NOT NULL,                  -- anchor / residual / residual_p
    payload    BYTEA NOT NULL,
    o          BYTEA NOT NULL,
    dec        BYTEA NOT NULL,
    own        BYTEA NOT NULL,
    res_share  REAL,
    PRIMARY KEY (acq_id, rail)
);

CREATE TABLE pattern (
    acq_id     TEXT NOT NULL,
    rail       TEXT NOT NULL,
    idx        INT  NOT NULL,                  -- position in frame's ledger
    track_id   INT  NOT NULL,
    kind       TEXT NOT NULL,
    key        DOUBLE PRECISION NOT NULL,
    f0         DOUBLE PRECISION NOT NULL,
    spacing    DOUBLE PRECISION,
    energy     DOUBLE PRECISION NOT NULL,
    energy_dec DOUBLE PRECISION NOT NULL,
    share      DOUBLE PRECISION NOT NULL,
    ver_harm   INT, ver_claimed INT, ver_frac REAL,
    PRIMARY KEY (acq_id, rail, idx)
);

CREATE TABLE pattern_track (
    workspace_id TEXT NOT NULL,
    sensor_id  TEXT NOT NULL REFERENCES sensor(sensor_id),
    rail       TEXT NOT NULL,
    track_id   INT  NOT NULL,
    kind       TEXT NOT NULL,
    key        DOUBLE PRECISION NOT NULL,
    first_seen TIMESTAMPTZ NOT NULL,
    last_seen  TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (sensor_id, rail, track_id)
);

-- Encoder / gate continuity: ingest is stateless across API replicas and
-- restarts because the sensor's chain lives here, not in process memory.
CREATE TABLE codec_state (
    sensor_id  TEXT NOT NULL REFERENCES sensor(sensor_id),
    rail       TEXT NOT NULL,
    state      BYTEA NOT NULL,                 -- npz: anchor, prev, mad, abytes
    last_ts    TIMESTAMPTZ,
    PRIMARY KEY (sensor_id, rail)
);
CREATE TABLE gate_state (
    sensor_id  TEXT PRIMARY KEY REFERENCES sensor(sensor_id),
    state      BYTEA NOT NULL,                 -- json-serialised Gate fields
    n_seen     INT NOT NULL DEFAULT 0
);

CREATE TABLE waveform (
    acq_id     TEXT NOT NULL,
    ts         TIMESTAMPTZ NOT NULL,
    sensor_id  TEXT NOT NULL,
    encoding   TEXT NOT NULL DEFAULT 'int16-zstd',
    scale      DOUBLE PRECISION NOT NULL,      -- value = int16 * scale  [g]
    fs         DOUBLE PRECISION NOT NULL,
    data       BYTEA NOT NULL,
    PRIMARY KEY (acq_id, ts)
);
CREATE INDEX waveform_sensor_ts ON waveform(sensor_id, ts DESC);

CREATE TABLE event (
    workspace_id TEXT NOT NULL,
    event_id   TEXT NOT NULL,
    sensor_id  TEXT,
    ts         TIMESTAMPTZ NOT NULL,
    type       TEXT NOT NULL,                  -- gate_change / repair / ...
    payload    JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (event_id, ts)
);
CREATE INDEX event_ws_ts ON event(workspace_id, ts DESC);
