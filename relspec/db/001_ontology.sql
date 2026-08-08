-- ===========================================================================
-- 001_ontology.sql : what exists, and what its frequencies mean
--
-- Portable core (Postgres 15+ / DuckDB). TimescaleDB specifics live in
-- 010_timescale.sql; the DuckDB pyramid lives in 011_duckdb.sql.
--
-- Design principle: nothing downstream should ever do order arithmetic. The
-- kinematic tables here expand into `forcing_frequency`, and every diagnostic
-- query joins to that instead of hard-coding 3.5848 somewhere in a dashboard.
-- ===========================================================================

-- ---------------------------------------------------------------- 1. estate
CREATE TABLE org (
    org_id        BIGINT PRIMARY KEY,
    name          TEXT NOT NULL
);

CREATE TABLE site (
    site_id       BIGINT PRIMARY KEY,
    org_id        BIGINT NOT NULL REFERENCES org(org_id),
    name          TEXT NOT NULL,
    tz            TEXT NOT NULL DEFAULT 'UTC',
    lat           DOUBLE PRECISION,
    lon           DOUBLE PRECISION
);

CREATE TABLE area (
    area_id       BIGINT PRIMARY KEY,
    site_id       BIGINT NOT NULL REFERENCES site(site_id),
    name          TEXT NOT NULL,
    process_stage TEXT                      -- e.g. 'crushing', 'flotation'
);

-- An asset is a machine train. Criticality drives attention weighting, not
-- alarm thresholds: a low-criticality machine still gets a correct diagnosis,
-- it just competes differently for an analyst's time.
CREATE TABLE asset (
    asset_id      BIGINT PRIMARY KEY,
    area_id       BIGINT NOT NULL REFERENCES area(area_id),
    tag           TEXT NOT NULL,            -- customer's own identifier
    name          TEXT NOT NULL,
    asset_class   TEXT NOT NULL,            -- pump | motor | fan | gearbox | compressor
    criticality   SMALLINT NOT NULL DEFAULT 3 CHECK (criticality BETWEEN 1 AND 5),
    duty          TEXT NOT NULL DEFAULT 'continuous', -- continuous | intermittent | batch
    drive         TEXT NOT NULL DEFAULT 'dol',        -- dol | vfd | turbine
    rated_kw      DOUBLE PRECISION,
    commissioned  DATE,
    UNIQUE (area_id, tag)
);

-- A component is a physically replaceable unit inside the train. Replacing one
-- resets its baselines and its bearing life, so it needs its own identity.
CREATE TABLE component (
    component_id  BIGINT PRIMARY KEY,
    asset_id      BIGINT NOT NULL REFERENCES asset(asset_id),
    name          TEXT NOT NULL,
    kind          TEXT NOT NULL,            -- motor | pump_end | gearbox | coupling | fan
    position      SMALLINT NOT NULL,        -- order along the train, drive to driven
    installed     DATE,
    removed       DATE
);

-- ------------------------------------------------------------ 2. kinematics
-- Every rotating element referenced to one nominal shaft, so orders compose.
CREATE TABLE shaft (
    shaft_id      BIGINT PRIMARY KEY,
    component_id  BIGINT NOT NULL REFERENCES component(component_id),
    name          TEXT NOT NULL,
    is_reference  BOOLEAN NOT NULL DEFAULT FALSE,  -- the 1x the order axis uses
    ratio_to_ref  DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    nominal_rpm   DOUBLE PRECISION
);

-- Manufacturer catalogue. Multipliers are per shaft revolution.
CREATE TABLE bearing_catalog (
    bearing_type  TEXT PRIMARY KEY,
    manufacturer  TEXT,
    n_rollers     SMALLINT,
    bpfo          DOUBLE PRECISION NOT NULL,
    bpfi          DOUBLE PRECISION NOT NULL,
    bsf           DOUBLE PRECISION NOT NULL,
    ftf           DOUBLE PRECISION NOT NULL
);

CREATE TABLE bearing_instance (
    bearing_id    BIGINT PRIMARY KEY,
    shaft_id      BIGINT NOT NULL REFERENCES shaft(shaft_id),
    bearing_type  TEXT NOT NULL REFERENCES bearing_catalog(bearing_type),
    position      TEXT NOT NULL,            -- DE | NDE | inboard | outboard
    installed     DATE,
    removed       DATE
);

CREATE TABLE gear_mesh (
    mesh_id       BIGINT PRIMARY KEY,
    component_id  BIGINT NOT NULL REFERENCES component(component_id),
    drive_shaft   BIGINT NOT NULL REFERENCES shaft(shaft_id),
    driven_shaft  BIGINT NOT NULL REFERENCES shaft(shaft_id),
    drive_teeth   SMALLINT NOT NULL,
    driven_teeth  SMALLINT NOT NULL
);

-- Vane / blade / lobe passing. One row covers pumps, fans and screw compressors.
CREATE TABLE passing_element (
    pass_id       BIGINT PRIMARY KEY,
    component_id  BIGINT NOT NULL REFERENCES component(component_id),
    shaft_id      BIGINT NOT NULL REFERENCES shaft(shaft_id),
    kind          TEXT NOT NULL,            -- vane | blade | lobe | slot
    n_elements    SMALLINT NOT NULL
);

-- ------------------------------------------------------- 3. instrumentation
CREATE TABLE sensor (
    sensor_id     BIGINT PRIMARY KEY,
    component_id  BIGINT NOT NULL REFERENCES component(component_id),
    serial        TEXT NOT NULL UNIQUE,
    model         TEXT NOT NULL DEFAULT 'MachineDoctor',
    fw_version    TEXT,
    mount         TEXT,                     -- stud | magnet | adhesive
    installed     DATE,
    removed       DATE,
    battery_j     DOUBLE PRECISION DEFAULT 18500.0
);

CREATE TABLE channel (
    channel_id    BIGINT PRIMARY KEY,
    sensor_id     BIGINT NOT NULL REFERENCES sensor(sensor_id),
    axis          TEXT NOT NULL,            -- X | Y | Z | radial | axial
    quantity      TEXT NOT NULL,            -- acceleration | velocity | temperature
    rail          TEXT NOT NULL,            -- acc | env
    fs_hz         DOUBLE PRECISION NOT NULL,
    UNIQUE (sensor_id, axis, rail)
);

-- Which shaft a channel's order axis is locked to. Without this the order
-- values in the spectrum tables are meaningless across a multi-shaft train.
CREATE TABLE channel_reference (
    channel_id    BIGINT PRIMARY KEY REFERENCES channel(channel_id),
    shaft_id      BIGINT NOT NULL REFERENCES shaft(shaft_id)
);

-- ------------------------------------------- 4. lifecycle and configuration
CREATE TABLE device_config (
    config_id     BIGINT PRIMARY KEY,
    sensor_id     BIGINT NOT NULL REFERENCES sensor(sensor_id),
    version       INTEGER NOT NULL,
    applied_at    TIMESTAMP,
    payload       TEXT NOT NULL,            -- JSON of the shadow
    UNIQUE (sensor_id, version)
);

-- Anything that invalidates a baseline: repair, replacement, re-mount, reset.
CREATE TABLE lifecycle_event (
    event_id      BIGINT PRIMARY KEY,
    asset_id      BIGINT NOT NULL REFERENCES asset(asset_id),
    component_id  BIGINT REFERENCES component(component_id),
    sensor_id     BIGINT REFERENCES sensor(sensor_id),
    occurred_at   TIMESTAMP NOT NULL,
    kind          TEXT NOT NULL,            -- repair | replace | remount | baseline_reset
    note          TEXT,
    resets_baseline BOOLEAN NOT NULL DEFAULT TRUE
);
