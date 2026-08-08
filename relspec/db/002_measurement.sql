-- ===========================================================================
-- 002_measurement.sql : what was observed, what was transmitted, what it means
-- ===========================================================================

-- --------------------------------------------------------- 1. acquisitions
-- One row per wake cycle, including the ones that transmitted nothing. The
-- non-transmitting rows are the record of the gate working and are needed to
-- audit duty cycle and battery; dropping them makes the fleet look busier
-- than it is.
CREATE TABLE acquisition (
    acq_id        BIGINT PRIMARY KEY,
    channel_id    BIGINT NOT NULL REFERENCES channel(channel_id),
    ts            TIMESTAMP NOT NULL,
    duration_s    DOUBLE PRECISION NOT NULL,
    fs_hz         DOUBLE PRECISION NOT NULL,
    op_state      TEXT NOT NULL,            -- idle | running | transient | unknown
    decision      TEXT NOT NULL,            -- up_change | up_heartbeat | no_quiet | ...
    change_kind   SMALLINT,                 -- 1 step, 2 drift
    fr_hz         DOUBLE PRECISION,         -- estimated shaft rate
    fr_conf       DOUBLE PRECISION,
    align_tier    SMALLINT,                 -- 0 A .. 3 D
    coherence_s   DOUBLE PRECISION,         -- T_coh actually achieved
    payload_bytes INTEGER NOT NULL DEFAULT 0
);

-- Scalar features, one row per acquisition. Wide rather than key/value: these
-- seven are fixed by the firmware and are queried together every time.
CREATE TABLE feature (
    acq_id        BIGINT PRIMARY KEY REFERENCES acquisition(acq_id),
    acc_rms       DOUBLE PRECISION,
    vel_rms       DOUBLE PRECISION,         -- mm/s, ISO 20816 band
    acc_kurt      DOUBLE PRECISION,
    acc_crest     DOUBLE PRECISION,
    env_rms       DOUBLE PRECISION,
    env_kurt      DOUBLE PRECISION,
    env_crest     DOUBLE PRECISION,
    score         DOUBLE PRECISION,         -- gate anomaly score
    drift         DOUBLE PRECISION,         -- CUSUM statistic
    mask_bands    SMALLINT
);

-- --------------------------------------------------------------- 2. codec
CREATE TABLE codec_anchor (
    anchor_id     BIGINT PRIMARY KEY,
    channel_id    BIGINT NOT NULL REFERENCES channel(channel_id),
    acq_id        BIGINT NOT NULL REFERENCES acquisition(acq_id),
    valid_from    TIMESTAMP NOT NULL,
    valid_to      TIMESTAMP,
    n_bins        SMALLINT NOT NULL,
    ord_fine      DOUBLE PRECISION NOT NULL,
    ord_max       DOUBLE PRECISION NOT NULL,
    q_db          DOUBLE PRECISION NOT NULL,
    db_offset     DOUBLE PRECISION NOT NULL,
    bins          BLOB NOT NULL,            -- uint8[n_bins]
    peak_idx      BLOB,                     -- uint16[]
    mad           BLOB                      -- uint8[n_bins]
);

CREATE TABLE codec_frame (
    acq_id        BIGINT PRIMARY KEY REFERENCES acquisition(acq_id),
    anchor_id     BIGINT REFERENCES codec_anchor(anchor_id),
    frame_kind    TEXT NOT NULL,            -- anchor | residual
    payload       BLOB NOT NULL,
    n_bytes       INTEGER NOT NULL,
    decoded_ok    BOOLEAN,
    recon_med_db  DOUBLE PRECISION,         -- decode error against device truth
    recon_p99_db  DOUBLE PRECISION
);

-- Decoded, order-binned spectrum. This is what every view below reads.
CREATE TABLE spectrum (
    acq_id        BIGINT NOT NULL REFERENCES acquisition(acq_id),
    rail          TEXT NOT NULL,            -- acc | env
    bins          BLOB NOT NULL,            -- uint8 log-dB, order axis
    PRIMARY KEY (acq_id, rail)
);

-- ------------------------------------------------- 3. transmission / energy
CREATE TABLE transmission (
    tx_id         BIGINT PRIMARY KEY,
    sensor_id     BIGINT NOT NULL REFERENCES sensor(sensor_id),
    ts            TIMESTAMP NOT NULL,
    n_frames      SMALLINT NOT NULL,
    n_bytes       INTEGER NOT NULL,
    trigger       TEXT NOT NULL,            -- schedule | onset_burst | queue_full
    setup_j       DOUBLE PRECISION NOT NULL,
    payload_j     DOUBLE PRECISION NOT NULL
);

CREATE TABLE transmission_frame (
    tx_id         BIGINT NOT NULL REFERENCES transmission(tx_id),
    acq_id        BIGINT NOT NULL REFERENCES acquisition(acq_id),
    PRIMARY KEY (tx_id, acq_id)
);

-- ------------------------------------------------------- 4. change evidence
-- Signed change per order band, straight from the upload header. 26 bytes on
-- the wire, and it means the fleet can be queried for "what moved" without
-- decoding a single spectrum.
CREATE TABLE band_delta (
    acq_id        BIGINT NOT NULL REFERENCES acquisition(acq_id),
    band          SMALLINT NOT NULL,
    ord_lo        DOUBLE PRECISION NOT NULL,
    ord_hi        DOUBLE PRECISION NOT NULL,
    delta_db      DOUBLE PRECISION NOT NULL,
    over_mask     BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (acq_id, band)
);

-- --------------------------------------------------------- 5. diagnostics
-- Detected combs. `forcing_id` is filled by the matcher when the fundamental
-- lines up with a kinematic frequency; NULL means a family we cannot name yet,
-- which is itself worth surfacing.
CREATE TABLE harmonic_family (
    family_id     BIGINT PRIMARY KEY,
    acq_id        BIGINT NOT NULL REFERENCES acquisition(acq_id),
    rail          TEXT NOT NULL,
    f0_order      DOUBLE PRECISION NOT NULL,
    score         DOUBLE PRECISION NOT NULL,
    n_harmonics   SMALLINT NOT NULL,
    forcing_id    BIGINT,
    match_err_pct DOUBLE PRECISION
);

CREATE TABLE sideband_family (
    sb_id         BIGINT PRIMARY KEY,
    acq_id        BIGINT NOT NULL REFERENCES acquisition(acq_id),
    rail          TEXT NOT NULL,
    carrier_order DOUBLE PRECISION NOT NULL,
    spacing_order DOUBLE PRECISION NOT NULL,
    scr_db        DOUBLE PRECISION NOT NULL,
    n_pairs       SMALLINT,
    carrier_forcing_id BIGINT,
    spacing_forcing_id BIGINT           -- what modulates it: usually 1x or FTF
);

-- A finding is a diagnosis with provenance. It points at the evidence rather
-- than restating it, so a finding can be audited and withdrawn.
CREATE TABLE finding (
    finding_id    BIGINT PRIMARY KEY,
    asset_id      BIGINT NOT NULL REFERENCES asset(asset_id),
    component_id  BIGINT REFERENCES component(component_id),
    bearing_id    BIGINT REFERENCES bearing_instance(bearing_id),
    opened_at     TIMESTAMP NOT NULL,
    closed_at     TIMESTAMP,
    fault_mode    TEXT NOT NULL,        -- outer_race | inner_race | imbalance | ...
    severity      SMALLINT NOT NULL CHECK (severity BETWEEN 1 AND 4),
    confidence    DOUBLE PRECISION NOT NULL,
    rationale     TEXT,
    first_acq_id  BIGINT REFERENCES acquisition(acq_id)
);

CREATE TABLE finding_evidence (
    finding_id    BIGINT NOT NULL REFERENCES finding(finding_id),
    acq_id        BIGINT NOT NULL REFERENCES acquisition(acq_id),
    kind          TEXT NOT NULL,        -- harmonic | sideband | band_delta | feature
    ref_id        BIGINT,
    weight        DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    PRIMARY KEY (finding_id, acq_id, kind, ref_id)
);

-- ------------------------------------------------------------- 6. synthetic
-- Ground truth for synthetic runs. Kept in the same database on purpose:
-- an evaluation that lives outside the schema drifts away from it.
CREATE TABLE synth_truth (
    acq_id        BIGINT PRIMARY KEY REFERENCES acquisition(acq_id),
    running       BOOLEAN NOT NULL,
    fault_mode    TEXT,
    severity      DOUBLE PRECISION NOT NULL DEFAULT 0.0,   -- 0 healthy .. 1 severe
    true_fr_hz    DOUBLE PRECISION NOT NULL,
    load_frac     DOUBLE PRECISION
);
