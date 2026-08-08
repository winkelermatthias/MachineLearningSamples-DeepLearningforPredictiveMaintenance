-- ===========================================================================
-- 003_views.sql : the ontology layer proper
--
-- `forcing_frequency` is the pivot of the whole design. It expands the
-- kinematic tables into every order that means something on a given channel,
-- with a label and a provenance. Everything downstream joins to it, so no
-- dashboard, model or query ever contains a hard-coded bearing multiplier.
-- ===========================================================================

-- --------------------------------------------- 1. kinematic expansion
CREATE VIEW forcing_frequency AS
-- shaft harmonics
SELECT
    ch.channel_id,
    a.asset_id,
    c.component_id,
    s.shaft_id,
    CAST(NULL AS BIGINT)                       AS bearing_id,
    'shaft_harmonic'                           AS family,
    CAST(h.n AS VARCHAR) || 'x'                AS label,
    s.ratio_to_ref * h.n                       AS order_value,
    h.n                                        AS harmonic,
    CASE WHEN h.n = 1 THEN 1.0 ELSE 0.6 END    AS prior,
    'imbalance,misalignment,looseness'         AS fault_modes
FROM channel ch
JOIN channel_reference cr ON cr.channel_id = ch.channel_id
JOIN sensor sn            ON sn.sensor_id  = ch.sensor_id
JOIN component c          ON c.component_id = sn.component_id
JOIN asset a              ON a.asset_id    = c.asset_id
JOIN shaft s              ON s.component_id = c.component_id
CROSS JOIN (SELECT 1 AS n UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL
            SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6) h

UNION ALL
-- bearing defect frequencies and their harmonics
SELECT
    ch.channel_id, a.asset_id, c.component_id, s.shaft_id, bi.bearing_id,
    'bearing',
    t.code || CASE WHEN h.n > 1 THEN ' x' || CAST(h.n AS VARCHAR) ELSE '' END,
    s.ratio_to_ref * t.mult * h.n,
    h.n,
    CASE WHEN h.n = 1 THEN 1.0 ELSE 0.5 END,
    t.modes
FROM channel ch
JOIN channel_reference cr ON cr.channel_id = ch.channel_id
JOIN sensor sn            ON sn.sensor_id  = ch.sensor_id
JOIN component c          ON c.component_id = sn.component_id
JOIN asset a              ON a.asset_id    = c.asset_id
JOIN shaft s              ON s.component_id = c.component_id
JOIN bearing_instance bi  ON bi.shaft_id   = s.shaft_id AND bi.removed IS NULL
JOIN bearing_catalog bc   ON bc.bearing_type = bi.bearing_type
CROSS JOIN (SELECT 1 AS n UNION ALL SELECT 2 UNION ALL SELECT 3) h
CROSS JOIN LATERAL (VALUES
    ('BPFO', bc.bpfo, 'outer_race'),
    ('BPFI', bc.bpfi, 'inner_race'),
    ('BSF',  bc.bsf,  'rolling_element'),
    ('FTF',  bc.ftf,  'cage')
) AS t(code, mult, modes)

UNION ALL
-- gear mesh and its harmonics
SELECT
    ch.channel_id, a.asset_id, c.component_id, gm.drive_shaft, NULL,
    'gear_mesh',
    'GMF' || CASE WHEN h.n > 1 THEN ' x' || CAST(h.n AS VARCHAR) ELSE '' END,
    sd.ratio_to_ref * gm.drive_teeth * h.n,
    h.n, CASE WHEN h.n = 1 THEN 1.0 ELSE 0.6 END,
    'tooth_wear,eccentricity,cracked_tooth'
FROM channel ch
JOIN channel_reference cr ON cr.channel_id = ch.channel_id
JOIN sensor sn            ON sn.sensor_id  = ch.sensor_id
JOIN component c          ON c.component_id = sn.component_id
JOIN asset a              ON a.asset_id    = c.asset_id
JOIN gear_mesh gm         ON gm.component_id = c.component_id
JOIN shaft sd             ON sd.shaft_id   = gm.drive_shaft
CROSS JOIN (SELECT 1 AS n UNION ALL SELECT 2 UNION ALL SELECT 3) h

UNION ALL
-- vane / blade / lobe passing
SELECT
    ch.channel_id, a.asset_id, c.component_id, pe.shaft_id, NULL,
    'passing',
    UPPER(SUBSTR(pe.kind,1,1)) || 'PF'
        || CASE WHEN h.n > 1 THEN ' x' || CAST(h.n AS VARCHAR) ELSE '' END,
    s.ratio_to_ref * pe.n_elements * h.n,
    h.n, CASE WHEN h.n = 1 THEN 1.0 ELSE 0.6 END,
    'cavitation,recirculation,clearance'
FROM channel ch
JOIN channel_reference cr ON cr.channel_id = ch.channel_id
JOIN sensor sn            ON sn.sensor_id  = ch.sensor_id
JOIN component c          ON c.component_id = sn.component_id
JOIN asset a              ON a.asset_id    = c.asset_id
JOIN passing_element pe   ON pe.component_id = c.component_id
JOIN shaft s              ON s.shaft_id    = pe.shaft_id
CROSS JOIN (SELECT 1 AS n UNION ALL SELECT 2 UNION ALL SELECT 3) h;

-- Stable surrogate key so findings and matches can reference a forcing line.
CREATE VIEW forcing_frequency_k AS
SELECT
    (channel_id * 100000)
      + CAST(ROUND(order_value * 100) AS BIGINT) AS forcing_id,
    *
FROM forcing_frequency;

-- --------------------------------------------- 2. what the sensor should see
-- The same expansion in Hz, given the speed actually measured. Useful for
-- reports that must be read in Hz, and for cross-checking the order lock.
CREATE VIEW forcing_hz AS
SELECT f.*, aq.acq_id, aq.ts, aq.fr_hz,
       f.order_value * aq.fr_hz AS freq_hz
FROM forcing_frequency_k f
JOIN acquisition aq ON aq.channel_id = f.channel_id
WHERE aq.op_state = 'running';

-- ------------------------------------------- 3. band membership of a forcing
-- Which of the 16 transmitted bands carries each forcing line, so band deltas
-- can be attributed to kinematics without decoding a spectrum.
CREATE VIEW forcing_band AS
SELECT f.forcing_id, f.channel_id, f.label, f.family, f.order_value,
       CAST(FLOOR(LEAST(f.order_value / 200.0, 0.999) * 16) AS SMALLINT) AS band
FROM forcing_frequency_k f;

-- ------------------------------------------------- 4. attributed band change
-- The headline diagnostic join: every transmitted band delta, labelled with
-- the kinematic lines that live in that band.
CREATE VIEW band_change_attributed AS
SELECT bd.acq_id, aq.ts, aq.channel_id, a.asset_id, a.tag,
       bd.band, bd.ord_lo, bd.ord_hi, bd.delta_db, bd.over_mask,
       fb.label, fb.family, fb.order_value
FROM band_delta bd
JOIN acquisition aq ON aq.acq_id = bd.acq_id
JOIN channel ch     ON ch.channel_id = aq.channel_id
JOIN sensor sn      ON sn.sensor_id = ch.sensor_id
JOIN component c    ON c.component_id = sn.component_id
JOIN asset a        ON a.asset_id = c.asset_id
LEFT JOIN forcing_band fb ON fb.channel_id = aq.channel_id AND fb.band = bd.band;

-- ---------------------------------------------- 5. named harmonic families
-- Detected combs resolved against kinematics. A family with no match is not an
-- error: unnamed combs are how you find a fault mode nobody modelled.
CREATE VIEW harmonic_named AS
SELECT hf.family_id, hf.acq_id, aq.ts, a.asset_id, a.tag, hf.rail,
       hf.f0_order, hf.score, hf.n_harmonics,
       ff.label, ff.family AS forcing_family, ff.fault_modes, ff.bearing_id,
       ABS(hf.f0_order - ff.order_value) / NULLIF(ff.order_value,0) * 100 AS err_pct
FROM harmonic_family hf
JOIN acquisition aq ON aq.acq_id = hf.acq_id
JOIN channel ch     ON ch.channel_id = aq.channel_id
JOIN sensor sn      ON sn.sensor_id = ch.sensor_id
JOIN component c    ON c.component_id = sn.component_id
JOIN asset a        ON a.asset_id = c.asset_id
LEFT JOIN forcing_frequency_k ff
       ON ff.channel_id = aq.channel_id
      AND ABS(hf.f0_order - ff.order_value) < 0.06 * GREATEST(ff.order_value, 1.0);

-- --------------------------------------------------------- 6. fleet health
CREATE VIEW asset_latest AS
SELECT a.asset_id, a.tag, a.name, a.asset_class, a.criticality,
       MAX(aq.ts) AS last_seen,
       SUM(CASE WHEN aq.op_state='running' THEN 1 ELSE 0 END) AS n_running,
       SUM(CASE WHEN aq.op_state='idle'    THEN 1 ELSE 0 END) AS n_idle,
       SUM(aq.payload_bytes) AS bytes_sent
FROM asset a
JOIN component c ON c.asset_id = a.asset_id
JOIN sensor sn   ON sn.component_id = c.component_id
JOIN channel ch  ON ch.sensor_id = sn.sensor_id
JOIN acquisition aq ON aq.channel_id = ch.channel_id
GROUP BY a.asset_id, a.tag, a.name, a.asset_class, a.criticality;

-- Attention index: recent evidence weighted by criticality. Deliberately a
-- view, not a stored score, so the weighting can be argued with.
CREATE VIEW asset_attention AS
SELECT a.asset_id, a.tag, a.criticality,
       MAX(f.score)  AS peak_score,
       MAX(f.drift)  AS peak_drift,
       COUNT(*) FILTER (WHERE aq.decision = 'up_change') AS n_change,
       MAX(f.vel_rms) AS peak_vel_rms,
       (COALESCE(MAX(f.score),0) * 0.5
        + COALESCE(MAX(f.drift),0) * 0.05
        + COUNT(*) FILTER (WHERE aq.decision='up_change') * 0.4)
         * (0.6 + 0.1 * a.criticality) AS attention
FROM asset a
JOIN component c ON c.asset_id = a.asset_id
JOIN sensor sn   ON sn.component_id = c.component_id
JOIN channel ch  ON ch.sensor_id = sn.sensor_id
JOIN acquisition aq ON aq.channel_id = ch.channel_id
JOIN feature f   ON f.acq_id = aq.acq_id
WHERE aq.op_state = 'running'
GROUP BY a.asset_id, a.tag, a.criticality;

-- ------------------------------------------------------- 7. battery / radio
-- Aggregate the two fact tables separately before joining. Joining acquisition
-- and transmission off the same sensor multiplies them together and inflates
-- both byte counts and radio energy by the acquisition count.
CREATE VIEW sensor_energy AS
WITH acq AS (
    SELECT sn.sensor_id, COUNT(*) AS n_acq,
           SUM(CASE WHEN aq.op_state='running' THEN 1 ELSE 0 END) AS n_running
    FROM sensor sn
    JOIN channel ch ON ch.sensor_id = sn.sensor_id
    JOIN acquisition aq ON aq.channel_id = ch.channel_id
    WHERE ch.rail = 'acc'
    GROUP BY sn.sensor_id
),
tx AS (
    SELECT sensor_id, COUNT(*) AS n_tx, SUM(n_bytes) AS bytes,
           SUM(setup_j) AS setup_j, SUM(payload_j) AS payload_j
    FROM transmission GROUP BY sensor_id
)
SELECT sn.sensor_id, sn.serial, sn.battery_j,
       COALESCE(acq.n_acq,0)      AS n_acq,
       COALESCE(acq.n_running,0)  AS n_running,
       COALESCE(tx.n_tx,0)        AS n_tx,
       COALESCE(tx.bytes,0)       AS bytes,
       COALESCE(acq.n_acq,0)*1.2  AS acq_j,
       COALESCE(tx.setup_j,0)     AS setup_j,
       COALESCE(tx.payload_j,0)   AS payload_j,
       COALESCE(acq.n_acq,0)*1.2 + COALESCE(tx.setup_j,0)
         + COALESCE(tx.payload_j,0) AS total_j
FROM sensor sn
LEFT JOIN acq ON acq.sensor_id = sn.sensor_id
LEFT JOIN tx  ON tx.sensor_id  = sn.sensor_id;
