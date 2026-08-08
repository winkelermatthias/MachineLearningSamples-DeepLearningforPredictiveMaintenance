-- ===========================================================================
-- 004_patterns.sql : pattern store and energy tracking
--
-- Replaces the diagnostic layer. A pattern is a harmonic comb or a modulation
-- family, identified by a stable key per channel so its energy forms a time
-- series. No fault modes, no findings.
--
-- Energy, not amplitude: a pattern is a set of bins and energy is the only
-- quantity that adds over a set without double counting. Every bin has at most
-- one owner, so per-acquisition the pattern energies plus the residual equal
-- the spectrum total exactly. That closure is what makes loss measurable.
-- ===========================================================================

CREATE TABLE pattern (
    pattern_id  BIGINT PRIMARY KEY,
    channel_id  BIGINT NOT NULL REFERENCES channel(channel_id),
    rail        TEXT   NOT NULL,
    kind        TEXT   NOT NULL,          -- comb | mod
    pkey        DOUBLE NOT NULL,          -- stable identity: f0 or carrier
    first_seen  TIMESTAMP NOT NULL,
    last_seen   TIMESTAMP NOT NULL,
    n_obs       INTEGER NOT NULL DEFAULT 0,
    UNIQUE (channel_id, rail, kind, pkey)
);

-- One row per pattern per acquisition. The fact table for all trending.
CREATE TABLE pattern_energy (
    acq_id      BIGINT NOT NULL REFERENCES acquisition(acq_id),
    pattern_id  BIGINT NOT NULL REFERENCES pattern(pattern_id),
    f0          DOUBLE NOT NULL,
    spacing     DOUBLE,
    snr         DOUBLE,
    n_pairs     SMALLINT,
    energy      DOUBLE NOT NULL,
    energy_db   DOUBLE NOT NULL,
    share       DOUBLE NOT NULL,          -- fraction of total spectral energy
    n_bins      SMALLINT NOT NULL,
    PRIMARY KEY (acq_id, pattern_id)
);

-- Energy balance per acquisition. residual_share is the fraction of energy no
-- pattern claimed, and is the honest measure of how much of the spectrum the
-- pattern model actually explains.
CREATE TABLE energy_balance (
    acq_id         BIGINT NOT NULL REFERENCES acquisition(acq_id),
    rail           TEXT NOT NULL,
    e_total        DOUBLE NOT NULL,
    e_patterned    DOUBLE NOT NULL,
    e_residual     DOUBLE NOT NULL,
    residual_share DOUBLE NOT NULL,
    n_combs        SMALLINT NOT NULL,
    n_mods         SMALLINT NOT NULL,
    PRIMARY KEY (acq_id, rail)   -- an acquisition has one balance PER RAIL
);

-- Loss ledger: the same extraction run on the pre-transmission spectrum and on
-- the decoded one, compared pattern by pattern. This is the project's headline
-- metric and it lives in the database so it can be queried per channel, per
-- pattern and over time rather than existing only in a report.
CREATE TABLE pattern_loss (
    acq_id        BIGINT NOT NULL REFERENCES acquisition(acq_id),
    pkey          DOUBLE NOT NULL,
    kind          TEXT NOT NULL,
    energy_orig   DOUBLE NOT NULL,
    energy_recon  DOUBLE,
    energy_err_db DOUBLE,
    f0_err_pct    DOUBLE,
    spacing_err_pct DOUBLE,
    recovered     BOOLEAN NOT NULL,
    PRIMARY KEY (acq_id, pkey, kind)
);

-- ---------------------------------------------------------------- views
-- Daily energy per pattern. The trending surface.
CREATE VIEW pattern_trend AS
SELECT p.pattern_id, p.channel_id, p.rail, p.kind, p.pkey,
       CAST(aq.ts AS DATE)         AS day,
       COUNT(*)                    AS n,
       MAX(pe.energy_db)           AS peak_db,
       MEDIAN(pe.energy_db)        AS med_db,
       MEDIAN(pe.share)            AS med_share,
       MEDIAN(pe.spacing)          AS med_spacing
FROM pattern_energy pe
JOIN pattern p     ON p.pattern_id = pe.pattern_id
JOIN acquisition aq ON aq.acq_id = pe.acq_id
GROUP BY 1,2,3,4,5,6;

-- Rise of a pattern's energy against its own early history. Structurally the
-- same as order_anomaly but keyed on discovered patterns rather than on a
-- kinematic table, so it needs no asset model at all.
CREATE VIEW pattern_energy_rise AS
WITH b AS (
    SELECT pattern_id, MIN(day) AS d0, MAX(day) AS d1 FROM pattern_trend GROUP BY 1
)
SELECT pt.pattern_id, pt.channel_id, pt.kind, pt.pkey,
       COUNT(*) AS n_days,
       MEDIAN(pt.med_db) FILTER (WHERE pt.day < b.d0 + INTERVAL 21 DAY) AS base_db,
       MEDIAN(pt.med_db) FILTER (WHERE pt.day > b.d1 - INTERVAL 14 DAY) AS recent_db,
       MAX(pt.peak_db) AS peak_db
FROM pattern_trend pt JOIN b ON b.pattern_id = pt.pattern_id
GROUP BY 1,2,3,4;

-- Fleet-level loss summary, straight from the ledger.
CREATE VIEW loss_summary AS
SELECT pl.kind,
       COUNT(*)                                            AS n,
       SUM(CASE WHEN pl.recovered THEN 1 ELSE 0 END)       AS n_recovered,
       MEDIAN(ABS(pl.energy_err_db))                       AS med_abs_err_db,
       QUANTILE_CONT(ABS(pl.energy_err_db), 0.95)          AS p95_abs_err_db,
       MEDIAN(ABS(pl.f0_err_pct))                          AS med_f0_err_pct,
       MEDIAN(ABS(pl.spacing_err_pct))                     AS med_spacing_err_pct
FROM pattern_loss pl WHERE pl.energy_recon IS NOT NULL GROUP BY 1;
