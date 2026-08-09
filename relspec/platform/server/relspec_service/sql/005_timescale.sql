-- Applied ONLY when the timescaledb extension is available (the migration
-- runner checks pg_available_extensions and runs this file statement by
-- statement in autocommit, as continuous aggregates cannot be created
-- inside a transaction).
CREATE EXTENSION IF NOT EXISTS timescaledb;

SELECT create_hypertable('acquisition', 'ts',
        chunk_time_interval => INTERVAL '7 days', migrate_data => TRUE);
SELECT create_hypertable('waveform', 'ts',
        chunk_time_interval => INTERVAL '30 days', migrate_data => TRUE);
SELECT create_hypertable('event', 'ts',
        chunk_time_interval => INTERVAL '30 days', migrate_data => TRUE);

ALTER TABLE acquisition SET (timescaledb.compress,
        timescaledb.compress_segmentby = 'sensor_id');
SELECT add_compression_policy('acquisition', INTERVAL '7 days');
ALTER TABLE waveform SET (timescaledb.compress,
        timescaledb.compress_segmentby = 'sensor_id');
SELECT add_compression_policy('waveform', INTERVAL '7 days');
SELECT add_retention_policy('waveform', INTERVAL '2 years');

-- Replace the portable trend view with a continuous aggregate of the same
-- name so the API's SQL is identical in both modes.
DROP VIEW IF EXISTS trend_daily;

CREATE MATERIALIZED VIEW trend_daily
WITH (timescaledb.continuous) AS
SELECT workspace_id, sensor_id,
       time_bucket(INTERVAL '1 day', ts) AS bucket,
       count(*)              AS n,
       avg(vel_rms)          AS vel_avg,
       max(vel_rms)          AS vel_max,
       avg(acc_rms)          AS acc_avg,
       max(acc_rms)          AS acc_max,
       avg(env_rms)          AS env_avg,
       max(env_rms)          AS env_max,
       max(gate_score)       AS gate_max,
       sum(payload_bytes)    AS bytes
FROM acquisition
GROUP BY 1, 2, 3
WITH NO DATA;

SELECT add_continuous_aggregate_policy('trend_daily',
        start_offset => INTERVAL '3 days',
        end_offset   => INTERVAL '1 hour',
        schedule_interval => INTERVAL '30 minutes');
