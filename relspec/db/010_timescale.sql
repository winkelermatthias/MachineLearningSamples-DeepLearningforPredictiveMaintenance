-- ===========================================================================
-- 010_timescale.sql : Postgres / TimescaleDB only.
-- Run after 001-003. DuckDB skips this file and uses 011 instead.
-- ===========================================================================
CREATE EXTENSION IF NOT EXISTS timescaledb;

SELECT create_hypertable('acquisition','ts', chunk_time_interval => interval '7 days',
                         migrate_data => true);
ALTER TABLE acquisition SET (timescaledb.compress,
    timescaledb.compress_segmentby='channel_id', timescaledb.compress_orderby='ts DESC');
SELECT add_compression_policy('acquisition', interval '30 days');

-- ---- render rail: dense 128-bin uint8, the only thing the pyramid reads ----
CREATE TABLE spec_render (
    ts        TIMESTAMPTZ NOT NULL,
    channel_id BIGINT NOT NULL,
    rail      TEXT NOT NULL,
    fr_hz     REAL NOT NULL,
    bins      SMALLINT[] NOT NULL,
    CONSTRAINT bins_len CHECK (array_length(bins,1) = 128)
);
SELECT create_hypertable('spec_render','ts', chunk_time_interval => interval '7 days');
ALTER TABLE spec_render SET (timescaledb.compress,
    timescaledb.compress_segmentby='channel_id, rail', timescaledb.compress_orderby='ts DESC');
SELECT add_compression_policy('spec_render', interval '7 days');

-- Element-wise MAX. Peak-hold, not mean: averaging across a zoom level makes
-- transients vanish exactly when you zoom out to look for them.
CREATE OR REPLACE FUNCTION arr_max_sf(smallint[], smallint[])
RETURNS smallint[] LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS $$
  SELECT array_agg(GREATEST(a,b) ORDER BY i)
  FROM unnest($1,$2) WITH ORDINALITY AS t(a,b,i);
$$;
CREATE AGGREGATE arr_max(smallint[]) (
  SFUNC=arr_max_sf, STYPE=smallint[], COMBINEFUNC=arr_max_sf, PARALLEL=SAFE);

-- Hierarchical pyramid. Payload size at render time is bounded regardless of
-- the span requested, because the row budget is fixed and the level is chosen
-- to match it.
CREATE MATERIALIZED VIEW wf_1h WITH (timescaledb.continuous) AS
SELECT time_bucket('1 hour', ts) AS bucket, channel_id, rail,
       arr_max(bins) AS mx, count(*)::int AS n, avg(fr_hz)::real AS fr_hz
FROM spec_render GROUP BY 1,2,3 WITH NO DATA;

CREATE MATERIALIZED VIEW wf_6h WITH (timescaledb.continuous) AS
SELECT time_bucket('6 hours', bucket) AS bucket, channel_id, rail,
       arr_max(mx) AS mx, sum(n)::int AS n, avg(fr_hz)::real AS fr_hz
FROM wf_1h GROUP BY 1,2,3 WITH NO DATA;

CREATE MATERIALIZED VIEW wf_1d WITH (timescaledb.continuous) AS
SELECT time_bucket('1 day', bucket) AS bucket, channel_id, rail,
       arr_max(mx) AS mx, sum(n)::int AS n, avg(fr_hz)::real AS fr_hz
FROM wf_6h GROUP BY 1,2,3 WITH NO DATA;

CREATE MATERIALIZED VIEW wf_7d WITH (timescaledb.continuous) AS
SELECT time_bucket('7 days', bucket) AS bucket, channel_id, rail,
       arr_max(mx) AS mx, sum(n)::int AS n, avg(fr_hz)::real AS fr_hz
FROM wf_1d GROUP BY 1,2,3 WITH NO DATA;

SELECT add_continuous_aggregate_policy('wf_1h', start_offset=>interval '3 days',
  end_offset=>interval '1 hour', schedule_interval=>interval '30 min');
SELECT add_continuous_aggregate_policy('wf_6h', start_offset=>interval '14 days',
  end_offset=>interval '6 hours', schedule_interval=>interval '2 hours');
SELECT add_continuous_aggregate_policy('wf_1d', start_offset=>interval '60 days',
  end_offset=>interval '1 day', schedule_interval=>interval '6 hours');
SELECT add_continuous_aggregate_policy('wf_7d', start_offset=>interval '365 days',
  end_offset=>interval '7 days', schedule_interval=>interval '1 day');

-- Order trend: one row per forcing line per bucket. This is the table the
-- trending UI and every model should read, and it exists because the ontology
-- knows which bin each kinematic line lands in.
CREATE MATERIALIZED VIEW order_trend_1d WITH (timescaledb.continuous) AS
SELECT time_bucket('1 day', ts) AS bucket, channel_id, rail,
       arr_max(bins) AS mx, avg(fr_hz)::real AS fr_hz, count(*)::int AS n
FROM spec_render GROUP BY 1,2,3 WITH NO DATA;

SELECT add_retention_policy('spec_render', interval '180 days');
