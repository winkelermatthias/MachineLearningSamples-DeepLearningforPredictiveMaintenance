-- ============================================================================
-- ReliabilityOS :: compressed spectral store + waterfall pyramid
-- TimescaleDB >= 2.13 (hierarchical CAGGs, CAGG compression)
--
-- Two rails:
--   spec_archive  768-bin sparse delta vs baseline   ~65 B/spectrum   (analysis)
--   spec_render   128-bin dense uint8, order-linear  ~128 B/spectrum  (waterfall)
--
-- Axis convention for the render rail: ORDER-LINEAR, 0..20 orders of shaft
-- speed, 128 bins => 0.15625 order/bin. Amplitude: uint8 log-dB,
-- 0.5 dB/count, offset -40 dB => covers -40..+87.5 dB re 1 ug.
-- TODO: confirm dB reference against MachineDoctor calibration constant.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ---------------------------------------------------------------------------
-- 1. Baselines (fed by the existing MAD-robust rolling-baseline pipeline)
-- ---------------------------------------------------------------------------
CREATE TABLE spec_baseline (
    baseline_id   bigserial PRIMARY KEY,
    asset_id      bigint      NOT NULL,
    channel       smallint    NOT NULL,   -- 0=ACC-X 1=ACC-Y 2=ACC-Z 3=ENV
    valid_from    timestamptz NOT NULL,
    valid_to      timestamptz,
    n_spectra     integer     NOT NULL,   -- support behind this baseline
    med_db        bytea       NOT NULL,   -- 768 uint8, per-bin median
    mad_db        bytea       NOT NULL,   -- 768 uint8, per-bin MAD (0.5 dB/count)
    speed_hz      real        NOT NULL,   -- shaft speed the orders are locked to
    scale_ug      real        NOT NULL    -- absolute scale recovery, ug per dB-ref
);
CREATE INDEX ON spec_baseline (asset_id, channel, valid_from DESC);

-- ---------------------------------------------------------------------------
-- 2. Archive rail :: sparse delta
--    payload layout, little-endian, repeated n times:
--      uint16 bin_idx | int8 delta_db_halves | uint8 peak_offset (4b frac, 4b flags)
-- ---------------------------------------------------------------------------
CREATE TABLE spec_archive (
    ts            timestamptz NOT NULL,
    asset_id      bigint      NOT NULL,
    channel       smallint    NOT NULL,
    baseline_id   bigint      NOT NULL REFERENCES spec_baseline(baseline_id),
    speed_hz      real        NOT NULL,   -- measured speed for this spectrum
    speed_conf    real        NOT NULL,   -- MAVEN confidence, gate reconstruction on this
    coverage      real        NOT NULL,   -- fraction of band with valid data
    n_active      smallint    NOT NULL,   -- bins in payload
    payload       bytea       NOT NULL
);
SELECT create_hypertable('spec_archive', 'ts', chunk_time_interval => interval '7 days');

ALTER TABLE spec_archive SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'asset_id, channel',
    timescaledb.compress_orderby   = 'ts DESC'
);
SELECT add_compression_policy('spec_archive', interval '3 days');

-- ---------------------------------------------------------------------------
-- 3. Render rail :: dense 128-bin, smallint[] so we can aggregate element-wise
--    smallint not uint8 because Postgres has no unsigned type; values 0..255.
-- ---------------------------------------------------------------------------
CREATE TABLE spec_render (
    ts        timestamptz NOT NULL,
    asset_id  bigint      NOT NULL,
    channel   smallint    NOT NULL,
    speed_hz  real        NOT NULL,
    bins      smallint[]  NOT NULL,       -- exactly 128 elements
    CONSTRAINT bins_len CHECK (array_length(bins, 1) = 128)
);
SELECT create_hypertable('spec_render', 'ts', chunk_time_interval => interval '7 days');

ALTER TABLE spec_render SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'asset_id, channel',
    timescaledb.compress_orderby   = 'ts DESC'
);
SELECT add_compression_policy('spec_render', interval '3 days');

-- ---------------------------------------------------------------------------
-- 4. Element-wise aggregates
--    STRICT + no INITCOND => first non-null row seeds the state directly.
--    GREATEST/LEAST ignore NULLs, so ragged arrays degrade safely.
--    PARALLEL SAFE + COMBINEFUNC are REQUIRED for use inside a CAGG.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION arr_max_sf(smallint[], smallint[])
RETURNS smallint[]
LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS $$
    SELECT array_agg(GREATEST(a, b) ORDER BY i)
    FROM unnest($1, $2) WITH ORDINALITY AS t(a, b, i);
$$;

CREATE AGGREGATE arr_max(smallint[]) (
    SFUNC       = arr_max_sf,
    STYPE       = smallint[],
    COMBINEFUNC = arr_max_sf,
    PARALLEL    = SAFE
);

-- Running sum + count for the "typical" rail. p50 is not distributive across
-- pyramid levels, so we carry a mean and accept it; use arr_max for anything
-- diagnostic and the mean only for the visual "typical" toggle.
CREATE OR REPLACE FUNCTION arr_sum_sf(integer[], smallint[])
RETURNS integer[]
LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS $$
    SELECT array_agg(COALESCE(a,0) + COALESCE(b,0)::int ORDER BY i)
    FROM unnest($1, $2) WITH ORDINALITY AS t(a, b, i);
$$;

CREATE OR REPLACE FUNCTION arr_sum_comb(integer[], integer[])
RETURNS integer[]
LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS $$
    SELECT array_agg(COALESCE(a,0) + COALESCE(b,0) ORDER BY i)
    FROM unnest($1, $2) WITH ORDINALITY AS t(a, b, i);
$$;

CREATE AGGREGATE arr_sum(smallint[]) (
    SFUNC       = arr_sum_sf,
    STYPE       = integer[],
    COMBINEFUNC = arr_sum_comb,
    PARALLEL    = SAFE
);

-- PERF NOTE: these are SQL-level O(128) per row. Fine at 50K sensors x 24/day
-- (~1.2M rows/day, incremental refresh only touches new buckets). If you push
-- to per-minute cadence, reimplement arr_max_sf in C or plrust; it is a
-- 20-line memcmp loop over bytea and runs ~50x faster.

-- ---------------------------------------------------------------------------
-- 5. The pyramid. L1 reads the hypertable; L2..L4 read the level below.
-- ---------------------------------------------------------------------------
CREATE MATERIALIZED VIEW wf_1h
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 hour', ts)   AS bucket,
       asset_id, channel,
       arr_max(bins)               AS mx,
       arr_sum(bins)               AS sm,
       count(*)::int               AS n,
       avg(speed_hz)::real         AS speed_hz
FROM spec_render
GROUP BY 1, 2, 3
WITH NO DATA;

CREATE MATERIALIZED VIEW wf_6h
WITH (timescaledb.continuous) AS
SELECT time_bucket('6 hours', bucket) AS bucket,
       asset_id, channel,
       arr_max(mx)                    AS mx,
       arr_sum_rollup(sm)             AS sm,   -- see note below
       sum(n)::int                    AS n,
       avg(speed_hz)::real            AS speed_hz
FROM wf_1h
GROUP BY 1, 2, 3
WITH NO DATA;

-- NOTE: wf_6h above needs an integer[] -> integer[] rollup, not smallint[].
-- Define it before running this block:
--   CREATE AGGREGATE arr_sum_rollup(integer[]) (
--       SFUNC = arr_sum_comb, STYPE = integer[],
--       COMBINEFUNC = arr_sum_comb, PARALLEL = SAFE);
-- Same aggregate is reused for wf_1d and wf_7d.

CREATE MATERIALIZED VIEW wf_1d
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 day', bucket) AS bucket,
       asset_id, channel,
       arr_max(mx) AS mx, arr_sum_rollup(sm) AS sm,
       sum(n)::int AS n, avg(speed_hz)::real AS speed_hz
FROM wf_6h GROUP BY 1, 2, 3 WITH NO DATA;

CREATE MATERIALIZED VIEW wf_7d
WITH (timescaledb.continuous) AS
SELECT time_bucket('7 days', bucket) AS bucket,
       asset_id, channel,
       arr_max(mx) AS mx, arr_sum_rollup(sm) AS sm,
       sum(n)::int AS n, avg(speed_hz)::real AS speed_hz
FROM wf_1d GROUP BY 1, 2, 3 WITH NO DATA;

-- Refresh policies. Lag the start_offset behind the parent so a child never
-- refreshes a bucket its parent has not finalised.
SELECT add_continuous_aggregate_policy('wf_1h',
    start_offset => interval '3 days',  end_offset => interval '1 hour',
    schedule_interval => interval '30 min');
SELECT add_continuous_aggregate_policy('wf_6h',
    start_offset => interval '14 days', end_offset => interval '6 hours',
    schedule_interval => interval '2 hours');
SELECT add_continuous_aggregate_policy('wf_1d',
    start_offset => interval '60 days', end_offset => interval '1 day',
    schedule_interval => interval '6 hours');
SELECT add_continuous_aggregate_policy('wf_7d',
    start_offset => interval '365 days', end_offset => interval '7 days',
    schedule_interval => interval '1 day');

ALTER MATERIALIZED VIEW wf_1h SET (timescaledb.compress = true);
ALTER MATERIALIZED VIEW wf_6h SET (timescaledb.compress = true);
SELECT add_compression_policy('wf_1h', compress_after => interval '30 days');
SELECT add_compression_policy('wf_6h', compress_after => interval '90 days');

-- Retention: the render rail is disposable, the archive rail is not.
SELECT add_retention_policy('spec_render', interval '90 days');
-- spec_archive: no retention. 65 B x 50K sensors x 4ch x 24/day = 312 MB/day.

-- ---------------------------------------------------------------------------
-- 6. Serving function. Auto-selects the pyramid level whose bucket count is
--    nearest the requested row budget, so payload size is bounded regardless
--    of the time span requested.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION waterfall(
    p_asset   bigint,
    p_channel smallint,
    p_from    timestamptz,
    p_to      timestamptz,
    p_rows    int DEFAULT 400,
    p_mode    text DEFAULT 'max'      -- 'max' | 'typical'
)
RETURNS TABLE (bucket timestamptz, speed_hz real, n int, bins smallint[])
LANGUAGE plpgsql STABLE AS $$
DECLARE
    span_s  double precision := extract(epoch FROM (p_to - p_from));
    target  double precision := span_s / GREATEST(p_rows, 1);
    lvl     text;
BEGIN
    -- pick the coarsest level that still yields >= p_rows buckets
    lvl := CASE
        WHEN target <   1800 THEN 'raw'
        WHEN target <  10800 THEN 'wf_1h'
        WHEN target <  43200 THEN 'wf_6h'
        WHEN target < 302400 THEN 'wf_1d'
        ELSE 'wf_7d'
    END;

    IF lvl = 'raw' THEN
        RETURN QUERY
        SELECT r.ts, r.speed_hz, 1, r.bins
        FROM spec_render r
        WHERE r.asset_id = p_asset AND r.channel = p_channel
          AND r.ts >= p_from AND r.ts < p_to
        ORDER BY r.ts;
        RETURN;
    END IF;

    RETURN QUERY EXECUTE format($q$
        SELECT w.bucket, w.speed_hz, w.n,
               CASE WHEN %L = 'max' THEN w.mx
                    ELSE (SELECT array_agg((s / GREATEST(w.n,1))::smallint ORDER BY i)
                          FROM unnest(w.sm) WITH ORDINALITY AS t(s, i))
               END
        FROM %I w
        WHERE w.asset_id = $1 AND w.channel = $2
          AND w.bucket >= $3 AND w.bucket < $4
        ORDER BY w.bucket
    $q$, p_mode, lvl)
    USING p_asset, p_channel, p_from, p_to;
END;
$$;

-- ---------------------------------------------------------------------------
-- 7. Client contract
--    Return bins as raw uint8. Do NOT render PNG server-side: keeping the
--    values lets the frontend change colormap, dB floor/ceiling, and
--    normalisation with zero round trips.
--
--    400 rows x 128 bins = 51,200 B raw, ~18 KB gzipped.
--
--    Canvas path:
--      const img = ctx.createImageData(128, rows.length);
--      rows.forEach((row, y) => row.bins.forEach((v, x) => {
--        const [r,g,b] = colormap(v);           // v is 0..255 log-dB
--        const o = (y * 128 + x) * 4;
--        img.data[o]=r; img.data[o+1]=g; img.data[o+2]=b; img.data[o+3]=255;
--      }));
--      ctx.putImageData(img, 0, 0);             // then scale with CSS
--
--    X axis is ORDERS, not Hz. Label harmonic combs at integer orders and the
--    bearing tones at their computed order positions; that is the whole point
--    of order-locking the axis. Show speed_hz in the row tooltip so an analyst
--    can convert back.
-- ---------------------------------------------------------------------------
