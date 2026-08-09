-- Portable views. Every read query in the API targets these names; in
-- Timescale mode 005 replaces trend_daily with a continuous aggregate of
-- the same name and adds refresh/compression policies. Plain Postgres gets
-- correct (if lazier) answers from the same SQL, which is also what the
-- test suite runs against.
CREATE VIEW trend_daily AS
SELECT workspace_id, sensor_id,
       date_trunc('day', ts) AS bucket,
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
GROUP BY 1, 2, 3;

CREATE VIEW sensor_latest AS
SELECT DISTINCT ON (a.sensor_id)
       a.workspace_id, a.sensor_id, a.ts AS last_ts,
       a.vel_rms, a.acc_rms, a.env_rms,
       a.gate_score, a.gate_decision, a.fr_est,
       a.payload_bytes, a.waveform_stored
FROM acquisition a
ORDER BY a.sensor_id, a.ts DESC;
