-- Backstop for per-sensor ingest serialization: the advisory lock in
-- ingest.py serializes read-process-write per sensor; this index guarantees
-- duplicate (sensor, ts) rows cannot land even if a writer misses the lock.
-- ts is the hypertable partition column, so this stays valid under 005.
CREATE UNIQUE INDEX acq_sensor_ts_uniq ON acquisition(sensor_id, ts);
