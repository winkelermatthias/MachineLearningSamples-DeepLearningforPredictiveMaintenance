"""Connection pool + migration runner.

Migrations are the .sql files in sql/, applied in filename order inside a
transaction each, tracked in schema_migrations. Files whose name contains
'timescale' are applied only when the timescaledb extension is available,
statement by statement in autocommit (continuous aggregates refuse to be
created inside a transaction block).
"""
from __future__ import annotations
import pathlib, re
import psycopg
from psycopg_pool import ConnectionPool
from . import config

SQL_DIR = pathlib.Path(__file__).parent / 'sql'
_pool: ConnectionPool | None = None

def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(config.DATABASE_URL, min_size=1, max_size=8,
                               kwargs={'autocommit': False}, open=True)
    return _pool

def close_pool():
    global _pool
    if _pool is not None:
        _pool.close(); _pool = None

def _split_statements(sql: str):
    """Top-level split on ';' — good enough for our migration files, which
    contain no procedural bodies with embedded semicolons except DO blocks,
    which we do not use in the timescale file."""
    out, buf = [], []
    for line in sql.splitlines():
        s = line.strip()
        if s.startswith('--') and not buf:
            continue
        buf.append(line)
        if s.endswith(';'):
            stmt = '\n'.join(buf).strip()
            if stmt and stmt != ';': out.append(stmt)
            buf = []
    tail = '\n'.join(buf).strip()
    if tail: out.append(tail)
    return out

def migrate():
    files = sorted(SQL_DIR.glob('*.sql'))
    with psycopg.connect(config.DATABASE_URL, autocommit=True) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS schema_migrations (
            filename TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now())""")
        ts_available = conn.execute(
            "SELECT 1 FROM pg_available_extensions WHERE name='timescaledb'"
        ).fetchone() is not None
        for f in files:
            done = conn.execute(
                'SELECT 1 FROM schema_migrations WHERE filename=%s',
                (f.name,)).fetchone()
            if done: continue
            sql = f.read_text()
            if 'timescale' in f.name:
                if not ts_available:
                    # recorded as applied so a later Timescale-capable boot
                    # doesn't try to hypertable a table with data under it
                    # unexpectedly; converting later is a manual operation.
                    conn.execute(
                        'INSERT INTO schema_migrations VALUES (%s)', (f.name,))
                    continue
                for stmt in _split_statements(sql):
                    conn.execute(stmt)
                conn.execute('INSERT INTO schema_migrations VALUES (%s)', (f.name,))
            else:
                with psycopg.connect(config.DATABASE_URL) as tx:
                    tx.execute(sql)
                    tx.execute('INSERT INTO schema_migrations VALUES (%s)', (f.name,))
                    tx.commit()

def bump_version(cur, workspace_id: str) -> int:
    row = cur.execute(
        'UPDATE workspace SET data_version = data_version+1 '
        'WHERE workspace_id=%s RETURNING data_version', (workspace_id,)).fetchone()
    return row[0]
