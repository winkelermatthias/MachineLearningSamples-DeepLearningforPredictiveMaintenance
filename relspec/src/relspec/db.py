"""Database layer.

Canonical DDL targets Postgres + TimescaleDB (db/001-003, 010). DuckDB runs the
same 001-003 unchanged and substitutes 011 for the continuous aggregates, which
is why the ontology views are written in portable SQL: one schema, two engines,
no divergence in the part that matters.
"""
from __future__ import annotations
import duckdb, numpy as np, json, os
from pathlib import Path

DDL_DIR = Path(__file__).resolve().parents[2]/'db'

DUCK_PYRAMID = """
-- 011_duckdb.sql equivalent. DuckDB has no continuous aggregates, so the
-- pyramid is materialised by the pipeline and refreshed on demand. Same shape,
-- same query surface, so application code does not branch on engine.
CREATE TABLE spec_render (
    ts TIMESTAMP NOT NULL, channel_id BIGINT NOT NULL, rail VARCHAR NOT NULL,
    fr_hz REAL NOT NULL, bins BLOB NOT NULL
);
CREATE TABLE wf_level (
    level VARCHAR NOT NULL, bucket TIMESTAMP NOT NULL, channel_id BIGINT NOT NULL,
    rail VARCHAR NOT NULL, n INTEGER NOT NULL, fr_hz REAL, mx BLOB NOT NULL
);
CREATE TABLE order_trend (
    bucket TIMESTAMP NOT NULL, channel_id BIGINT NOT NULL, rail VARCHAR NOT NULL,
    forcing_id BIGINT NOT NULL, label VARCHAR NOT NULL, family VARCHAR NOT NULL,
    order_value DOUBLE NOT NULL, amp_db DOUBLE NOT NULL, n INTEGER NOT NULL
);
"""

class DB:
    def __init__(self, path=':memory:'):
        self.con = duckdb.connect(path)
        self.con.execute("SET preserve_insertion_order=false")

    def migrate(self, ddl_dir=DDL_DIR):
        for f in ['001_ontology.sql', '002_measurement.sql', '003_views.sql']:
            sql = (ddl_dir/f).read_text()
            sql = sql.replace('DOUBLE PRECISION', 'DOUBLE')
            for stmt in _split(sql):
                try: self.con.execute(stmt)
                except Exception as e:
                    raise RuntimeError(f'{f}: {e}\n---\n{stmt[:400]}')
        for stmt in _split(DUCK_PYRAMID):
            self.con.execute(stmt)
        return self

    def q(self, sql, *a):
        return self.con.execute(sql, list(a)).fetchdf()

    def insert_df(self, table, df):
        if len(df) == 0: return
        self.con.register('_t', df)
        cols = ','.join(df.columns)
        self.con.execute(f'INSERT INTO {table} ({cols}) SELECT {cols} FROM _t')
        self.con.unregister('_t')

def _split(sql):
    out, buf, depth = [], [], 0
    for line in sql.splitlines():
        s = line.split('--')[0] if not line.strip().startswith('--') else ''
        if line.strip().startswith('--'): continue
        buf.append(line)
        depth += line.count('(')-line.count(')')
        if ';' in line and depth <= 0:
            stmt = '\n'.join(buf).strip()
            stmt = '\n'.join(l for l in stmt.splitlines()
                             if not l.strip().startswith('--'))
            if stmt.strip(' ;\n'): out.append(stmt)
            buf, depth = [], 0
    return out

# --------------------------------------------------------------- fleet build
def build_fleet(db: DB, n_assets=16, seed=11):
    """Instantiate a small estate with real kinematics. Everything downstream
    reads the ontology, so this is the only place machine structure is stated."""
    from .synth import machine_catalog, CATALOG
    rng = np.random.default_rng(seed)
    cat = machine_catalog(); classes = list(cat)
    rows = dict(org=[], site=[], area=[], asset=[], component=[], shaft=[],
                bearing_catalog=[], bearing_instance=[], gear_mesh=[],
                passing_element=[], sensor=[], channel=[], channel_reference=[])
    rows['org'].append((1, 'Northwind Minerals'))
    for s in range(1, 3):
        rows['site'].append((s, 1, f'Site {s}', 'UTC', 53.5+s, -113.5-s))
        for a in range(1, 3):
            rows['area'].append((s*10+a, s, f'Area {a}',
                                 ['crushing', 'flotation'][a-1]))
    for code, b in CATALOG.items():
        rows['bearing_catalog'].append((code, code[:3], b.n_rollers,
                                        b.bpfo, b.bpfi, b.bsf, b.ftf))
    ids = dict(asset=0, component=0, shaft=0, bearing=0, sensor=0, channel=0,
               mesh=0, pass_=0)
    specs = {}
    for i in range(n_assets):
        cls = classes[i % len(classes)]
        spec = cat[cls]
        ids['asset'] += 1; aid = ids['asset']
        area = list({r[0] for r in rows['area']})[i % 4]
        rpm = float(rng.choice([1480, 1780, 990, 3560, 740]))
        rows['asset'].append((aid, area, f'{cls[:2].upper()}-{aid:03d}',
                              f'{cls.replace("_"," ").title()} {aid}',
                              spec.asset_class, int(rng.integers(2, 6)),
                              'continuous' if i % 3 else 'intermittent',
                              'vfd' if i % 5 == 0 else 'dol',
                              float(rng.choice([55, 110, 200, 315])), None))
        ids['component'] += 1; cid = ids['component']
        rows['component'].append((cid, aid, 'drive end', cls, 1, None, None))
        ids['shaft'] += 1; sid = ids['shaft']
        rows['shaft'].append((sid, cid, 'main', True, 1.0, rpm))
        for bearing, ratio, pos in spec.kin.bearings:
            ids['bearing'] += 1
            rows['bearing_instance'].append((ids['bearing'], sid, bearing.code, pos, None, None))
        if spec.kin.gear_teeth:
            ids['mesh'] += 1
            rows['gear_mesh'].append((ids['mesh'], cid, sid, sid,
                                      spec.kin.gear_teeth[0], spec.kin.gear_teeth[1]))
        if spec.kin.n_vanes:
            ids['pass_'] += 1
            kind = 'vane' if spec.asset_class in ('pump',) else 'blade'
            rows['passing_element'].append((ids['pass_'], cid, sid, kind, spec.kin.n_vanes))
        ids['sensor'] += 1; snid = ids['sensor']
        rows['sensor'].append((snid, cid, f'MD-{snid:05d}', 'MachineDoctor',
                               '2.4.1', 'stud', None, 18500.0))
        for rail in ('acc', 'env'):
            ids['channel'] += 1; chid = ids['channel']
            rows['channel'].append((chid, snid, 'radial', 'acceleration', rail, 12000.0))
            rows['channel_reference'].append((chid, sid))
            if rail == 'acc':
                specs[chid] = dict(asset_id=aid, spec=spec, cls=cls, fr=rpm/60.0,
                                   sensor_id=snid, shaft_id=sid,
                                   criticality=rows['asset'][-1][5])
    # explicit column lists: positional inserts break silently the moment the
    # schema gains a column, and this schema will gain columns
    COLS = {
      'org': 'org_id,name',
      'site': 'site_id,org_id,name,tz,lat,lon',
      'area': 'area_id,site_id,name,process_stage',
      'asset': 'asset_id,area_id,tag,name,asset_class,criticality,duty,drive,rated_kw,commissioned',
      'component': 'component_id,asset_id,name,kind,position,installed,removed',
      'shaft': 'shaft_id,component_id,name,is_reference,ratio_to_ref,nominal_rpm',
      'bearing_catalog': 'bearing_type,manufacturer,n_rollers,bpfo,bpfi,bsf,ftf',
      'bearing_instance': 'bearing_id,shaft_id,bearing_type,position,installed,removed',
      'gear_mesh': 'mesh_id,component_id,drive_shaft,driven_shaft,drive_teeth,driven_teeth',
      'passing_element': 'pass_id,component_id,shaft_id,kind,n_elements',
      'sensor': 'sensor_id,component_id,serial,model,fw_version,mount,installed,battery_j',
      'channel': 'channel_id,sensor_id,axis,quantity,rail,fs_hz',
      'channel_reference': 'channel_id,shaft_id',
    }
    for t, cols in COLS.items():
        if rows.get(t):
            ph = ','.join(['?']*len(cols.split(',')))
            db.con.executemany(f'INSERT INTO {t} ({cols}) VALUES ({ph})', rows[t])
    return specs
