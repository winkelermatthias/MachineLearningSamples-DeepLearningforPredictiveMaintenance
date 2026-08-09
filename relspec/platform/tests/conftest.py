import os, sys, pathlib, uuid
import pytest

PLATFORM = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM / 'server'))
sys.path.insert(0, str(PLATFORM.parent / 'src'))   # relspec package

import pgserver  # noqa: E402

@pytest.fixture(scope='session')
def pg():
    d = str(PLATFORM / '.pgdata-test')
    srv = pgserver.get_server(d)
    yield srv
    srv.cleanup()

@pytest.fixture()
def client(pg, monkeypatch):
    """Fresh database + migrated schema + TestClient per test."""
    dbname = 'relspec_' + uuid.uuid4().hex[:10]
    import psycopg
    with psycopg.connect(pg.get_uri(), autocommit=True) as c:
        c.execute(f'CREATE DATABASE {dbname}')
    uri = pg.get_uri().replace('/postgres?', f'/{dbname}?')
    from relspec_service import config, db
    monkeypatch.setattr(config, 'DATABASE_URL', uri)
    db.close_pool()
    db.migrate()
    from relspec_service.app import app
    from fastapi.testclient import TestClient
    with TestClient(app) as tc:
        yield tc
    db.close_pool()

def make_wave(fault=None, seconds=2.0, fr=29.5, seed=1, load=0.8):
    """A synthetic pump waveform in g, deterministic per seed."""
    import numpy as np
    from relspec.synth2 import FaultState2, machine_catalog2, generate2
    spec = machine_catalog2()['centrifugal_pump']
    f = fault or FaultState2()
    rng = np.random.default_rng(seed)
    return generate2(spec, f, seconds, fr, load=load,
                     speed_wander_pct=0.5, rng=rng).astype('float64')

def b64f32(x):
    import base64, numpy as np
    return base64.b64encode(np.asarray(x, dtype='<f4').tobytes()).decode()

PHRASE = 'correct horse battery staple relspec'

def make_ws(client, phrase=PHRASE, params=None):
    r = client.post('/v1/workspaces',
                    json=dict(passphrase=phrase, name='test',
                              params=params or {}))
    assert r.status_code == 200, r.text
    return r.json()

def hdr(tok):
    return {'Authorization': f'Bearer {tok}'}

def make_tree(client, tok):
    tree = dict(plants=[dict(name='P1', assets=[dict(tag='PUMP-1',
                machine_type='pump', components=[dict(name='DE-brg',
                kind='bearing', sensors=[dict(code='V1', units='g',
                fs=12000, fr_nominal=29.5)])])])])
    r = client.post('/v1/hierarchy:ensure', json=tree, headers=hdr(tok))
    assert r.status_code == 200, r.text
    return r.json()['plants'][0]['assets'][0]['components'][0]['sensors'][0]['sensor_id']
