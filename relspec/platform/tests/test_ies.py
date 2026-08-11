"""Fast-SC IES second-opinion endpoint. A short campaign stores two raw
waveforms (first_waveform, then a faulty weekly_baseline) with one
codec-only day between them; the endpoint must find the BPFO line on the
fault, stay quiet at bearing orders on the healthy record, 404 honestly on
the codec-only acquisition, and honour its immutable ETag."""
import numpy as np
from relspec.synth2 import FaultState2
from conftest import make_wave, b64f32, make_ws, make_tree, hdr

BPFO, BPFI = 3.5848, 5.4152    # SKF6205 DE orders (synth pump bearing)

def _post(client, tok, sid, day, x):
    body = dict(sensor_id=sid, ts=f'2026-04-{day:02d}T06:00:00Z',
                fs=12000, units='g', encoding='float32',
                data_b64=b64f32(x), client_ref=f'd{day}')
    v = client.post('/v1/waveforms', json=body, headers=hdr(tok)).json()
    assert 'acq_id' in v, v
    return v

def test_ies_second_opinion(client):
    ws = make_ws(client)
    tok = ws['token']
    sid = make_tree(client, tok)

    v1 = _post(client, tok, sid, 1, make_wave(seed=11))          # healthy, stored
    v2 = _post(client, tok, sid, 2, make_wave(seed=12))          # healthy, codec-only
    v3 = _post(client, tok, sid, 9,                              # fault, weekly baseline
               make_wave(fault=FaultState2(outer_race=0.5), seed=13))
    assert v1['waveform_stored'] and v1['sig_reason'] == 'first_waveform'
    assert not v2['waveform_stored']
    assert v3['waveform_stored'] and v3['sig_reason'] == 'weekly_baseline'

    # fault record: the BPFO line stands clear of the advisory threshold
    r = client.get(f"/v1/acquisitions/{v3['acq_id']}/ies", headers=hdr(tok))
    assert r.status_code == 200, r.text
    out = r.json()
    assert out['fr'] and abs(out['fr'] - 29.5) / 29.5 < 0.03
    assert 1 <= len(out['alpha_hz']) == len(out['ies']) <= 256
    hits = [p for p in out['peaks']
            if abs(p['order'] - BPFO) < 0.025 * BPFO
            and p['snr_db'] > out['advisory_db']]
    assert hits, f"no BPFO advisory line in peaks: {out['peaks']}"
    assert out['advisory'] is True

    # healthy record: no advisory-strength line at either bearing order
    r = client.get(f"/v1/acquisitions/{v1['acq_id']}/ies", headers=hdr(tok))
    assert r.status_code == 200, r.text
    h = r.json()
    bad = [p for p in h['peaks'] if p['snr_db'] > h['advisory_db']
           and any(abs(p['order'] - o) < 0.025 * o for o in (BPFO, BPFI))]
    assert not bad, f"healthy record shows a bearing advisory: {bad}"

    # codec-only acquisition: honest 404, not an empty spectrum
    r = client.get(f"/v1/acquisitions/{v2['acq_id']}/ies", headers=hdr(tok))
    assert r.status_code == 404
    assert 'raw waveform' in r.json()['detail']

    # unknown acquisition (and other workspaces) are indistinguishable
    r = client.get('/v1/acquisitions/nope/ies', headers=hdr(tok))
    assert r.status_code == 404

    # immutable resource: the ETag round-trips as a 304
    r = client.get(f"/v1/acquisitions/{v3['acq_id']}/ies", headers=hdr(tok))
    etag = r.headers['etag']
    r = client.get(f"/v1/acquisitions/{v3['acq_id']}/ies",
                   headers={**hdr(tok), 'If-None-Match': etag})
    assert r.status_code == 304
