"""The exit test from the plan: the service pipeline — with its state
round-tripping through (de)serialisation between every frame, as it does
through the database — must produce byte-identical codec payloads and
decoded spectra to the offline pipeline holding live objects. Any drift
here means stored history and offline evaluation would disagree."""
import numpy as np
from conftest import make_wave

def test_service_pipeline_matches_offline():
    from relspec_service import pipeline as sp
    from relspec.pipeline import Gate, NBINS, ENV_BINS
    from relspec.codec2 import Codec2
    from relspec import pipeline2
    from relspec.pipeline2 import extract2

    waves = [make_wave(seed=s) for s in range(1, 9)]
    sensor = dict(sensor_id='parity-test', fr_nominal=29.5,
                  band_lo_hz=None, band_hi_hz=None)

    # --- service path: state serialised between every frame
    blobs, gate_blob = {}, None
    tracks = {'acc': [], 'env': []}
    svc = []
    band = None
    for i, x in enumerate(waves):
        s2 = dict(sensor)
        if band: s2['band_lo_hz'], s2['band_hi_hz'] = band
        res = sp.process(x, 12000.0, s2, blobs, gate_blob, tracks,
                         t_epoch=i*86400)
        blobs = res['codec_blobs']; gate_blob = res['gate_blob']
        band = res['band']
        svc.append(res)

    # --- offline path: live objects, same inputs
    pipeline2._band_cache.clear()
    pipeline2._band_cache['offline'] = band  # same pinned band
    enc_a = Codec2(nbins=NBINS, n_bands=16, refs='best')
    enc_e = Codec2(nbins=ENV_BINS, n_peaks=32, n_bands=12, refs='best')
    mad_a = np.full(NBINS, 4.0); mad_e = np.full(ENV_BINS, 4.0)
    gate = Gate()
    for i, x in enumerate(waves):
        e = extract2(x, 12000.0, band_key='offline', fr_nominal=29.5)
        pa, ka = enc_a.encode(e.acc_u8, mad_a)
        pe, ke = enc_e.encode(e.env_u8, mad_e)
        # offline decode = encoder's own model (prev), anchors lossless
        ra = (np.asarray(enc_a.prev) if 'residual' in ka
              else np.asarray(enc_a.anchor))
        re_ = (np.asarray(enc_e.prev) if 'residual' in ke
               else np.asarray(enc_e.anchor))
        mad_a = np.maximum(0.95*mad_a+0.05*np.abs(e.acc_u8.astype(float)-ra), 1.0)
        mad_e = np.maximum(0.95*mad_e+0.05*np.abs(e.env_u8.astype(float)-re_), 1.0)
        if gate.n == 0: gate.init(e)
        else: gate.decide(e, i*86400)

        f = svc[i]['frames']
        assert f['acc']['payload'] == pa, f'acc payload differs at frame {i}'
        assert f['env']['payload'] == pe, f'env payload differs at frame {i}'
        assert np.array_equal(
            np.frombuffer(f['acc']['dec'], dtype=np.uint8),
            np.clip(ra, 0, 255).astype(np.uint8)), f'acc decode differs at {i}'
        assert np.array_equal(
            np.frombuffer(f['env']['dec'], dtype=np.uint8),
            np.clip(re_, 0, 255).astype(np.uint8)), f'env decode differs at {i}'
    # gate state at the end agrees too
    import json
    d = json.loads(gate_blob.decode())
    assert d['n'] == gate.n and d['n_slow'] == gate.n_slow
    assert np.allclose(np.array(d['med']), gate.med)
