"""Real-data run on CWRU: dual-domain codec, pattern extraction in both
domains, original vs reconstructed waterfalls, and an ontology view benchmark.
Emits report_data.json for the HTML report."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__) or '.')
import numpy as np, scipy.io as sio, scipy.signal as sg
from relspec.pipeline import (extract, Codec, Decoder, CENTERS, ENV_CENTERS,
                              to_amp, Q_DB, NBINS)
from relspec.patterns import extract_patterns

DATA = os.environ.get('CWRU', '../../data')
FS = 12000.0
SEQ = [('97',0,'normal')]*6 + [('105',0,'IR 0.007in')]*6 + \
      [('169',0,'IR 0.014in')]*6 + [('209',0,'IR 0.021in')]*6
WIN = 2.0

def load(fid):
    m = sio.loadmat(f'{DATA}/{fid}.mat')
    return m[[k for k in m if k.endswith('_DE_time')][0]].ravel().astype(np.float64)

def build():
    rng = np.random.default_rng(3); n = int(WIN*FS); frames = []
    for k, (fid, load_hp, label) in enumerate(SEQ):
        x = load(fid); s = rng.integers(0, len(x)-n)
        frames.append(dict(k=k, day=k/2.0, seg=x[s:s+n], label=label, fid=fid))
    return frames

def run():
    frames = build()
    ca, cd = Codec(), Decoder()
    ce = Codec(nbins=len(ENV_CENTERS), n_peaks=32, n_bands=12)
    de = Decoder(nbins=len(ENV_CENTERS), n_bands=12)
    mad_a = np.full(NBINS, 4.0); mad_e = np.full(len(ENV_CENTERS), 4.0)
    out = dict(acc_o=[], acc_r=[], env_o=[], env_r=[], bytes=[], kinds=[],
               labels=[], days=[], t_extract=[], t_enc=[], t_dec=[],
               pat_o=[], pat_r=[], bal_o=[], bal_r=[], fr=[])
    raw_lines = 0
    for f in frames:
        t0 = time.perf_counter(); e = extract(f['seg']); t_ex = time.perf_counter()-t0
        raw_lines += 2*(4096//2+1)
        t0 = time.perf_counter()
        pa, ka = ca.encode(e.acc_u8, mad_a)
        pe, ke = ce.encode(e.env_u8, mad_e)
        t_enc = time.perf_counter()-t0
        t0 = time.perf_counter()
        ra, _ = cd.decode(pa); re_, _ = de.decode(pe)
        t_dec = time.perf_counter()-t0
        mad_a = np.maximum(0.95*mad_a+0.05*np.abs(e.acc_u8.astype(float)-ra.astype(float)), 1.0)
        mad_e = np.maximum(0.95*mad_e+0.05*np.abs(e.env_u8.astype(float)-re_.astype(float)), 1.0)
        ao, ar = to_amp(e.acc_u8), to_amp(ra)
        eo, er = to_amp(e.env_u8), to_amp(re_)
        _, _, bo = extract_patterns(ao, CENTERS)
        _, _, br = extract_patterns(ar, CENTERS)
        _, _, beo = extract_patterns(eo, ENV_CENTERS, min_harmonics=3, f_hi=6.5)
        _, _, ber = extract_patterns(er, ENV_CENTERS, min_harmonics=3, f_hi=6.5)
        out['acc_o'].append(e.acc_u8.tolist()); out['acc_r'].append(ra.tolist())
        out['env_o'].append(e.env_u8.tolist()); out['env_r'].append(re_.tolist())
        out['bytes'].append(len(pa)+len(pe)); out['kinds'].append(ka+'/'+ke)
        out['labels'].append(f['label']); out['days'].append(f['day'])
        out['fr'].append(round(e.fr, 3))
        out['t_extract'].append(t_ex*1000); out['t_enc'].append(t_enc*1000)
        out['t_dec'].append(t_dec*1000)
        strip = lambda b: dict(resid=b['residual_share'],
            pats=[{k: p[k] for k in ('kind','key','energy','share','spacing','snr')}
                  for p in b['patterns'] if p['energy'] > 0])
        out['pat_o'].append(strip(bo)); out['pat_r'].append(strip(br))
        out['bal_o'].append(strip(beo)); out['bal_r'].append(strip(ber))
    out['raw_bytes'] = raw_lines*4
    out['coded_bytes'] = int(sum(out['bytes']))
    try:
        import zstandard as zstd
        c = zstd.ZstdCompressor(level=19)
        out['z_bytes'] = out['coded_bytes']   # per-frame dict not meaningful at n=24
    except Exception:
        out['z_bytes'] = out['coded_bytes']
    out['centers'] = [round(x, 4) for x in CENTERS.tolist()]
    out['env_centers'] = [round(x, 4) for x in ENV_CENTERS.tolist()]
    return out

def bench_ontology():
    from relspec.db import DB, build_fleet
    res = {}
    for n in (8, 32, 128):
        db = DB().migrate(); build_fleet(db, n)
        db.con.execute('CREATE TABLE ff_mat AS SELECT * FROM forcing_frequency_k')
        rows = int(db.q('SELECT count(*) AS c FROM ff_mat').c[0])
        def timeit(sql, reps=20):
            db.q(sql)
            t = time.perf_counter()
            for _ in range(reps): db.q(sql)
            return (time.perf_counter()-t)/reps*1000
        res[n] = dict(
            assets=n, lines=rows,
            view_full=timeit('SELECT count(*) AS c FROM forcing_frequency_k'),
            mat_full=timeit('SELECT count(*) AS c FROM ff_mat'),
            view_ch=timeit('SELECT * FROM forcing_frequency_k WHERE channel_id=3'),
            mat_ch=timeit('SELECT * FROM ff_mat WHERE channel_id=3'),
            view_fam=timeit("SELECT family, count(*) AS c FROM forcing_frequency_k GROUP BY 1"),
            mat_fam=timeit("SELECT family, count(*) AS c FROM ff_mat GROUP BY 1"))
        db.con.close()
    return res

if __name__ == '__main__':
    d = run()
    print('frames', len(d['days']), 'coded', d['coded_bytes'],
          'ratio %.1fx' % (d['raw_bytes']/d['coded_bytes']))
    print('extract %.1f ms  enc %.2f ms  dec %.2f ms'
          % (np.mean(d['t_extract']), np.mean(d['t_enc']), np.mean(d['t_dec'])))
    d['ontology'] = bench_ontology()
    for n, r in d['ontology'].items():
        print(f"  {r['assets']:>4} assets {r['lines']:>6} lines | view {r['view_full']:.2f}/"
              f"{r['view_ch']:.2f}/{r['view_fam']:.2f} ms | mat {r['mat_full']:.2f}/"
              f"{r['mat_ch']:.2f}/{r['mat_fam']:.2f} ms")
    json.dump(d, open('../report_data.json', 'w'))
    print('wrote report_data.json  %.1f KB' % (os.path.getsize('../report_data.json')/1024))
