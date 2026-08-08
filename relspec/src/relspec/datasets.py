"""Loaders for the public datasets this repo evaluates against.

Three rigs, three sample rates, three bearing geometries, sourced from public
GitHub mirrors (see README for provenance):

  CWRU  Case Western bearing test rig, 12 kHz drive-end accelerometer.
        The full 12k DE matrix: normal, IR/B at 007/014/021/028 mil,
        OR at 007/014/021 mil in three load-zone positions, 4 motor loads each.
  MFPT  Machinery Failure Prevention Technology bearing set: baseline at
        97.656 kHz, inner/outer race faults at 48.828 kHz under varied load.
  SEU   Southeast University DDS gearbox rig at 5.12 kHz: bearing and gear
        fault sets under two speed/load conditions. Nyquist is 2.56 kHz, so
        any fixed multi-kHz demodulation band is structurally inapplicable -
        this dataset exists in the eval to keep band selection honest.

Every loader yields the same record shape:
  dict(dataset, name, state, group, fs, fr_nominal, x, lines)
where lines maps line names to ORDERS (multiples of shaft speed), and group
identifies one physical machine state (files that may share a codec GOP).
"""
from __future__ import annotations
import os, re
import numpy as np
import scipy.io as sio

CWRU_ROOT = os.environ.get('CWRU_ROOT', '/workspace/s-whynot/cwru-dataset')
MFPT_ROOT = os.environ.get(
    'MFPT_ROOT', '/workspace/mathworks/rollingelementbearingfaultdiagnosis-data')
SEU_ROOT = os.environ.get(
    'SEU_ROOT', '/workspace/cathysiyu/mechanical-datasets/gearbox')

# drive-end bearing SKF6205, orders of shaft speed
CWRU_LINES = dict(BPFO=3.5848, BPFI=5.4152, BSF=4.7135, FTF=0.3983)
CWRU_NOMINAL_RPM = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1730.0}

def cwru_records(win_check_s=2.0):
    """Walk the mirror's 12k drive-end tree plus normals."""
    out = []
    norm_dir = os.path.join(CWRU_ROOT, 'Normal')
    for fn in sorted(os.listdir(norm_dir)):
        m = re.match(r'(\d+)_Normal_(\d)\.mat', fn)
        if not m: continue
        out.append(('normal', int(m.group(2)), os.path.join(norm_dir, fn)))
    root12 = os.path.join(CWRU_ROOT, '12k_Drive_End_Bearing_Fault_Data')
    for dirpath, _, files in os.walk(root12):
        for fn in sorted(files):
            if not fn.endswith('.mat'): continue
            rel = os.path.relpath(dirpath, root12).split(os.sep)
            fault = rel[0]                       # B / IR / OR
            size = rel[1]
            pos = rel[2] if len(rel) > 2 else ''  # @3/@6/@12 for OR
            m = re.match(r'(\d+)(?:@\d+)?_(\d)\.mat', fn)
            if not m: continue
            state = f'{fault}{size}' + (pos.replace("@", "_") if pos else '')
            out.append((state, int(m.group(2)), os.path.join(dirpath, fn)))
    recs = []
    for state, load, path in out:
        m = sio.loadmat(path)
        de = [k for k in m if k.endswith('_DE_time')]
        if not de: continue
        x = m[de[0]].ravel().astype(np.float64)
        rpmk = [k for k in m if k.endswith('RPM')]
        rpm = float(m[rpmk[0]].ravel()[0]) if rpmk else CWRU_NOMINAL_RPM[load]
        fid = os.path.basename(path).split('_')[0].split('@')[0]
        recs.append(dict(dataset='cwru', name=fid, state=state,
                         group=f'{state}_L{load}', fs=12000.0,
                         fr_nominal=rpm/60.0, x=x, lines=dict(CWRU_LINES)))
    return recs

def mfpt_records():
    """MFPT via the MathWorks packaging: bearing struct with sr/gs/load/rate,
    fault frequencies included in the file (Hz at 25 Hz shaft)."""
    recs = []
    for sub in ('train_data', 'test_data'):
        d = os.path.join(MFPT_ROOT, sub)
        if not os.path.isdir(d): continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith('.mat'): continue
            m = sio.loadmat(os.path.join(d, fn))
            b = m['bearing']
            sr = float(b['sr'][0, 0].ravel()[0])
            x = b['gs'][0, 0].ravel().astype(np.float64)
            rate = float(b['rate'][0, 0].ravel()[0])
            load = float(b['load'][0, 0].ravel()[0])
            state = ('normal' if 'baseline' in fn else
                     'OR' if 'Outer' in fn else 'IR')
            lines = {k: float(m[k].ravel()[0])/max(rate, 1e-9)
                     for k in ('BPFO', 'BPFI', 'BSF', 'FTF') if k in m}
            recs.append(dict(dataset='mfpt', name=fn[:-4], state=state,
                             group=f'{state}_{fn[:-4]}', fs=sr,
                             fr_nominal=rate, x=x, lines=lines,
                             load_lbs=load))
    return recs

_SEU_SPEED = {'20': 20.0, '30': 30.0}

def seu_records(which='bearingset', channel=3, max_rows=400_000):
    """SEU DDS gearbox CSVs. fs = 5120 Hz (frequency limit 2000, x2.56).
    Filenames encode condition: <state>_<speedHz>_<loadV>.csv."""
    import pandas as pd
    d = os.path.join(SEU_ROOT, which)
    recs = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith('.csv'): continue
        m = re.match(r'([A-Za-z]+)_(\d+)_(\d+)\.csv', fn)
        if not m: continue
        state, spd, load = m.group(1).lower(), m.group(2), m.group(3)
        path = os.path.join(d, fn)
        # Header length and separator vary BETWEEN FILES of this dataset
        # (ball_20_0 is comma-separated, the rest tab): sniff, don't assume.
        with open(path, 'r', errors='replace') as fh:
            skip, sep = 16, '\t'
            for i in range(40):
                ln = fh.readline()
                if ln.strip().split(sep='\t')[0].strip(',') == 'Data':
                    skip = i+1
                    probe = fh.readline()
                    sep = ',' if probe.count(',') > probe.count('\t') else '\t'
                    break
        df = pd.read_csv(path, sep=sep, skiprows=skip,
                         header=None, usecols=[channel], nrows=max_rows,
                         dtype=np.float64, engine='c')
        x = df.iloc[:, 0].to_numpy()
        recs.append(dict(dataset=f'seu_{which}', name=fn[:-4], state=state,
                         group=f'{state}_{spd}_{load}', fs=5120.0,
                         fr_nominal=_SEU_SPEED.get(spd, 20.0), x=x,
                         lines={}))
    return recs

def nperseg_for(fs, base_fs=12000.0, base=4096):
    """Keep Welch resolution constant in HERTZ across rigs, so order-domain
    bin occupancy is comparable: 2.93 Hz at every sample rate."""
    n = base*fs/base_fs
    return 1 << int(round(np.log2(n)))
