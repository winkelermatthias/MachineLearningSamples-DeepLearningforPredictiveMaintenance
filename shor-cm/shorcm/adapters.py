"""Real-data adapters: everything the cascade needs from the outside
world is (X, fs, sheet). These functions produce exactly that.

The cascade itself never changes when the data source does — that is
the point of freezing the synthetic-trained artifacts first.
"""
import numpy as np

MAFAULDA_FS = 50_000
# MAFAULDA CSV column map (features.py CH): 0 tacho, 1..3 underhang
# ax/rad/tan, 4..6 overhang ax/rad/tan, 7 mic
MAFAULDA_CHANNELS = {"radial_de": 2, "axial": 1, "radial_nde": 5}


def from_mafaulda_csv(path, channels=MAFAULDA_CHANNELS):
    """(X, fs, sheet) from one MAFAULDA run CSV. The rig is a direct
    coupled shaft with two bearing housings; sheet carries no gearing.
    The tacho column is deliberately NOT exposed — the cascade is blind
    by contract; tacho is for judgment only."""
    import pandas as pd
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, [channels["radial_de"], channels["axial"],
                    channels["radial_nde"]]].to_numpy(float)
    sheet = {}                     # direct rig: no ratio/mesh/passage
    return X, MAFAULDA_FS, sheet


def mafaulda_truth_speed(path):
    """Judgment-side only: mean speed from the tacho channel."""
    import pandas as pd
    from . import tacho as T
    df = pd.read_csv(path, header=None)
    _, meta = T.phase_from_tacho(df.iloc[:, 0].to_numpy(float),
                                 MAFAULDA_FS)
    return float(meta["rate_hz"])


def from_relos_timeseries(samples, fs, equipment_meta=None):
    """(X, fs, sheet) from Relos historian samples (list/array of floats
    or (n, ch)). equipment_meta: the asset registry record; asset_class
    maps to a coarse kinematic sheet until the registry carries tooth
    counts."""
    X = np.asarray(samples, float)
    sheet = {}
    ac = (equipment_meta or {}).get("asset_class", "")
    if ac == "fan":
        sheet["passage"] = 8           # coarse prior; refine per asset
    elif ac == "pump":
        sheet["passage"] = 6
    elif ac == "blower":
        sheet["passage"] = 4
    return X, fs, sheet


# ---------------- MFPT (Machinery Failure Prevention Technology) ------
# Mirror: mathworks/RollingElementBearingFaultDiagnosis-Data (CC BY-NC-SA
# 4.0, owner Eric Bechhoefer). Single radial channel, NICE bearing on a
# 25 Hz input shaft. Published defect orders (contact angle ~0):
MFPT_ORDERS = {"BPFO": 3.245, "BPFI": 4.755, "FTF": 0.406, "BSF": 2.555}


def from_mfpt_mat(path):
    """(x, fs, sheet, meta) from one MFPT .mat record. rate/load are
    nameplate metadata (judgment side may use rate as truth)."""
    import scipy.io as sio
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    b = m["bearing"]
    x = np.asarray(b.gs, float).ravel()
    return x, float(b.sr), {}, {"rate_hz": float(b.rate),
                                "load_lbs": float(b.load)}


# ---------------- SEU gearbox (Drivetrain Dynamics Simulator) ---------
# Mirror: cathysiyu/Mechanical-datasets. 8 tab-separated channels after
# a ~16-line header; fs = 2.56 x 2000 Hz frequency limit = 5120 Hz.
# Channels: 0 motor vib, 1-3 planetary gearbox xyz, 4 motor torque,
# 5-7 parallel gearbox xyz. Motor speed 20 Hz (20_0) or 30 Hz (30_2).
SEU_FS = 5120.0


def from_seu_csv(path, channel=1, t0=0.0, dur=8.0):
    """(x, fs, sheet, meta) — one window of one SEU file. channel 1 =
    planetary gearbox x (the effective vibration signal per the source
    README). gearset files are tab-separated, bearingset files comma-
    separated; both carry a 'Data' marker line before the samples.
    NOTE: some files' internal Title disagrees with the filename (a
    known mirror quirk) — filenames are the canonical labels; the
    internal title is returned in meta. Kinematic sheet stays {} —
    tooth counts are not published with the mirror: the cascade runs
    blind."""
    import pandas as pd
    title, data_row = "", 16
    with open(path, "r", errors="ignore") as fh:
        for k in range(40):
            line = fh.readline()
            if k == 0:
                title = line.replace("Title:", "").strip(" ,\t\n")
            if line.startswith("Data"):
                data_row = k + 1
                break
        probe = fh.readline()
    sep = "\t" if probe.count("\t") >= probe.count(",") else ","
    i0 = int(t0 * SEU_FS)
    n = int(dur * SEU_FS)
    df = pd.read_csv(path, sep=sep, skiprows=data_row, header=None,
                     nrows=i0 + n, engine="c")
    col = pd.to_numeric(df.iloc[i0:i0 + n, channel], errors="coerce")
    x = col.to_numpy(float)
    x = x[np.isfinite(x)]
    f_nom = 30.0 if "_30_2" in str(path) else 20.0
    return x, SEU_FS, {}, {"rate_hz": f_nom, "title": title}
