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
