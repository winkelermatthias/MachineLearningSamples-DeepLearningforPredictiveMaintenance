"""Anti-leakage splits. ALWAYS group by speed band: MAFAULDA records the
same machine state at many speeds, and random row splits leak trivially
(nearby-speed twins land in train and test)."""
import numpy as np
from sklearn.model_selection import GroupKFold

BAND_HZ = 5.0


def speed_band(rate_hz):
    return (np.asarray(rate_hz) // BAND_HZ).astype(int)


def grouped_folds(df, n_splits=5, seed=20260709):
    """Standard evaluation: GroupKFold over speed bands."""
    g = speed_band(df["rate_hz"].values)
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(df, df["y"], groups=g))


def extreme_speed_split(df, lo=20.0, hi=50.0):
    """Speed-invariance test: train mid speeds, test extremes."""
    r = df["rate_hz"].values
    train = np.flatnonzero((r >= lo) & (r <= hi))
    test = np.flatnonzero((r < lo) | (r > hi))
    return train, test
