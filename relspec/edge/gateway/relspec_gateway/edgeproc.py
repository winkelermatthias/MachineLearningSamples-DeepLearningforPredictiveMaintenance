"""Local (edge) feature extraction: instant readouts the plant can see
on OPC UA without any cloud round-trip. Deliberately cheap — the heavy
spectral pipeline runs in the cloud where its state lives."""
from __future__ import annotations
import numpy as np


def features(samples_i16: bytes, scale: float) -> dict:
    x = np.frombuffer(samples_i16, dtype='<i2').astype(np.float64) * scale
    if x.size == 0:
        return dict(rms_g=0.0, peak_g=0.0, crest=0.0, mean_g=0.0)
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    return dict(rms_g=rms, peak_g=peak,
                crest=peak / rms if rms > 1e-12 else 0.0,
                mean_g=float(x.mean()))
