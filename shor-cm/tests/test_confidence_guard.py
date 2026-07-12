"""Pin for the audit S-bug: degenerate records must never carry high
speed confidence (the isotonic calibrator clamps a margin against the
-99 sentinel to its MAXIMUM)."""
import numpy as np
from shorcm.cascade import Cascade


def test_degenerate_record_confidence_floor():
    casc = Cascade.load("models")
    rng = np.random.default_rng(5)
    x = 0.02 * rng.standard_normal(4096)      # drowned: no structure
    out = casc.analyze_record(x, 16384.0, {})
    if len([c for c in out["speed_candidates"]
            if np.isfinite(c["hz"])]) < 2:
        assert out["speed_confidence"] <= 0.5, out
        assert out["abstain_speed"], out
