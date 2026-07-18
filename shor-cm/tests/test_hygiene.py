"""T58/T59: input hygiene + scale invariance (audit F1/F2)."""
import numpy as np
import pytest
from shorcm import hygiene as HY
from shorcm.cascade import Cascade
from shorcm import simforge_v2 as V2


def test_T58_hygiene_repairs_flags_and_raises():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(50000) + 2.5          # DC offset
    x[1000:1010] = np.nan                          # short gap: repaired
    xc, flags = HY.validate_signal(x, 16384.0)
    assert np.isfinite(xc).all()
    assert abs(xc.mean()) < 1e-9
    assert flags.get("nan_frac", 0) < 0.001
    # dropout run flagged
    x2 = rng.standard_normal(50000)
    x2[5000:5600] = 0.3712                         # frozen DAQ
    _, f2 = HY.validate_signal(x2, 16384.0)
    assert "dropout_run" in f2 and f2["dropout_run"] >= 590
    # unusable records raise
    with pytest.raises(HY.SignalHygieneError):
        HY.validate_signal(np.full(50000, np.nan), 16384.0)
    with pytest.raises(HY.SignalHygieneError):
        HY.validate_signal(np.zeros(50000), 16384.0)
    with pytest.raises(HY.SignalHygieneError):
        HY.validate_signal(rng.standard_normal(1000), 16384.0)


def test_T59_scale_invariance_of_diagnosis():
    """x and 137x must produce the SAME fault call, severity, and
    speed through the deployment API — unknown field gains cannot
    move a model input."""
    casc = Cascade.load("models")
    m, x = V2.sample_run(4242)
    a = casc.analyze_record(x, V2.FS, V2.kinematic_sheet(m))
    b = casc.analyze_record(137.0 * x, V2.FS, V2.kinematic_sheet(m))
    assert a.get("fault_ml") == b.get("fault_ml")
    assert a.get("fault_rules") == b.get("fault_rules")
    if "severity_score" in a:
        assert abs(a["severity_score"] - b["severity_score"]) < 1e-6
    assert abs(a["speed_hz"] / b["speed_hz"] - 1) < 1e-6
    assert abs(b["signal_rms"] / a["signal_rms"] - 137.0) < 0.01
