"""Contract pins for the deployable surface (cascade, decomposer,
adapters).

Motivated by the v4 ledger finding `v4_feature_order_defect`: training
consumed le_* columns in parquet order while analyze_record fed
LEDGER_FEATURES_V2 order — same 24 features, different positions, and
LGBM is positional, so held-out accuracy collapsed while OOF looked
fine. These tests pin the train/serve feature contract so a
count/order skew can never ship silently again, plus the decomposer's
share-closure invariant (both raw and envelope modes — the
envelope-mode gate inversions are their own ledger mechanism) and the
real-data adapter contract.
"""
from pathlib import Path

import numpy as np
import pytest

from shorcm import patterns as PT
from shorcm import simforge_corpus as SC
from shorcm import simforge_v2 as V2
from shorcm.cascade import Cascade, FAULTS6

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MFPT_DIR = ROOT / "data" / "mfpt"

FS = 16384
DUR = 4.0

_have_models = (MODEL_DIR / "fault_lgbm.joblib").exists() \
    and (MODEL_DIR / "meta.json").exists()


@pytest.mark.skipif(not _have_models, reason="models/ not frozen here")
def test_CT1_frozen_model_feature_contract():
    """The frozen models' n_features_in_ must equal EXACTLY the length
    of the vector analyze_record builds (meta features + features_pf +
    features_pfe), and the meta lists must be the module constants in
    the SAME ORDER — the module constants ARE the serve-time order, so
    any drift between train manifest and serve build is a skew."""
    casc = Cascade.load(MODEL_DIR)
    fm = casc.meta.get("fault_model", {})
    assert fm.get("features"), "meta.json must carry the train manifest"
    # order-sensitive equality: positions, not just membership
    assert fm["features"] == list(SC.LEDGER_FEATURES_V2), \
        "ledger feature ORDER in meta.json differs from " \
        "SC.LEDGER_FEATURES_V2 (the serve-time order)"
    assert fm["features_pf"] == list(PT.PF_COLS), \
        "pattern-feature ORDER in meta.json differs from PT.PF_COLS"
    assert fm["features_pfe"] == list(PT.PF_ENV_COLS), \
        "envelope-feature ORDER in meta.json differs from PT.PF_ENV_COLS"
    n_expected = len(fm["features"]) + len(fm["features_pf"]) \
        + len(fm["features_pfe"])
    assert casc.fault_model.n_features_in_ == n_expected, \
        (casc.fault_model.n_features_in_, n_expected)
    if casc.severity_model is not None:
        assert casc.severity_model.n_features_in_ == n_expected, \
            (casc.severity_model.n_features_in_, n_expected)


@pytest.mark.skipif(not _have_models, reason="models/ not frozen here")
def test_CT2_analyze_record_end_to_end_output_contract():
    """analyze_record on a seeded SimForge run must actually reach the
    frozen model (a feature-count skew makes predict_proba raise) and
    honor the output contract. Correctness of the blind single-record
    diagnosis is NOT asserted — that is the corpus gate's job."""
    casc = Cascade.load(MODEL_DIR)
    m, x = V2.sample_run(12345)
    out = casc.analyze_record(x, V2.FS, V2.kinematic_sheet(m))
    assert "fault_ml" in out, \
        "frozen model never consulted — feature build failed"
    assert out["fault_ml"] in FAULTS6, out["fault_ml"]
    assert np.isfinite(out["fault_ml_margin"])
    assert 0.0 <= out["fault_ml_margin"] <= 1.0
    assert isinstance(out["abstain_fault"], bool)
    assert 0.0 <= out["severity_score"] <= 1.0
    assert np.isfinite(out["speed_hz"]) and out["speed_hz"] > 0
    assert 0.0 <= out["speed_confidence"] <= 1.0
    assert out["fault_rules"], "rules arm must always answer"
    for k in ("ledger", "patterns_hz", "severity_drivers"):
        assert k in out, k


def _mixed_signal(seed=42):
    rng = np.random.default_rng(seed)
    n = int(DUR * FS)
    t = np.arange(n) / FS
    f_inst = 25.0 * (1 + 0.005 * np.sin(2 * np.pi * 0.7 * t + 1.0))
    ph = 2 * np.pi * np.cumsum(f_inst) / FS
    x = rng.standard_normal(n) + 0.4 * np.cos(ph) \
        + 0.2 * np.cos(2 * ph + 1.0) \
        + 0.15 * np.cos(2 * np.pi * 60.0 * t + 0.3)
    return x


def test_CT3_share_closure_raw_and_envelope_modes():
    """shares + FLOOR sum to 1 with non-negative finite energies in
    BOTH mode='raw' and mode='envelope' (the three envelope gate
    inversions must not break the closure invariant), and through the
    decompose_envelope entry point."""
    x = _mixed_signal()

    def check(pats, tag):
        assert pats, tag
        floors = [p for p in pats if p["type"] == "FLOOR"]
        assert len(floors) == 1, (tag, [p["type"] for p in pats])
        for p in pats:
            assert np.isfinite(p["energy"]) and p["energy"] >= 0, (tag, p)
            assert np.isfinite(p["share"]) and p["share"] >= 0, (tag, p)
        s = sum(p["share"] for p in pats)
        assert abs(s - 1.0) < 1e-6, (tag, s)

    pats_raw, e_raw = PT.decompose(x, FS, 25.0, mode="raw")
    assert np.isfinite(e_raw) and e_raw > 0
    check(pats_raw, "raw")
    pats_env, e_env = PT.decompose(x, FS, 25.0, mode="envelope")
    assert np.isfinite(e_env) and e_env > 0
    check(pats_env, "envelope-mode")
    pats_ev, e_ev, band = PT.decompose_envelope(x, FS, 25.0)
    assert np.isfinite(e_ev)
    assert len(band) == 2 and band[0] < band[1] <= 0.5 * FS
    check(pats_ev, "decompose_envelope")


def test_CT4_pattern_feature_blocks_complete_and_finite():
    """pattern_features / pattern_features_env are the model input
    surface: they must return EXACTLY the PF_COLS / PF_ENV_COLS keys
    with finite values — a missing or NaN cell here is a silent
    serve-time poisoning of the frozen model."""
    x = _mixed_signal(seed=3)
    pats, _ = PT.decompose(x, FS, 25.0)
    pf = PT.pattern_features(pats)
    assert sorted(pf) == sorted(PT.PF_COLS), sorted(pf)
    assert all(np.isfinite(v) for v in pf.values()), pf
    pats_env, _, _ = PT.decompose_envelope(x, FS, 25.0)
    pfe = PT.pattern_features_env(pats_env)
    assert sorted(pfe) == sorted(PT.PF_ENV_COLS), sorted(pfe)
    assert all(np.isfinite(v) for v in pfe.values()), pfe
    # empty-input degradation: all-zero blocks, never KeyError/NaN
    pfe0 = PT.pattern_features_env([])
    assert sorted(pfe0) == sorted(PT.PF_ENV_COLS)
    assert all(v == 0.0 for v in pfe0.values()), pfe0


_mfpt_files = sorted(MFPT_DIR.glob("*.mat")) if MFPT_DIR.exists() else []


@pytest.mark.skipif(not _mfpt_files, reason="data/mfpt absent")
def test_CT5_mfpt_adapter_contract():
    """from_mfpt_mat must return a finite float 1-D array at one of
    the two published MFPT sample rates (97656 baseline / 48828 fault
    records), an empty sheet (direct rig), and a positive nameplate
    rate — the entire surface the cascade consumes."""
    from shorcm import adapters as AD
    for fp in _mfpt_files[:3]:
        x, fs, sheet, meta = AD.from_mfpt_mat(fp)
        assert isinstance(x, np.ndarray) and x.ndim == 1, (fp.name, x.shape)
        assert x.dtype == np.float64
        assert np.all(np.isfinite(x)), fp.name
        assert fs in (48828.0, 97656.0), (fp.name, fs)
        assert len(x) >= fs, (fp.name, len(x))     # at least 1 s
        assert sheet == {}, sheet
        assert np.isfinite(meta["rate_hz"]) and meta["rate_hz"] > 0, meta
