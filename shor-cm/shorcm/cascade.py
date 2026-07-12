"""The deployable cascade: one API from waveform(s) to diagnosis, and a
per-machine monitor that tracks pattern energies over time.

This is the assembly of the whole synthetic program — peak-Shor speed
with kinematic sheet + octave arbitration, calibrated confidence with
abstain, Hz+order pattern ledgers, rules + frozen-ML fault arms, axial
subtype evidence, severity drivers, and growth tracking — behind two
entry points, so applying it to MAFAULDA or Relos data later is a
one-liner through `adapters`:

    from shorcm.cascade import Cascade
    casc = Cascade.load("models/")            # frozen artifacts
    out  = casc.analyze_record(X, fs, sheet)  # X: (n,) or (n, ch)

    mon = casc.monitor(sheet, f_ref=None)     # one per asset
    verdict = mon.feed(t, X, fs)              # streaming records

Input contract (the whole real-data adapter surface):
  X      float ndarray, shape (n,) single channel or (n, ch) with
         channel 0 = radial DE, 1 = axial, 2 = radial NDE (extra
         channels ignored; missing channels degrade gracefully)
  fs     sample rate in Hz
  sheet  kinematic sheet dict (see simforge_v2.kinematic_sheet); {} if
         nothing is known about the asset
"""
import json
from pathlib import Path

import numpy as np

from . import peakshor as PS
from . import simforge_corpus as SC
from . import tracker as TK

FAULTS6 = ["healthy", "imbalance", "misalignment", "looseness",
           "bearing", "gear"]


class Cascade:
    def __init__(self, fault_model=None, calibrator=None, meta=None,
                 severity_model=None):
        self.fault_model = fault_model
        self.calibrator = calibrator
        self.severity_model = severity_model
        self.meta = meta or {}

    # ---------- persistence ----------
    @classmethod
    def load(cls, model_dir):
        import joblib
        d = Path(model_dir)
        fm = joblib.load(d / "fault_lgbm.joblib") \
            if (d / "fault_lgbm.joblib").exists() else None
        cal = joblib.load(d / "conf_isotonic.joblib") \
            if (d / "conf_isotonic.joblib").exists() else None
        sev = joblib.load(d / "severity_lgbm.joblib") \
            if (d / "severity_lgbm.joblib").exists() else None
        meta = json.loads((d / "meta.json").read_text()) \
            if (d / "meta.json").exists() else {}
        return cls(fm, cal, meta, severity_model=sev)

    def save(self, model_dir):
        import joblib
        d = Path(model_dir)
        d.mkdir(parents=True, exist_ok=True)
        if self.fault_model is not None:
            joblib.dump(self.fault_model, d / "fault_lgbm.joblib")
        if self.calibrator is not None:
            joblib.dump(self.calibrator, d / "conf_isotonic.joblib")
        if self.severity_model is not None:
            joblib.dump(self.severity_model, d / "severity_lgbm.joblib")
        (d / "meta.json").write_text(json.dumps(self.meta, indent=1))

    # ---------- single record ----------
    def analyze_record(self, X, fs, sheet=None, meta=None):
        sheet = sheet or {}
        X = np.asarray(X, float)
        x0 = X if X.ndim == 1 else X[:, 0]
        # field-data contract (audit F1/F2): hygiene first — dropouts
        # and NaNs must never reach the envelope band selector — then
        # unit-RMS normalization so every downstream feature is
        # SCALE-INVARIANT (unknown gains and unit systems cannot move
        # a model input); absolute level preserved in signal_rms.
        from . import hygiene as HY
        raw_rms = float(np.sqrt(np.mean(
            np.square(x0[np.isfinite(x0)])))) if len(x0) else 0.0
        x0, in_flags = HY.validate_signal(x0, fs)
        rms = float(np.sqrt(np.mean(x0 ** 2)))
        x0 = x0 / rms
        est = PS.estimate_speed_sheet(x0, fs, sheet, meta=meta)
        est = [c for c in est if np.isfinite(c["hz"])] or \
            [{"hz": float("nan"), "confidence": 0.0, "score": -9,
              "ev": {}}]
        s = [c.get("score", -9.0) for c in est] + [-99.0]
        margin = s[0] - s[1]
        p_ok = float(self.calibrator.predict([margin])[0]) \
            if self.calibrator is not None else est[0]["confidence"]
        # degenerate-record guard (audit S-bug): a drowned or single-
        # candidate record produces margin vs the -99 sentinel, which
        # the isotonic calibrator clamps to its MAXIMUM -> confidence
        # ~1.0 on garbage. Fewer than 2 real candidates or a non-finite
        # top speed means the calibrated margin is meaningless.
        if len(est) < 2 or not np.isfinite(est[0]["hz"]) \
                or est[0]["hz"] <= 0:
            p_ok = 0.0
        out = {"speed_candidates": est,
               "speed_hz": est[0]["hz"],
               "speed_confidence": round(p_ok, 3),
               "abstain_speed": bool(p_ok < 0.5),
               "signal_rms": round(rms, 6),
               "input_flags": in_flags}
        f_hat = est[0]["hz"]
        if not np.isfinite(f_hat) or f_hat <= 0:
            out["fault"] = "unknown"
            return out
        pf, pa, pc = PS.spectral_peaks(x0, fs)
        spec = PS.spectrum(x0, fs)
        out["patterns_hz"] = PS.pattern_ledger_peaks(pf, pa, pc, f_hat,
                                                     spec=spec)
        led = SC.ledger(x0, fs, f_hat, spr=256, uns_hi=16.0)
        out["ledger"] = led
        # envelope evidence surface: impulsive bearing faults live in
        # the HF-resonance envelope, not the raw order spectrum (the
        # MFPT transfer gate measured raw attribution 0.0 there)
        from . import patterns as PT
        pats_env = None
        try:
            pats_env, _, band = PT.decompose_envelope(x0, fs, f_hat)
            out["patterns_envelope"] = [
                {"type": p["type"], "params": p["params"],
                 "share": round(p["share"], 4)}
                for p in pats_env
                if p["type"] != "FLOOR" and p["share"] > 0.01]
            out["envelope_band_hz"] = [round(b) for b in band]
        except Exception:
            out["patterns_envelope"] = []
        out["fault_rules"] = SC.rules_from_ledger(led)
        if self.fault_model is not None and led is not None:
            feats = [led.get(f, 0.0) for f in SC.LEDGER_FEATURES_V2]
            if self.meta.get("fault_model", {}).get("features_pf"):
                try:
                    pats, _ = PT.decompose(x0, fs, f_hat,
                                           sheet=sheet or None)
                    pfd = PT.pattern_features(pats)
                except Exception:
                    pfd = dict.fromkeys(PT.PF_COLS, 0.0)
                feats = feats + [pfd[c] for c in PT.PF_COLS]
            if self.meta.get("fault_model", {}).get("features_pfe"):
                pfe = PT.pattern_features_env(pats_env or [])
                feats = feats + [pfe[c] for c in PT.PF_ENV_COLS]
            proba = self.fault_model.predict_proba([feats])[0]
            i = int(np.argmax(proba))
            ps = np.sort(proba)
            out["fault_ml"] = FAULTS6[i]
            out["fault_ml_margin"] = round(float(ps[-1] - ps[-2]), 3)
            out["abstain_fault"] = bool(ps[-1] - ps[-2] < 0.1)
            if self.severity_model is not None:
                out["severity_score"] = round(float(np.clip(
                    self.severity_model.predict([feats])[0], 0.0, 1.0)),
                    3)
        # multi-channel evidence
        if X.ndim == 2 and X.shape[1] >= 2:
            from . import simforge_mc as MC
            af = MC.axial_features(X, fs, f_hat)
            out["axial"] = af
            if out.get("fault_ml", out["fault_rules"]) == "misalignment" \
                    or out["fault_rules"] == "misalignment":
                out["misalignment_subtype"] = (
                    "angular" if af["ax_ratio_2"] > 0.61 else
                    "parallel_or_coupling")
        # severity drivers (interpretation left to the O4 stage)
        if led is not None:
            out["severity_drivers"] = {
                "imbalance_a1": led["a1"],
                "misalignment_a2_over_a1": led["a2_over_a1"],
                "looseness_e_half": led["e_half"],
                "bearing_uns_abs": led["uns_abs"],
                "gear_gmf_sb_energy": led["gmf_sb_energy"]}
        return out

    # ---------- streaming monitor ----------
    def monitor(self, sheet=None, f_ref=None):
        return MachineMonitor(self, sheet or {}, f_ref)


class MachineMonitor:
    """One per asset. Feed records over time; get diagnosis + growth
    alarms. Uses the FrameSelector for a consistent speed frame and the
    PatternTracker for per-invariant energy series."""

    def __init__(self, cascade, sheet, f_ref=None):
        self.cascade = cascade
        self.sheet = sheet
        self.sel = TK.FrameSelector(warmup=5)
        self.tracker = TK.PatternTracker(f_ref=f_ref)
        from . import patterns as PT
        self.gen = PT.GeneralTracker()

    def feed(self, t, X, fs):
        from . import patterns as PT
        X = np.asarray(X, float)
        x0 = X if X.ndim == 1 else X[:, 0]
        rec = self.cascade.analyze_record(X, fs, self.sheet)
        pf, pa, pc = PS.spectral_peaks(x0, fs)
        f_lock = self.sel.observe(rec["speed_candidates"], pf, pa)
        if not np.isfinite(f_lock) or f_lock <= 0:
            f_lock = rec["speed_hz"]
        en = TK.pattern_energies(x0, fs, f_lock) \
            if np.isfinite(f_lock) and f_lock > 0 else None
        self.tracker.update(t, en, f_lock)
        rec["speed_locked_hz"] = f_lock
        rec["alarms"] = {k: d for k, d in
                         self.tracker.trends(adaptive=True).items()
                         if d["alarm"]}
        # general-pattern layer: isolating decomposition tracked by
        # per-type invariant identities; GROUP alarms treat a bearing
        # tone and its modulation fan as one physical source
        rec["patterns"] = []
        if np.isfinite(f_lock) and f_lock > 0:
            try:
                pats, _ = PT.decompose(x0, fs, f_lock,
                                       sheet=self.sheet or None)
                self.gen.update(t, pats)
                rec["patterns"] = [
                    {"type": p["type"], "params": p["params"],
                     "share": round(p["share"], 4)}
                    for p in pats if p["share"] > 0.005]
            except Exception:
                pass
        rec["pattern_alarms"] = [g for g in self.gen.trends_grouped()
                                 if g["alarm"]]
        return rec
