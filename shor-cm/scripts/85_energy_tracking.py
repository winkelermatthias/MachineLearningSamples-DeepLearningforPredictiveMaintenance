#!/usr/bin/env python3
"""Fleet pattern-energy tracking experiment on the v2 ensemble.

Each fleet machine is monitored for RECORDS records with record-to-record
operating-point speed changes (VFD +/-20%, mains slip-wiggle only) and a
severity scenario:
  stable_healthy   40%  no fault, nothing may alarm
  stationary_fault 25%  fault PRESENT at constant severity — an
                        indicator, not a growing fault: must NOT alarm
  growing_fault    35%  severity ramps 0.05 -> 0.6..1.0 from onset
                        record r0 (linear / exponential / late step)

Ground truth per record comes from the generator's composition truth
(injected per-family energies). Judged, separately (true speed) and
together (blind speed per record):
  1. per-sample energy fidelity (Spearman + median |dB| error, per family)
  2. growth classification (alarm on the fault key): AUC by rise_db,
     TP/FP at the alarm gate, fleet false-alarm rate
  3. detection delay in records from onset
  4. speed invariance: normalized SHAFT_1 energy vs speed on healthy VFD
     machines, with and without the omega^2 normalization

ENV: FLEET (200), RECORDS (16), WORKERS (4).
"""
import json, os, sys, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import simforge_v2 as V2                 # noqa: E402
from shorcm import peakshor as PS                    # noqa: E402
from shorcm import tracker as TK                     # noqa: E402

FLEET = int(os.environ.get("FLEET", 200))
RECORDS = int(os.environ.get("RECORDS", 16))
WORKERS = int(os.environ.get("WORKERS", 4))
OUT = Path("experiments/tracking")

TRUE_KEY = {"imbalance": ("SHAFT", 0), "misalignment": ("SHAFT", 1),
            "looseness": ("HALF", None), "bearing": ("BEARING", None),
            "gear": ("GMF", None)}


def true_energy(truth, fam, idx):
    e = 0.0
    for t in truth:
        if t["family"] != fam:
            continue
        amps = t["amps"] if idx is None else t["amps"][idx:idx + 1]
        e += sum(a ** 2 / 2 for a in amps)
    return e


def scenario_for(i, m, rng):
    u = rng.random()
    if u < 0.40 or m["fault"] == "healthy":
        m["fault"], m["subtype"], m["severity"] = "healthy", "", 0.0
        return "stable_healthy", None, None
    if u < 0.65:
        return "stationary_fault", None, float(rng.uniform(0.4, 0.8))
    shape = str(rng.choice(["linear", "exp", "step"]))
    r0 = int(rng.integers(2, 7))
    return "growing_fault", (shape, r0), None


def severity_at(scn, prof, sev_const, t, rng_noise):
    if scn == "stable_healthy":
        return 0.0
    if scn == "stationary_fault":
        return float(np.clip(sev_const * (1 + 0.10 * rng_noise), 0.05, 1.0))
    shape, r0 = prof
    if t < r0:
        s = 0.05
    else:
        u = (t - r0) / max(RECORDS - 1 - r0, 1)
        s1 = 0.6 + 0.4 * abs(np.sin(r0))          # deterministic in r0
        s = {"linear": 0.05 + (s1 - 0.05) * u,
             "exp": 0.05 * (s1 / 0.05) ** u,
             "step": 0.05 if u < 0.5 else s1}[shape]
    return float(np.clip(s * (1 + 0.08 * rng_noise), 0.03, 1.0))


def one_machine(i):
    rng0 = V2.rng_for_run(30_000_000 + i)
    m = V2.sample_machine(rng0)
    scn, prof, sev_const = scenario_for(i, m, rng0)
    f_base = m["f_shaft"]
    tr_t = TK.PatternTracker(f_ref=f_base)
    tr_e = TK.PatternTracker(f_ref=f_base)
    tr_l = TK.PatternTracker(f_ref=f_base)
    fkey = TK.FAULT_KEY.get(m["fault"])
    tfam = TRUE_KEY.get(m["fault"])
    rows = []
    deferred = []          # lock arm runs as a second pass: FrameSelector
    sel = TK.FrameSelector(warmup=5)
    for t in range(RECORDS):
        rng = V2.rng_for_run((40_000_000 + i, t))
        if m["population"] == "vfd":
            sc = float(np.clip(1 + 0.18 * np.sin(1.7 * t + i)
                               + rng.normal(0, 0.03), 0.75, 1.25))
        else:
            sc = float(1 + rng.normal(0, 0.004))
        mm = dict(m)
        mm["severity"] = severity_at(scn, prof, sev_const,
                                     t, rng.standard_normal())
        mm["f_shaft"] = f_base * sc
        if m["population"] == "vfd":
            mm["f_e"] = mm["f_shaft"] * mm["pole_pairs"] / (1 - mm["slip"])
        for k in ("f2", "f3", "fc", "gmf", "gmf2"):
            if k in mm:
                mm[k] = mm[k] * sc
        truth = []
        x = V2.synth_run(mm, rng, truth=truth, omega1x=sc ** 2)
        en_t = TK.pattern_energies(x, V2.FS, mm["f_shaft"])
        est = PS.estimate_speed_sheet(x, V2.FS, V2.kinematic_sheet(m),
                                      meta={"component": "motor"})
        f_hat = est[0]["hz"] if np.isfinite(est[0]["hz"]) else f_base
        en_e = TK.pattern_energies(x, V2.FS, f_hat)
        pfk, pak, _ = PS.spectral_peaks(x, V2.FS)
        deferred.append((t, x, est, pfk, pak, f_hat, en_e,
                         mm["f_shaft"]))
        tr_t.update(t, en_t, mm["f_shaft"])
        tr_e.update(t, en_e, f_hat)
        row = {"machine": i, "t": t, "scenario": scn, "fault": m["fault"],
               "archetype": m["archetype"], "population": m["population"],
               "sev": mm["severity"], "f_op": mm["f_shaft"],
               "speed_scale": sc,
               "spd_ok": abs(f_hat / mm["f_shaft"] - 1) <= 0.01}
        bt = next((tt for tt in truth if tt["family"] == "BEARING"), None)
        o_bt = (bt["freqs_hz"][0] / mm["f_shaft"]) if bt else None

        def pick(en):
            if en is None:
                return None
            if fkey == "NEARRAT":
                lst = en.get("NEARRAT_LIST", [])
                if o_bt and lst:
                    near = [(e, o) for e, o in lst
                            if abs(o / o_bt - 1) < 0.05]
                    return near[0][0] if near else 0.0
                return en["NEARRAT"][0]
            v = en[fkey]
            return v[0] if isinstance(v, tuple) else v

        if fkey and en_t is not None:
            row["e_tracked_true"] = pick(en_t)
        if fkey and en_e is not None:
            row["e_tracked_est"] = pick(en_e)
        if tfam:
            row["e_true"] = true_energy(truth, *tfam)
        if en_t is not None:                       # speed-invariance probe
            row["shaft1_norm"] = en_t["SHAFT_1"] / sc ** 4
            row["shaft1_raw"] = en_t["SHAFT_1"]
        rows.append(row)
    # ---- lock arm, second pass: FrameSelector over the whole sequence,
    # warmup records re-framed retroactively (no record-1 authority) ----
    frames = []
    for (t, x, est, pfk, pak, f_hat, en_e, f_true_t) in deferred:
        frames.append(sel.observe(est, pfk, pak))
    if sel.warmup_choices is not None:
        frames[:len(sel.warmup_choices)] = sel.warmup_choices
    for (t, x, est, pfk, pak, f_hat, en_e, f_true_t), f_lock in \
            zip(deferred, frames):
        en_l = en_e if abs(f_lock / f_hat - 1) < 1e-3 \
            else TK.pattern_energies(x, V2.FS, f_lock)
        tr_l.update(t, en_l, f_lock)
        rows[t]["spd_lock_ok"] = abs(f_lock / f_true_t - 1) <= 0.01
        if fkey and en_l is not None:
            if fkey == "NEARRAT":
                lst = en_l.get("NEARRAT_LIST", [])
                rows[t]["e_tracked_lock"] = lst[0][0] if lst else 0.0
            else:
                v = en_l[fkey]
                rows[t]["e_tracked_lock"] = v[0] if isinstance(v, tuple) \
                    else v
    res = {"machine": i, "scenario": scn, "fault": m["fault"],
           "archetype": m["archetype"], "population": m["population"],
           "onset": prof[1] if prof else np.nan}
    for arm, tr in (("true", tr_t), ("est", tr_e), ("lock", tr_l)):
        tds = tr.trends()
        tda = tr.trends(adaptive=True)
        alarms = {k: d for k, d in tds.items() if d["alarm"]}
        res[f"{arm}_any_alarm"] = bool(alarms)
        res[f"{arm}_any_alarm_adaptive"] = bool(
            any(d["alarm"] for d in tda.values()))
        res[f"{arm}_alarm_keys"] = ",".join(sorted(alarms))
        if fkey:
            if fkey == "NEARRAT":
                keys = [k for k in tds if k.startswith("NEARRAT#")] or ["_"]
                d = max((tds.get(k, {}) for k in keys),
                        key=lambda dd: dd.get("rise_db", -99),
                        default={})
                da = any(tda.get(k, {}).get("alarm", False) for k in keys)
                fas = [tr.first_alarm(k) for k in keys if k != "_"]
                fas = [v for v in fas if v is not None]
                fa = min(fas) if fas else None
            else:
                d = tds.get(fkey, {})
                da = tda.get(fkey, {}).get("alarm", False)
                fa = tr.first_alarm(fkey)
            res[f"{arm}_fault_alarm"] = bool(d.get("alarm", False))
            res[f"{arm}_fault_alarm_adaptive"] = bool(da)
            res[f"{arm}_fault_rise_db"] = float(d.get("rise_db", np.nan))
            res[f"{arm}_fault_slope"] = float(d.get("slope", np.nan))
            res[f"{arm}_first_alarm_t"] = fa if fa is not None else np.nan
    return rows, res


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    recs, machines = [], []
    with Pool(WORKERS, maxtasksperchild=20) as pool:
        for k, (rows, res) in enumerate(
                pool.imap_unordered(one_machine, range(FLEET), chunksize=2)):
            recs.extend(rows); machines.append(res)
            if (k + 1) % 20 == 0:
                el = time.time() - t0
                print(f"{k+1}/{FLEET} machines {el:.0f}s "
                      f"(eta {el/(k+1)*(FLEET-k-1):.0f}s)", flush=True)
    dr = pd.DataFrame(recs)
    dm = pd.DataFrame(machines)
    dr.to_parquet(OUT / "records.parquet", index=False)
    dm.to_parquet(OUT / "machines.parquet", index=False)
    print(f"wall {time.time()-t0:.0f}s; blind speed top1 on fleet: "
          f"{dr.spd_ok.mean():.3f}")

    from scipy.stats import spearmanr
    find = {"fleet": FLEET, "records": RECORDS,
            "scenarios": dm.scenario.value_counts().to_dict(),
            "blind_speed_top1": round(float(dr.spd_ok.mean()), 3),
            "locked_speed_top1": round(float(dr.spd_lock_ok.mean()), 3)}
    # 1 energy fidelity per faulted machine
    fid = {}
    for arm in ("true", "est", "lock"):
        rows = []
        for (mach, fam), g in dr.dropna(
                subset=[f"e_tracked_{arm}", "e_true"]).groupby(
                ["machine", "fault"]):
            if len(g) >= 8 and g.e_true.max() > 0:
                rho = spearmanr(g.e_true, g[f"e_tracked_{arm}"]).statistic
                dbe = np.median(np.abs(
                    10 * np.log10((g[f"e_tracked_{arm}"] + 1e-12)
                                  / (g.e_true + 1e-12))))
                rows.append((fam, rho, dbe))
        d = pd.DataFrame(rows, columns=["fault", "rho", "dbe"])
        fid[arm] = {f: {"median_spearman": round(float(g.rho.median()), 3),
                        "median_abs_dB_err": round(float(g.dbe.median()), 1),
                        "n": len(g)}
                    for f, g in d.groupby("fault")}
    find["energy_fidelity"] = fid
    # 2 growth detection
    det = {}
    for arm in ("true", "est", "lock"):
        grow = dm[dm.scenario == "growing_fault"]
        other = dm[dm.scenario != "growing_fault"]
        from sklearn.metrics import roc_auc_score
        sc = pd.concat([grow[f"{arm}_fault_rise_db"].fillna(-20),
                        dm[dm.scenario == "stationary_fault"]
                        [f"{arm}_fault_rise_db"].fillna(-20)])
        lb = np.r_[np.ones(len(grow)),
                   np.zeros((dm.scenario == "stationary_fault").sum())]
        auc = roc_auc_score(lb, sc) if len(set(lb)) > 1 else np.nan
        delays = (grow[f"{arm}_first_alarm_t"] - grow.onset).dropna()
        det[arm] = {
            "recall_growing": round(float(grow[f"{arm}_fault_alarm"]
                                          .mean()), 3),
            "recall_growing_adaptive": round(float(
                grow[f"{arm}_fault_alarm_adaptive"].mean()), 3),
            "auc_grow_vs_stationary": round(float(auc), 3),
            "false_alarm_rate_machines": round(float(
                other[f"{arm}_any_alarm"].mean()), 3),
            "false_alarm_rate_adaptive": round(float(
                other[f"{arm}_any_alarm_adaptive"].mean()), 3),
            "median_detection_delay_records": (round(float(
                delays.median()), 1) if len(delays) else None),
            "n_growing": len(grow)}
    find["growth_detection"] = det
    # 3 speed invariance (healthy VFD machines, SHAFT_1)
    h = dr[(dr.scenario == "stable_healthy") & (dr.population == "vfd")]
    if len(h) > 50:
        rho_n = spearmanr(h.speed_scale, h.shaft1_norm).statistic
        rho_r = spearmanr(h.speed_scale, h.shaft1_raw).statistic
        find["speed_invariance_shaft1"] = {
            "spearman_raw_vs_speed": round(float(rho_r), 3),
            "spearman_normalized_vs_speed": round(float(rho_n), 3)}
    find["wall_s"] = round(time.time() - t0, 1)
    (OUT / "findings.json").write_text(json.dumps(find, indent=1,
                                                  default=str))
    print(json.dumps(find, indent=1, default=str))


if __name__ == "__main__":
    main()
