#!/usr/bin/env python3
"""Corpus-scale judgment of the general pattern decomposer + tracker.

 A. Attribution on N_ATTR v2 machines (true speed, isolating the
    pattern layer): per generator-truth family, energy captured by
    type-compatible patterns near the truth orders — median |dB| error,
    residual share on faulted runs, share-sum identity.
 B. Tracking on N_TRK machines x 14 records, speed +/-20%: identity
    persistence of the fault-relevant pattern instance, growth alarms
    (right instance), false alarms on stationary/healthy.

ENV: N_ATTR (600), N_TRK (60), WORKERS (4).
"""
import json, os, sys, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import simforge_v2 as V2                 # noqa: E402
from shorcm import patterns as PT                    # noqa: E402

N_ATTR = int(os.environ.get("N_ATTR", 600))
N_TRK = int(os.environ.get("N_TRK", 60))
WORKERS = int(os.environ.get("WORKERS", 4))
OUT = Path("experiments/patterns")

# v2 amendment (post-hoc, recorded): (a) bearing truth INCLUDES its
# modulation sidebands, and an off-integer-carrier SIDEBAND fan is a
# correct isolation of a modulated bearing tone — credit it; (b) the
# new TONE type is the correct home for lone vane-pass/second-shaft/
# drive lines. v1 findings preserved in findings_pre_fix.json.
JUDGMENT_VERSION = 2


def _off_int_carrier(p):
    c = p["params"]["carrier"]
    return abs(c - round(c)) > 0.15


COMPAT = {"SHAFT": lambda p: p["type"] == "HARM"
          and abs(p["params"]["base"] - 1.0) < 0.03,
          "HALF": lambda p: p["type"] == "HALFHARM",
          "BEARING": lambda p: p["type"] == "NEARRAT"
          or (p["type"] == "SIDEBAND" and _off_int_carrier(p)),
          "GMF": lambda p: p["type"] in ("SIDEBAND", "HARM", "BAND",
                                         "TONE"),
          "HUM": lambda p: p["type"] in ("FIXEDHZ", "TONE"),
          "ELEC": lambda p: p["type"] in ("FIXEDHZ", "HARM", "TONE"),
          "NEIGHBOR": lambda p: p["type"] in ("FIXEDHZ", "HARM", "TONE"),
          "SHAFT2": lambda p: p["type"] in ("HARM", "NEARRAT", "FIXEDHZ",
                                            "TONE"),
          "PASSAGE": lambda p: p["type"] in ("HARM", "SIDEBAND", "TONE")}


def attr_one(i):
    truth = []
    m, x = V2.sample_run(i + 60_000_000 % 10**9, truth=truth)
    f0 = m["f_shaft"]
    try:
        pats, e_tot = PT.decompose(x, V2.FS, f0)
    except Exception:
        return None
    rows = []
    share_sum = sum(p["share"] for p in pats)
    floor_share = next(p["share"] for p in pats if p["type"] == "FLOOR")
    for tf in truth:
        fam = tf["family"]
        if fam not in COMPAT or fam == "ELEC_HF":
            continue
        freqs = [f for f in tf["freqs_hz"] if 2.0 < f]
        orders = [f / f0 for f in freqs if f / f0 < 128]
        if not orders:
            continue
        e_true = sum(a ** 2 / 2 for f, a in zip(tf["freqs_hz"],
                                                tf["amps"])
                     if 2.0 < f and f / f0 < 128)
        if e_true < 1e-4:
            continue
        ok = COMPAT[fam]
        e_got = 0.0
        for p in pats:
            if not ok(p):
                continue
            for mo, me in p["members"]:
                if any(abs(mo / o - 1) < 0.06 for o in orders):
                    e_got += me
        rows.append({"run_id": i, "fault": m["fault"],
                     "archetype": m["archetype"], "family": fam,
                     "e_true": e_true, "e_got": e_got,
                     "db_err": 10 * np.log10((e_got + 1e-12)
                                             / (e_true + 1e-12)),
                     "share_sum": share_sum,
                     "floor_share": floor_share})
    return rows


def trk_one(i):
    rng0 = V2.rng_for_run((61_000_000, i))
    m = V2.sample_machine(rng0)
    u = rng0.random()
    if u < 0.4 or m["fault"] == "healthy":
        m["fault"], m["severity"], scn = "healthy", 0.0, "healthy"
    elif u < 0.65:
        scn = "stationary"
    else:
        scn = "growing"
    want = {"imbalance": "HARM", "misalignment": "HARM",
            "looseness": "HALFHARM", "bearing": "NEARRAT",
            "gear": "SIDEBAND"}.get(m["fault"])
    f_base = m["f_shaft"]
    tr = PT.GeneralTracker()
    for t in range(14):
        rng = V2.rng_for_run((62_000_000 + i, t))
        sc = float(np.clip(1 + 0.18 * np.sin(1.7 * t + i)
                           + rng.normal(0, 0.03), 0.75, 1.25)) \
            if m["population"] == "vfd" else float(1 + rng.normal(0, .004))
        mm = dict(m)
        if scn == "growing":
            uu = max(t - 3, 0) / 10
            mm["severity"] = float(np.clip(0.05 + 0.85 * uu, 0.03, 1.0))
        elif scn == "stationary":
            mm["severity"] = float(np.clip(
                m["severity"] * (1 + 0.1 * rng.standard_normal()),
                0.05, 1.0))
        mm["f_shaft"] = f_base * sc
        if m["population"] == "vfd":
            mm["f_e"] = mm["f_shaft"] * mm["pole_pairs"] / (1 - mm["slip"])
        for k in ("f2", "f3", "fc", "gmf", "gmf2"):
            if k in mm:
                mm[k] = mm[k] * sc
        x = V2.synth_run(mm, rng, omega1x=sc ** 2)
        try:
            pats, _ = PT.decompose(x, V2.FS, mm["f_shaft"])
        except Exception:
            continue
        tr.update(t, pats)
    tds = tr.trends()
    persist = 0.0
    alarm_right = False
    if want:
        insts = [r for r in tr.reg if r["type"] == want]
        if insts:
            persist = max(len(r["pts"]) for r in insts) / 14
        alarm_right = any(tds.get(str(r["id"]), {}).get("alarm", False)
                          for r in insts)
    any_alarm = any(d["alarm"] for d in tds.values())
    return {"machine": i, "scenario": scn, "fault": m["fault"],
            "persistence": persist, "alarm_right_type": alarm_right,
            "any_alarm": any_alarm,
            "n_instances": len(tr.reg)}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(WORKERS, maxtasksperchild=50) as pool:
        attr = [r for rr in pool.imap_unordered(
            attr_one, range(N_ATTR), chunksize=4) if rr for r in rr]
        trk = list(pool.imap_unordered(trk_one, range(N_TRK),
                                       chunksize=1))
    da = pd.DataFrame(attr)
    dt = pd.DataFrame(trk)
    da.to_parquet(OUT / "attribution.parquet", index=False)
    dt.to_parquet(OUT / "tracking.parquet", index=False)
    find = {"judgment_version": JUDGMENT_VERSION,
            "n_attr_rows": len(da), "n_track": len(dt)}
    find["attribution_median_absdB_by_family"] = {
        f: round(float(g.db_err.abs().median()), 2)
        for f, g in da.groupby("family")}
    find["attribution_recall_within_3dB"] = {
        f: round(float((g.db_err.abs() < 3).mean()), 3)
        for f, g in da.groupby("family")}
    find["share_sum_max_dev"] = round(float(
        (da.share_sum - 1).abs().max()), 6)
    find["floor_share_median_faulted"] = round(float(
        da[da.fault != "healthy"].floor_share.median()), 3)
    grow = dt[dt.scenario == "growing"]
    other = dt[dt.scenario != "growing"]
    find["tracking"] = {
        "persistence_median": round(float(
            dt[dt.fault != "healthy"].persistence.median()), 3),
        "growth_alarm_right_type": round(float(
            grow.alarm_right_type.mean()), 3),
        "false_alarm_machines": round(float(other.any_alarm.mean()), 3),
        "n_growing": len(grow)}
    find["wall_s"] = round(time.time() - t0, 1)
    (OUT / "findings.json").write_text(json.dumps(find, indent=1))
    print(json.dumps(find, indent=1))


if __name__ == "__main__":
    main()
