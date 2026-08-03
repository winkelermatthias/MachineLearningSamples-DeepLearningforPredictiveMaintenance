"""Rejection sampling: parse teacher/student answers and verify them against
the simulator's ground truth. Only fully-correct traces enter training."""
import json
import os
import re

_JSON_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def parse_answer(text):
    m = _JSON_RE.findall(text or "")
    raw = m[-1] if m else None
    if raw is None:                                   # fall back: last {...} blob
        b = re.findall(r"\{[^{}]*\}", text or "", re.DOTALL)
        raw = b[-1] if b else None
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def check(sample, ans):
    out = dict(parsed=ans is not None, speed=False, fault=False, zone=False,
               cond=False, brg=True, accepted=False)
    if ans is None:
        out["brg"] = False
        return out
    try:
        out["speed"] = abs(float(ans.get("inferred_speed_rpm", 0)) - sample["rpm"]) / sample["rpm"] <= 0.04
    except (TypeError, ValueError):
        pass
    out["fault"] = ans.get("fault_type") == sample["fault"]
    out["zone"] = ans.get("iso_zone") == sample["zone"]
    out["cond"] = ans.get("bearing_condition") == sample["cond"]
    if sample["blind"]:
        out["brg"] = str(ans.get("bearing_designation", "")).replace(" ", "").upper() == sample["machine"]["bearing"]
    out["accepted"] = all(v for k, v in out.items() if k != "accepted")
    return out


def collect(teachers, dataset, out_dir, n_eval, min_accepted=300):
    """Load all cached teacher answers, verify each against ground truth, and
    return (accepted traces, per-teacher stats DataFrame)."""
    import pandas as pd

    by_id = {s["id"]: s for s in dataset}
    accepted, rows = [], []
    for name in teachers:
        path = os.path.join(out_dir, f"teacher_{name}.jsonl")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"WARNING: no answers on disk for teacher '{name}' — all its calls failed; "
                  "check the key/model id and re-run the generation cell (finished calls are cached)")
            continue
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                if rec["id"] < n_eval:
                    continue   # never train on eval ids (guards stale caches from smaller-n_eval runs)
                s = by_id[rec["id"]]
                res = check(s, parse_answer(rec["text"]))
                rows.append({"teacher": name, **res})
                if res["accepted"]:
                    accepted.append({"sample": s, "text": rec["text"], "teacher": name})

    stats = pd.DataFrame(rows).groupby("teacher").mean(numeric_only=True).round(3) if rows else None
    if stats is not None:
        print(stats)
    print(f"\naccepted traces: {len(accepted)} "
          f"({len(accepted) / max(len(rows), 1):.0%} of {len(rows)} teacher answers)")
    if len(accepted) < min_accepted:
        print(f"\nWARNING: {len(accepted)} accepted traces is far too few to distill anything — "
              "the student will just memorize formatting. Raise n_teacher (>= 400 for a demo, "
              "1500+ for real transfer) and re-run the teacher cell; cached answers are kept.")
    return accepted, stats
