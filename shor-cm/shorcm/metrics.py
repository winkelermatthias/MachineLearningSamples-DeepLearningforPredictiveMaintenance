"""Success measurement for the autonomous research loop.

North star (NS): macro-F1 on the 7-way fault problem, best NO-TACHO
variant (onex or comb), extreme-speed hold-out, evaluated on the sealed
vault. This mirrors MachineDoctor deployment: accelerometer + estimated
speed, unseen speeds, unseen runs.

Loop rules implemented here:
- vault_mask: ~20% of runs sealed by run_id hash at catalog time. Dev
  iterations NEVER touch them. Vault evaluations are budgeted (max 3 per
  program) and counted in a file the adversary audits.
- promotion_gate: a candidate method is promoted over the incumbent only
  if the paired-bootstrap delta on the dev metric is positive at a
  confidence that TIGHTENS with the number of attempts logged in the
  ledger (alpha = 0.05 / attempts). This is the anti-forking-paths brake:
  the more the loop tries, the stronger the evidence required.
- ledger: EVERY attempt (promoted or killed) is appended, with config
  hash. Deleting ledger lines is a CRITICAL adversary finding.
"""
import hashlib, json, time
from pathlib import Path
import numpy as np

VAULT_FRAC = 0.20
VAULT_BUDGET = 3
LEDGER = Path("work/loop_ledger.jsonl")
VAULT_COUNTER = Path("work/vault_access.json")
SCOREBOARD = Path("work/scoreboard.jsonl")


def vault_mask(run_ids, frac=VAULT_FRAC):
    """Deterministic, stateless seal: hash of run_id. Same runs are sealed
    forever, on every machine, no matter when the code runs."""
    return np.array([
        int(hashlib.sha1(str(r).encode()).hexdigest(), 16) % 1000
        < frac * 1000 for r in run_ids])


def paired_bootstrap_delta(y, pred_a, pred_b, metric, n=2000, seed=20260709):
    """Distribution of metric(b) - metric(a) over paired resamples."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); pa = np.asarray(pred_a); pb = np.asarray(pred_b)
    idx = np.arange(len(y))
    d = []
    for _ in range(n):
        s = rng.choice(idx, len(idx))
        d.append(metric(y[s], pb[s]) - metric(y[s], pa[s]))
    return np.asarray(d)


def n_attempts():
    if not LEDGER.exists():
        return 0
    return sum(1 for _ in LEDGER.open())


def promotion_gate(deltas, guardrails_ok, attempts=None):
    """Promote iff the alpha-adjusted lower quantile of the paired delta
    is > 0 AND all guardrails hold. alpha = 0.05 / max(attempts, 1)."""
    a = attempts if attempts is not None else max(n_attempts(), 1)
    alpha = 0.05 / max(a, 1)
    lo = float(np.quantile(deltas, alpha))
    return {"promote": bool(lo > 0 and guardrails_ok),
            "delta_mean": float(np.mean(deltas)),
            "delta_lo_alpha_adj": lo,
            "alpha": alpha, "attempts": a,
            "guardrails_ok": bool(guardrails_ok)}


def log_attempt(name, config, verdict, metrics):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "name": name,
           "config_sha": hashlib.sha1(
               json.dumps(config, sort_keys=True).encode()).hexdigest()[:12],
           "verdict": verdict, "metrics": metrics}
    with LEDGER.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    with SCOREBOARD.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def request_vault_access(reason):
    """Budgeted. Returns True if granted; increments counter."""
    VAULT_COUNTER.parent.mkdir(parents=True, exist_ok=True)
    state = {"used": 0, "log": []}
    if VAULT_COUNTER.exists():
        state = json.load(VAULT_COUNTER.open())
    if state["used"] >= VAULT_BUDGET:
        return False
    state["used"] += 1
    state["log"].append({"ts": time.time(), "reason": reason})
    json.dump(state, VAULT_COUNTER.open("w"))
    return True


GUARDRAILS_DOC = """Guardrails (all must hold for promotion):
G-a  tacho-variant H1 grouped-CV AUC does not drop > 0.01 vs incumbent
G-b  adversary suite passes (shuffle, leakage, confound, self-ref mask)
G-c  feature extraction < 5 s CPU per 5 s recording (edge deployability)
G-d  every promoted method ships >=1 synthetic mechanism test in tests/
G-e  no regression on the pinned regression suite (past promoted methods)
"""
