"""Pattern-energy tracking over time (fundamental vib-analysis layer).

A pattern's identity across records is its ORDER-DOMAIN INVARIANT — the
Shor bookkeeping: integer/rational orders name shaft-locked families
regardless of operating speed, near-rational drifting orders name
bearing tones through their slip band, the mesh integer names the gear.
Frequencies move with speed; the invariant does not. Presence is an
indicator; GROWTH is the alarm:

  per record   pattern_energies()   energy per invariant key
  over time    PatternTracker       association + log-energy series
  decision     trends()             Theil-Sen slope + Mann-Kendall
                                    significance + >= 6 dB rise over the
                                    commissioning baseline

Speed-law normalization: mass-force (1x) energy scales ~ omega^2; the
tracker divides SHAFT_1 energy by (f/f_ref)^2 so an operating-point
ramp does not masquerade as fault growth.
"""
import numpy as np
from scipy.stats import theilslopes, kendalltau

from . import tacho as T
from . import spectra as S

INT_KEYS = 16          # SHAFT_1 .. SHAFT_16 tracked individually
ALARM_P = 0.01
ALARM_DB = 6.0         # energy rise gate over baseline (6 dB = 4x energy)


def pattern_energies(x, fs, f_hat, spr=256):
    """Per-invariant pattern energies for one record, under speed
    hypothesis f_hat. Returns dict: 'SHAFT_k' -> E, 'HALF' -> E,
    'NEARRAT' -> (E, order), 'GMF' -> (E, order). E = sum a^2/2 over
    members in the fine order spectrum."""
    try:
        ph, _ = T.phase_from_comb(x, fs, f_nom=f_hat, prior_rel_sigma=0.008)
        xa = S.angular_resample(x, ph, spr)
        A, o = S.fine_order_spectrum(xa, spr)
    except Exception:
        return None

    def aabs(oo):
        bi = int(np.argmin(np.abs(o - oo)))
        return float(A[max(bi - 1, 0):bi + 2].max())

    out = {}
    for k in range(1, INT_KEYS + 1):
        out[f"SHAFT_{k}"] = aabs(float(k)) ** 2 / 2
    out["HALF"] = float(sum(aabs(k) ** 2 / 2
                            for k in (0.5, 1.5, 2.5, 3.5)))
    # NEARRAT: strongest drifting/unsnapped cluster in the bearing band
    pk = S.peak_orders(A, o, 16.0, 25, guard=4.0)
    tol = max(S.snap_tol(max(int(len(xa) / spr), 10)), 0.004)
    clusters = []
    for oo, aa, _ in sorted(pk):
        if clusters and oo - clusters[-1][-1][0] < 0.008 * oo:
            clusters[-1].append((oo, aa))
        else:
            clusters.append([(oo, aa)])
    best = (0.0, 0.0)
    do = o[1] - o[0]
    floor_a = float(np.median(A))
    ENBW = 1.5                             # hann noise bandwidth in bins
    for cl in clusters:
        a_c = sum(a for _, a in cl)
        o_c = sum(o_ * a for o_, a in cl) / (a_c + 1e-15)
        if not (1.8 < o_c < 16.0):
            continue
        width = (cl[-1][0] - cl[0][0]) / o_c if len(cl) > 1 else 0.0
        fr = S.cf_snap(o_c, tol=tol, qmax=8)
        if fr is not None and fr.denominator >= 2 and fr.numerator > 10:
            fr = None
        drifting = width > 0.004 or len(cl) >= 3 or fr is None
        if drifting and abs(o_c - round(o_c)) > tol:
            # BAND energy, not peak tips: a phase-walking tone smears
            # its power across the cluster band. Integrate |A|^2 over
            # the band (+guard), subtract the floor's share, divide by
            # the window ENBW so a steady tone calibrates to a^2/2.
            i0 = max(int((cl[0][0] - 0.015 * o_c) / do), 0)
            i1 = min(int((cl[-1][0] + 0.015 * o_c) / do) + 1, len(A))
            band = A[i0:i1]
            e_c = float(max(np.sum(band ** 2) / 2
                            - len(band) * floor_a ** 2 / 2, 0.0)) / ENBW
            if e_c > best[0]:
                best = (e_c, o_c)
    out["NEARRAT"] = best
    # GMF: best-modulated high-order line + its sideband fan
    max_o = min(spr / 2 - 2.0, 130.0)
    gbest = (0.0, 0.0)
    if max_o > 12:
        floor = float(np.median(A)) + 1e-15
        pk_hi = sorted([(oo, aa) for oo, aa, _ in
                        S.peak_orders(A, o, max_o, 40, guard=5.0)
                        if oo > 11.0], key=lambda t: -t[1])[:5]
        for o_g, a_g in pk_hi:
            cnt_b, e_b = 0, 0.0
            for dlt in np.arange(0.15, 1.26, 0.02):
                cnt, e = 0, 0.0
                for k in (1, 2, 3):
                    for sgn in (-1, 1):
                        a_sb = aabs(o_g + sgn * k * dlt)
                        if a_sb > max(0.15 * a_g, 4 * floor):
                            cnt += 1
                            e += a_sb ** 2 / 2
                if cnt > cnt_b:
                    cnt_b, e_b = cnt, e
            e_tot = a_g ** 2 / 2 + e_b
            if (cnt_b >= 1 or a_g > 8 * floor) and e_tot > gbest[0]:
                gbest = (e_tot, o_g)
    out["GMF"] = gbest
    return out


class PatternTracker:
    """Associates per-record pattern energies into per-invariant time
    series and detects growth. f_ref: speed normalization reference."""

    def __init__(self, f_ref=None):
        self.f_ref = f_ref
        self.series = {}                 # key -> list of (t, log10 E)
        self.meta = {}                   # key -> last order (NEARRAT/GMF)

    def update(self, t, energies, f_hat):
        if energies is None:
            return
        if self.f_ref is None:
            self.f_ref = f_hat
        for k, v in energies.items():
            if k in ("NEARRAT", "GMF"):
                e, o_c = v
                if e <= 0:
                    continue
                # slip-tolerant association: same instance if the order
                # moved < 4% (NEARRAT) / < 1.5 orders (GMF)
                last = self.meta.get(k)
                if last is not None:
                    tolr = 0.04 if k == "NEARRAT" else 1.5 / max(last, 12)
                    if abs(o_c / last - 1) > tolr:
                        self.series.pop(k, None)   # new instance
                self.meta[k] = o_c
            else:
                e = v
            if k == "SHAFT_1":           # omega^2 mass-force law
                e = e / max(f_hat / self.f_ref, 1e-3) ** 2
            if e > 0:
                self.series.setdefault(k, []).append((t, np.log10(e)))

    def trends(self, min_n=6, adaptive=False):
        """adaptive=True: per-key gate from the commissioning scatter —
        max(3 dB, 3 x the key's own baseline std in dB) — instead of the
        global 6 dB. Keys with tight baselines alarm earlier; noisy keys
        pay a higher bar."""
        out = {}
        for k, pts in self.series.items():
            if len(pts) < min_n:
                continue
            t = np.array([p[0] for p in pts], float)
            y = np.array([p[1] for p in pts], float)
            slope = float(theilslopes(y, t)[0])
            tau, p = kendalltau(t, y)
            base = np.median(y[:3])
            rise_db = 10.0 * float(np.median(y[-3:]) - base)
            gate = ALARM_DB
            if adaptive and len(y) >= 5:
                gate = max(3.0, 3.0 * 10.0 * float(np.std(y[:5])))
            out[k] = {"slope": slope, "mk_p": float(p),
                      "rise_db": rise_db, "gate_db": float(gate),
                      "alarm": bool(slope > 0 and p < ALARM_P
                                    and rise_db > gate),
                      "n": len(pts)}
        return out

    def first_alarm(self, key, min_n=5):
        """Online detection: earliest t where the key alarms using only
        data up to t. None if never."""
        pts = self.series.get(key, [])
        for i in range(min_n, len(pts) + 1):
            t = np.array([p[0] for p in pts[:i]], float)
            y = np.array([p[1] for p in pts[:i]], float)
            if len(t) < min_n:
                continue
            slope = float(theilslopes(y, t)[0])
            tau, p = kendalltau(t, y)
            rise = 10.0 * float(np.median(y[-3:]) - np.median(y[:3]))
            if slope > 0 and p < ALARM_P and rise > ALARM_DB:
                return float(t[-1])
        return None


FAULT_KEY = {"imbalance": "SHAFT_1", "misalignment": "SHAFT_2",
             "looseness": "HALF", "bearing": "NEARRAT", "gear": "GMF"}


def frame_fingerprint(pf, pa, c, kmax=16.0):
    """Sparse invariant fingerprint of a speed frame: amplitude mass on
    the integer+half order grid under hypothesis c. Two records of the
    same machine agree in fingerprint exactly when their frames name the
    same physical orders — regardless of the operating speed."""
    d = {}
    for f, a in zip(pf, pa):
        o = f / c
        if not (0.4 <= o <= kmax + 0.3):
            continue
        b = round(o * 2) / 2
        if b >= 0.5 and abs(o - b) <= max(0.012 * o, 0.01):
            d[b] = d.get(b, 0.0) + float(a)
    return d


def fp_cos(d1, d2):
    keys = set(d1) | set(d2)
    if not keys:
        return 0.0
    v1 = np.array([d1.get(k, 0.0) for k in keys])
    v2 = np.array([d2.get(k, 0.0) for k in keys])
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(v1 @ v2 / n) if n > 0 else 0.0


class FrameSelector:
    """Speed lock v2: choose each record's frame by INVARIANT-LEDGER
    CONSISTENCY, not speed proximity (v1 died of early-error lock-in).
    Warmup: buffer the first `warmup` records' candidates, then pick the
    joint assignment by coordinate ascent on (confidence + pairwise
    fingerprint agreement) — record 1 gets no special authority. After
    warmup: score = alpha*conf + beta*cos(fingerprint, consensus);
    consensus follows chosen fingerprints by EMA."""

    def __init__(self, warmup=5, alpha=1.0, beta=1.4, gamma=1.0, ema=0.85):
        self.warmup, self.alpha, self.beta, self.ema = warmup, alpha, beta, ema
        self.gamma = gamma
        self.buf = []                    # [(cands, fps)]
        self.consensus = None
        self.warmup_choices = None       # final frame per warmup record

    @staticmethod
    def quality(fp):
        """Structural frame quality: a consistently ALIASED frame is as
        self-consistent as the true one, but its mass sits on the half
        grid; the true frame concentrates mass on integers with a live
        1x. quality = integer-mass fraction + 0.5 * 1x share."""
        tot = sum(fp.values()) + 1e-12
        ints = sum(v for k, v in fp.items() if k == round(k))
        return ints / tot + 0.5 * fp.get(1.0, 0.0) / tot

    def _fps(self, cands, pf, pa):
        return [frame_fingerprint(pf, pa, c["hz"])
                if np.isfinite(c["hz"]) and c["hz"] > 0 else {}
                for c in cands]

    def observe(self, cands, pf, pa):
        """Feed one record. Returns the chosen hz for this record; during
        warmup this is provisional (top-1) — read `warmup_choices` after
        the warmup completes to re-frame the buffered records."""
        fps = self._fps(cands, pf, pa)
        if self.consensus is not None:
            scores = [self.alpha * c["confidence"]
                      + self.beta * fp_cos(fp, self.consensus)
                      + self.gamma * self.quality(fp)
                      for c, fp in zip(cands, fps)]
            i = int(np.argmax(scores))
            ch = fps[i]
            if ch:
                for k in set(self.consensus) | set(ch):
                    self.consensus[k] = (self.ema * self.consensus.get(k, 0.0)
                                         + (1 - self.ema) * ch.get(k, 0.0))
            return cands[i]["hz"]
        self.buf.append((cands, fps))
        if len(self.buf) < self.warmup:
            return cands[0]["hz"]
        # consensus init by coordinate ascent, 3 sweeps
        pick = [0] * len(self.buf)
        for _ in range(3):
            for r, (cands_r, fps_r) in enumerate(self.buf):
                best_i, best_s = pick[r], -1e9
                for i, (c, fp) in enumerate(zip(cands_r, fps_r)):
                    s = (self.alpha * c["confidence"]
                         + self.gamma * self.quality(fp)
                         + self.beta * np.mean(
                             [fp_cos(fp, self.buf[q][1][pick[q]])
                              for q in range(len(self.buf)) if q != r]))
                    if s > best_s:
                        best_i, best_s = i, s
                pick[r] = best_i
        self.warmup_choices = [self.buf[r][0][pick[r]]["hz"]
                               for r in range(len(self.buf))]
        cons = {}
        for r in range(len(self.buf)):
            for k, v in self.buf[r][1][pick[r]].items():
                cons[k] = cons.get(k, 0.0) + v / len(self.buf)
        self.consensus = cons
        return self.warmup_choices[-1]


def select_speed(cands, f_prev, band=0.35):
    """Temporal speed lock: a monitored machine is never blind-per-
    record. Among O1's top-k candidates, blend the estimator's own
    confidence with consistency against the machine's speed history:
    a candidate near f_prev (within the plausible operating move `band`
    in log terms) gets a bonus; octave jumps of the coordinate frame are
    what this suppresses. Returns the selected hz."""
    if f_prev is None or not np.isfinite(f_prev):
        return cands[0]["hz"]
    best, best_s = cands[0]["hz"], -1e9
    for c in cands:
        if not np.isfinite(c["hz"]) or c["hz"] <= 0:
            continue
        d = abs(np.log(c["hz"] / f_prev))
        s = c["confidence"] + 0.6 * np.exp(-(d / band) ** 2)
        if s > best_s:
            best, best_s = c["hz"], s
    return best
