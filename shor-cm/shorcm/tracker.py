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
            e_c = sum(a ** 2 / 2 for _, a in cl)
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

    def trends(self, min_n=6):
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
            out[k] = {"slope": slope, "mk_p": float(p),
                      "rise_db": rise_db,
                      "alarm": bool(slope > 0 and p < ALARM_P
                                    and rise_db > ALARM_DB),
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
