"""Persistent probabilistic speed belief per asset.

The CWRU lesson: a healthy, well-balanced machine can show NO 1x at
all — blind per-record speed is then unidentifiable, and committing to
the wrong frame manufactures fault false alarms. The system answer:

 1. never commit prematurely — carry a POSTERIOR over speed hypotheses
    across records, seeded by the nameplate (asset registries always
    know approximate RPM);
 2. when a developing fault finally makes the shaft lattice undeniable,
    the posterior converges — and if the early MAP was wrong, a
    DECISIVE FLIP is detected and reported as a frame_correction so
    downstream trackers re-key their history instead of restarting;
 3. cross-sensor intelligence: sensors on the same shaft update the
    same belief (hypothesis-space product of calibrated confidences —
    NOT raw score averaging, which was measured harmful), each sensor
    weighted by its ONLINE reliability (agreement with the running
    MAP), so a misleading mount discounts itself. Sibling assets can
    seed extra prior hypotheses via `extra_prior`.

Hypotheses live on a fixed grid of RELATIVE speed (steady/nameplate
machines; VFD support = tracking the grid in relative terms is future
work). Octave-class structure is explicit: the grid spans
[nom/4, 4*nom] so 1/3x..3x locks are representable and correctable.
"""
import numpy as np

LOG_FLOOR = np.log(0.02)          # likelihood floor for unmatched hyp


class SpeedBelief:
    def __init__(self, f_nom, rel_lo=0.25, rel_hi=4.0, n_grid=161,
                 prior_sigma_oct=0.35, extra_prior=(),
                 flip_ratio=6.0, flip_hold=3):
        """f_nom: nameplate Hz (the anchor). Grid is log-spaced over
        [rel_lo, rel_hi] * f_nom. prior_sigma_oct: nameplate prior
        width in OCTAVES. extra_prior: iterable of (hz, weight) from
        siblings/history. flip: MAP must beat the previously-committed
        hypothesis by exp(flip_ratio) for flip_hold consecutive
        updates to fire a correction."""
        self.f_nom = float(f_nom)
        self.grid = f_nom * np.logspace(np.log10(rel_lo),
                                        np.log10(rel_hi), n_grid)
        d_oct = np.log2(self.grid / f_nom)
        self.logp = -(d_oct / prior_sigma_oct) ** 2 / 2
        for hz, w in extra_prior:
            j = int(np.argmin(np.abs(self.grid - hz)))
            self.logp[j] += float(w)
        self.logp -= self.logp.max()
        self.rel = {}                  # sensor -> reliability [0.2, 1]
        self._votes = {}               # sensor -> {band_key: count}
        self.committed = None          # committed grid index
        self._flip_count = 0
        self.flip_ratio = float(flip_ratio)
        self.flip_hold = int(flip_hold)
        self.n_updates = 0

    KIN = ((1.0, 1.0), (1 / 3, 0.35), (0.5, 0.35), (2 / 3, 0.2),
           (1.5, 0.2), (2.0, 0.35), (3.0, 0.35))

    def _like(self, cands):
        """Per-sensor log-likelihood over the grid: each candidate is
        evidence for its whole RATIONAL FAMILY (the Shor reading — a
        3x lock is evidence FOR the fundamental at c/3), with kin
        bumps at reduced weight. The nameplate prior then selects
        WITHIN the family, which is exactly what a candidate list
        alone cannot do on a machine with no 1x."""
        li = np.full(len(self.grid), np.exp(LOG_FLOOR))
        for c in cands:
            hz, w = c.get("hz"), max(float(c.get("confidence", 0.1)),
                                     0.05)
            if hz is None or not np.isfinite(hz) or hz <= 0:
                continue
            for k, kw in self.KIN:
                li += w * kw * np.exp(
                    -((self.grid / (hz * k) - 1) / 0.02) ** 2 / 2)
        return np.log(li / li.max())

    def update(self, sensor_cands):
        """sensor_cands: {sensor_name: candidate list}. Returns dict
        with map_hz, p_map, undecided, correction (None or dict)."""
        self.n_updates += 1
        for s, cands in sensor_cands.items():
            r = self.rel.get(s, 1.0)
            ll = self._like(cands)
            # correlated-evidence temper: re-measuring an UNCHANGED
            # spectrum is not new information — a sensor repeating the
            # same top hypothesis contributes with diminishing weight,
            # so a static wrong lock cannot out-shout the nameplate,
            # while NEW evidence (an emerging fault lattice) arrives
            # at full weight
            best = max((c for c in cands
                        if c.get("hz") and np.isfinite(c["hz"])),
                       key=lambda c: c.get("confidence", 0),
                       default=None)
            w_nov = 1.0
            if best is not None:
                key = int(round(np.log2(best["hz"] / self.f_nom) * 12))
                seen = self._votes.setdefault(s, {})
                w_nov = 1.0 / (1.0 + 0.6 * seen.get(key, 0))
                seen[key] = seen.get(key, 0) + 1
            self.logp += r * w_nov * ll
            # online reliability: does this sensor's best candidate
            # agree with the current MAP? EWMA into [0.2, 1].
            # KNOWN LIMIT (measured, iteration 27): in a long no-1x
            # phase this discounts honest sensors that merely see no
            # shaft evidence; the reliability/temper/kin parameter
            # surface needs an offline dev sweep before deployment.
            j = int(np.argmax(self.logp))
            agree = 0.0
            if best is not None:
                agree = float(any(
                    abs(best["hz"] * k / self.grid[j] - 1) < 0.04
                    for k in (1.0,)))
            self.rel[s] = float(np.clip(
                0.8 * r + 0.2 * (0.2 + 0.8 * agree), 0.2, 1.0))
        self.logp -= self.logp.max()
        self.logp = np.maximum(self.logp, -60.0)
        j = int(np.argmax(self.logp))
        p = np.exp(self.logp)
        p = p / p.sum()
        # mass within +/-4% of MAP (a hypothesis is a slip band)
        band = np.abs(self.grid / self.grid[j] - 1) < 0.04
        p_map = float(p[band].sum())
        undecided = p_map < 0.5
        correction = None
        if self.committed is None:
            if not undecided:
                self.committed = j
        else:
            jc = self.committed
            if abs(self.grid[j] / self.grid[jc] - 1) > 0.06:
                gap = float(self.logp[j] - self.logp[jc])
                self._flip_count = self._flip_count + 1 \
                    if gap > np.log(self.flip_ratio) else 0
                if self._flip_count >= self.flip_hold:
                    correction = {"from_hz": float(self.grid[jc]),
                                  "to_hz": float(self.grid[j]),
                                  "ratio": float(self.grid[j]
                                                 / self.grid[jc])}
                    self.committed = j
                    self._flip_count = 0
            else:
                self.committed = j       # slow drift within the band
                self._flip_count = 0
        return {"map_hz": float(self.grid[j]), "p_map": round(p_map, 3),
                "undecided": bool(undecided),
                "correction": correction,
                "reliability": {s: round(r, 2)
                                for s, r in self.rel.items()}}
