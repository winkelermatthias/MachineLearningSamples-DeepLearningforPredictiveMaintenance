"""General pattern decomposition and tracking (O2 proper).

The spectrum of one record is decomposed into NAMED, ISOLATED patterns:
every spectral bin belongs to at most one pattern, energies are
band-integrated (ENBW-corrected, floor-subtracted), and shares plus the
FLOOR residual sum to 1 by construction. Vocabulary:

  FIXEDHZ(f)        crystal-narrow non-synchronous line family (grid
                    hum, drive line, neighbor at fixed Hz) — identity
                    lives in Hz, smears in the order domain
  HARM(r)           harmonic family of base order r: the input shaft
                    (r = 1), a second shaft / belt lattice (r != 1)
  HALFHARM          the q = 2 half-order ladder (looseness)
  SIDEBAND(c, d)    symmetric fan spaced d around carrier order c —
                    WITH or WITHOUT a visible carrier (suppressed-
                    carrier modulation is a classic fault signature)
  NEARRAT(o)        drifting near-rational tone (bearing): wide cluster
                    or unsnapped narrow with LOW block coherence;
                    harmonic at 2o joins the same pattern
  TONE(o)           strong isolated narrow line with no family — a lone
                    vane-pass, second-shaft or drive line
  BAND(lo, hi)      broadband hump (resonance excitation)
  FLOOR             everything unclaimed

Isolation is enforced by a claimed-bin mask: extraction is greedy in
the order above, and a bin claimed once never counts again.

Tracking: each pattern type carries an INVARIANT identity so instances
persist under varying speed — HARM by base ratio, SIDEBAND by (carrier
order, spacing), NEARRAT by its slip band, FIXEDHZ by Hz. The
GeneralTracker keeps a registry per identity with the same Theil-Sen +
Mann-Kendall + adaptive-gate growth layer as the key tracker.
"""
import numpy as np
from fractions import Fraction

from . import tacho as T
from . import spectra as S
from . import peakshor as PS

ENBW = 1.5
MAX_ORDER = 130.0


def _trend(pts, alarm_p=0.01, base_db=6.0, adaptive=True):
    from scipy.stats import theilslopes, kendalltau
    if len(pts) < 6:
        return None
    t = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    slope = float(theilslopes(y, t)[0])
    _, p = kendalltau(t, y)
    rise = 10.0 * float(np.median(y[-3:]) - np.median(y[:3]))
    gate = base_db
    if adaptive and len(y) >= 5:
        gate = max(3.0, 3.0 * 10.0 * float(np.std(y[:5])))
    return {"slope": slope, "mk_p": float(p), "rise_db": rise,
            "gate_db": float(gate),
            "alarm": bool(slope > 0 and p < alarm_p and rise > gate),
            "n": len(pts)}


class _Spec:
    """Order spectrum + bookkeeping for one record."""

    def __init__(self, x, fs, f_hat, spr=256):
        ph, _ = T.phase_from_comb(x, fs, f_nom=f_hat, prior_rel_sigma=0.008)
        f_inst = np.gradient(ph) * fs / (2 * np.pi)
        med = np.median(f_inst) + 1e-15
        q16, q84 = np.percentile(f_inst, [16, 84])
        self.wander_frac = float(max((q84 - q16) / 2 / med, 0.0))
        xa = S.angular_resample(x, ph, spr)
        self.A, self.o = S.fine_order_spectrum(xa, spr)
        Z, _ = S.block_spectra(xa, spr, revs=5)
        _, _, self.ratio_b = S.coherent_split(Z)
        self.do = self.o[1] - self.o[0]
        self.floor = float(np.median(self.A))
        self.claimed = np.zeros(len(self.A), bool)
        self.max_o = min(spr / 2 - 2.0, MAX_ORDER)
        m = self.o <= self.max_o
        self.e_tot = float(np.sum(self.A[m] ** 2) / 2) / ENBW
        self.f_hat = f_hat
        self.revs = max(int(len(xa) / spr), 10)

    def coh(self, o_c):
        kb = int(round(o_c * 5))
        return float(self.ratio_b[kb]) if kb < len(self.ratio_b) else 0.0

    def band(self, o_lo, o_hi, claim=True):
        """Band energy over [o_lo, o_hi], floor-subtracted, only over
        UNCLAIMED bins; optionally claims them."""
        i0 = max(int(o_lo / self.do), 1)
        i1 = min(int(o_hi / self.do) + 1, len(self.A))
        if i1 <= i0:
            return 0.0
        m = ~self.claimed[i0:i1]
        if not m.any():
            return 0.0
        seg = self.A[i0:i1][m]
        e = float(max(np.sum(seg ** 2) / 2
                      - len(seg) * self.floor ** 2 / 2, 0.0)) / ENBW
        if claim:
            self.claimed[i0:i1] |= True
        return e

    def band_local(self, o_lo, o_hi):
        """Like band(), but subtracts the LOCAL background density
        (median of the unclaimed ring around the band) instead of the
        global floor — a narrow line sitting on a hump must be credited
        its line energy only; the hump's slice under it goes to FLOOR,
        not to the line. Bins are claimed either way."""
        i0 = max(int(o_lo / self.do), 1)
        i1 = min(int(o_hi / self.do) + 1, len(self.A))
        if i1 <= i0:
            return 0.0
        ring = np.concatenate([self.A[max(i0 - 60, 0):i0][
            ~self.claimed[max(i0 - 60, 0):i0]],
            self.A[i1:i1 + 60][~self.claimed[i1:i1 + 60]]])
        bg = float(np.median(ring)) if len(ring) >= 10 else self.floor
        bg = max(bg, self.floor)
        m = ~self.claimed[i0:i1]
        if not m.any():
            return 0.0
        seg = self.A[i0:i1][m]
        e = float(max(np.sum(seg ** 2) / 2
                      - len(seg) * bg ** 2 / 2, 0.0)) / ENBW
        self.claimed[i0:i1] |= True
        return e

    def amp_at(self, o_c, halfw=None):
        i = int(round(o_c / self.do))
        h = 2 if halfw is None else max(int(halfw / self.do), 2)
        if i < 2 or i > len(self.A) - 3:
            return 0.0
        return float(self.A[max(i - h, 0):i + h + 1].max())

    def unclaimed_peaks(self, guard=4.0, max_order=None):
        mo = max_order or self.max_o
        pk = S.peak_orders(self.A, self.o, mo, 60, guard=guard)
        out = []
        for oo, aa, prom in pk:
            i = int(round(oo / self.do))
            if 0 <= i < len(self.A) and not self.claimed[i]:
                out.append((oo, aa))
        return out


def _tone_halfw(o_c):
    return max(0.006 * o_c, 0.03)


def decompose(x, fs, f_hat, spr=256, sheet=None):
    """Isolating pattern decomposition. Returns (patterns, e_total):
    patterns is a list of dicts with type, params, energy, share,
    members; shares + FLOOR sum to 1."""
    sp = _Spec(x, fs, f_hat, spr)
    pats = []

    # ---- 1 FIXEDHZ: known grid + detected narrow Hz lines ----
    fz, Az = PS.spectrum(x, fs)
    medz = np.median(Az) + 1e-15
    fixed_hz = []

    def _crystal(hz):
        # a mains/drive line lives in 1-2 Hz bins; a shaft harmonic at
        # the same position smears over its wander width — narrowness
        # discriminates ONLY where that width is resolvable (>= 3 Hz
        # bins), which protects low-order shaft lines whose smear is
        # sub-bin
        dfb = fs / len(x)
        if sp.wander_frac * hz < 3.0 * dfb:
            return False
        j = int(round(hz * len(x) / fs))
        if not (3 < j < len(Az) - 26):
            return False
        j = j - 2 + int(np.argmax(Az[j - 2:j + 3]))
        seg = Az[j - 25:j + 26]
        loc = float(np.median(seg))       # floor-subtract: a modest
        num = float(Az[j - 1:j + 2].sum() - 3 * loc)   # line's wide-
        den = float(seg.sum() - 51 * loc)              # window sum is
        return num > 0 and num / (den + 1e-15) > 0.72  # noise-dominated

    def shaft_coincident(hz):
        # cap the proportional tolerance: beyond order ~8 an uncapped
        # 3%-of-order window exceeds the half-integer grid spacing and
        # would exempt EVERY high-order fixed line; and a line that is
        # crystal-narrow in Hz is a fixed source even ON a half-integer
        # (a coincident shaft harmonic would be smeared by wander)
        o = hz / f_hat
        near = abs(o - round(o * 2) / 2) < min(0.03 * max(o, 1.0), 0.12)
        return near and not _crystal(hz)

    for base in (50.0, 60.0, 100.0, 120.0):
        j = int(round(base * len(x) / fs))
        if 0 < j < len(Az) - 1 and Az[max(j - 2, 0):j + 3].max() > 5 * medz \
                and not shaft_coincident(base):
            fixed_hz.append(base)
    for j in range(12, len(Az) - 12):
        if Az[j] > 8 * medz and Az[j] >= Az[j - 1] and Az[j] > Az[j + 1]:
            conc = Az[j - 1:j + 2].sum() / (Az[j - 12:j + 13].sum() + 1e-12)
            hz = j * fs / len(x)
            if conc > 0.6 and all(abs(hz - b) > 2.0 for b in fixed_hz) \
                    and not shaft_coincident(hz):
                fixed_hz.append(round(hz, 1))
    for hz in fixed_hz:
        o_c = hz / f_hat
        if not (0.3 < o_c < sp.max_o):
            continue
        # a fixed line smears over ~wander*o in the order domain; claim
        # THAT width and credit only the local-background-subtracted
        # line energy — a wide claim credited against the global floor
        # swallows any hump it sits on (15 dB overcredit observed)
        w = max(1.5 * sp.wander_frac * o_c, 0.03)
        e = sp.band_local(o_c - w, o_c + w)
        if e > 0:
            pats.append({"type": "FIXEDHZ", "params": {"hz": float(hz)},
                         "energy": e, "members": [(round(o_c, 3), e)]})

    # ---- 2 HARM(1): input-shaft harmonic family ----
    def harm_family(r, kmax=None):
        """A harmonic family is a CONTIGUOUS decaying run from k = 1;
        it ends after 3 consecutive missing members. A resurgence of
        integer-order energy after a long gap (mesh + fan around order
        ~z) is a different pattern and must stay unclaimed here."""
        kmax = kmax or int(sp.max_o / r)
        members = []
        e = 0.0
        misses = 0
        bands = []
        for k in range(1, min(kmax, 60) + 1):
            o_c = k * r
            if o_c > sp.max_o or misses >= 3:
                break
            a = sp.amp_at(o_c)
            i = int(round(o_c / sp.do))
            loc = np.median(sp.A[max(i - 40, 0):i + 40]) + 1e-15
            if a > 3.0 * loc:
                w = _tone_halfw(o_c)
                ek = sp.band(o_c - w, o_c + w, claim=False)  # dry run:
                if ek > 0:               # a rejected family must leave
                    members.append((round(float(o_c), 3), ek))
                    bands.append((o_c - w, o_c + w))
                    e += ek              # no claims behind (probe leak
                    misses = 0           # punctured NEARRAT humps)
                else:
                    misses += 1
            else:
                misses += 1
        return e, members, bands

    e1, m1, b1 = harm_family(1.0)
    if e1 > 0:
        for lo_, hi_ in b1:
            sp.band(lo_, hi_)            # commit claims
        pats.append({"type": "HARM", "params": {"base": 1.0},
                     "energy": e1, "members": m1})

    # ---- 3 HALFHARM ----
    eh, mh = 0.0, []
    for k in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5):
        a = sp.amp_at(k)
        i = int(round(k / sp.do))
        loc = np.median(sp.A[max(i - 40, 0):i + 40]) + 1e-15
        if a > 3.0 * loc:
            w = _tone_halfw(k)
            ek = sp.band(k - w, k + w)
            if ek > 0:
                mh.append((k, ek)); eh += ek
    if len(mh) >= 2:
        pats.append({"type": "HALFHARM", "params": {},
                     "energy": eh, "members": mh})
    elif mh:                                # lone half-line: give it back
        pass                                # (stays claimed; counted FLOOR-
                                            # adjacent via residual)

    # ---- 4 second-lattice HARM(r) from remaining peaks ----
    rem = sp.unclaimed_peaks(guard=4.0, max_order=20.0)
    if len(rem) >= 2:
        po = np.array([p[0] for p in rem])
        pa_ = np.array([p[1] for p in rem])
        cands = PS.pair_candidates(po, pa_, lo=0.2, hi=6.0, topn=3)
        for r in cands:
            if abs(r - 1.0) < 0.03 or abs(r - 0.5) < 0.02:
                continue
            # a second shaft is PHASE-LOCKED to the input (gear ratio);
            # a bearing pair at (o, 2o) mimics the lattice geometry but
            # drifts — coherence is the discriminator
            if sp.coh(r) < 0.45:
                continue
            e2, m2, b2 = harm_family(r, kmax=int(min(sp.max_o / r, 12)))
            if len(m2) >= 2 and e2 > 0:
                for lo_, hi_ in b2:
                    sp.band(lo_, hi_)    # commit only on acceptance
                pats.append({"type": "HARM",
                             "params": {"base": round(float(r), 4)},
                             "energy": e2, "members": m2})
                break

    # ---- 4b sheet-seeded mesh fans: the asset registry KNOWS the
    # tooth count — at high order, wander smears mesh + fan beyond
    # blind detection (smear = wander*order vs fan spacing ~1); seed
    # the carrier and spacings from the kinematic sheet ----
    if sheet:
        mesh_orders, spacings = [], [1.0]
        if sheet.get("mesh"):
            zg = float(sheet["mesh"])
            mesh_orders += [zg, 2.0 * zg]
        pl = sheet.get("planetary")
        if pl:
            Zs, Zp, Zr, Np = pl
            rc = Zs / (Zs + Zr)
            mesh_orders += [Zr * rc, 2 * Zr * rc]
            spacings += [rc, Np * rc]
        if sheet.get("ratio"):
            spacings.append(float(sheet["ratio"]))
        for o_g in mesh_orders:
            if not (4.0 < o_g < sp.max_o - 1):
                continue
            i_g = int(round(o_g / sp.do))
            if sp.claimed[i_g]:
                continue
            # resolved-fan attempt at known spacings
            best = None
            for d in spacings:
                mem = []
                for k in (1, 2, 3):
                    for sgn in (-1, 1):
                        o_m = o_g + sgn * k * d
                        if not (0.5 < o_m < sp.max_o):
                            continue
                        a = sp.amp_at(o_m)
                        j = int(round(o_m / sp.do))
                        loc = np.median(
                            sp.A[max(j - 40, 0):j + 40]) + 1e-15
                        if a > 3.5 * loc:
                            mem.append(o_m)
                if len(mem) >= 2 and (best is None
                                      or len(mem) > len(best[1])):
                    best = (d, mem)
            e, members = 0.0, []
            a_c = sp.amp_at(o_g)
            loc_c = np.median(sp.A[max(i_g - 40, 0):i_g + 40]) + 1e-15
            has_c = a_c > 3.0 * loc_c
            if best is not None and (has_c or len(best[1]) >= 3):
                d, mem = best
                if has_c:
                    w = max(_tone_halfw(o_g),
                            1.5 * sp.wander_frac * o_g)
                    ec = sp.band(o_g - w, o_g + w)
                    e += ec
                    members.append((round(float(o_g), 3), ec))
                for o_m in mem:
                    w = min(max(_tone_halfw(o_m),
                                1.5 * sp.wander_frac * o_m), 0.35 * d)
                    em = sp.band(o_m - w, o_m + w)
                    e += em
                    members.append((round(float(o_m), 3), em))
                dlt = d
            else:
                # fused case: one lump around the mesh order — claim a
                # BOUNDED region (fan extent), gated on local density
                dmax = max(spacings)
                ext = min(1.5 * sp.wander_frac * o_g + 2.0 * dmax,
                          3.2 * dmax)
                e_p = sp.band(o_g - ext, o_g + ext, claim=False)
                nb = 2 * ext / sp.do
                floor_e = nb * sp.floor ** 2 / 2 / ENBW
                if e_p <= 5.0 * floor_e:
                    continue
                e = sp.band(o_g - ext, o_g + ext)
                members = [(round(float(o_g), 3), e)]
                dlt = dmax
            if e > 0:
                pats.append({"type": "SIDEBAND",
                             "params": {"carrier": round(float(o_g), 3),
                                        "spacing": round(float(dlt), 3),
                                        "carrier_visible": bool(has_c),
                                        "seeded": True},
                             "energy": e, "members": members})

    # ---- 5 SIDEBAND fans (with or without carrier) ----
    for _pass in range(2):
        rem = sp.unclaimed_peaks(guard=4.0)
        if len(rem) < 2:
            break
        rem.sort(key=lambda p: -p[1])
        found = False
        carr_cands = [p[0] for p in rem[:6]]
        # suppressed-carrier candidates: midpoints of equal-spaced pairs
        for i in range(min(len(rem), 8)):
            for j in range(i + 1, min(len(rem), 8)):
                mid = (rem[i][0] + rem[j][0]) / 2
                if abs(rem[i][0] - rem[j][0]) > 0.2 and mid > 1.5:
                    carr_cands.append(mid)
        best = None
        for c_o in carr_cands:
            # modulation fans live around mesh/passage carriers; a "fan"
            # below order 5 is almost surely a diffused tone's fragments
            if c_o < 5.0 or c_o > sp.max_o - 1:
                continue
            for dlt in np.arange(0.15, 3.55, 0.02):
                mem = []
                for k in (1, 2, 3):
                    for sgn in (-1, 1):
                        o_m = c_o + sgn * k * dlt
                        if not (0.5 < o_m < sp.max_o):
                            continue
                        a = sp.amp_at(o_m)
                        i = int(round(o_m / sp.do))
                        # a member is unavailable only when its window
                        # is MOSTLY claimed — a narrow FIXEDHZ line
                        # riding on it must not break the fan geometry
                        # (band() integrates only the unclaimed
                        # remainder, so nothing double-counts)
                        hw = max(int(_tone_halfw(o_m) / sp.do), 2)
                        cl = sp.claimed[max(i - hw, 0):i + hw + 1]
                        if len(cl) and cl.mean() > 0.7:
                            continue
                        loc = np.median(
                            sp.A[max(i - 40, 0):i + 40]) + 1e-15
                        if a > 3.5 * loc:
                            mem.append(o_m)
                if len(mem) >= 3 and (best is None
                                      or len(mem) > len(best[2])):
                    best = (c_o, dlt, mem)
        if best is None:
            break
        c_o, dlt, mem = best
        e = 0.0
        members = []
        i_c = int(round(c_o / sp.do))
        has_carrier = not sp.claimed[i_c] and sp.amp_at(c_o) > 4 * sp.floor
        if has_carrier:
            w = _tone_halfw(c_o)
            ec = sp.band(c_o - w, c_o + w)
            e += ec; members.append((round(float(c_o), 3), ec))
        for o_m in mem:
            w = min(_tone_halfw(o_m), 0.35 * dlt)
            em = sp.band(o_m - w, o_m + w)
            e += em; members.append((round(float(o_m), 3), em))
        if e > 0:
            pats.append({"type": "SIDEBAND",
                         "params": {"carrier": round(float(c_o), 3),
                                    "spacing": round(float(dlt), 3),
                                    "carrier_visible": bool(has_carrier)},
                         "energy": e, "members": members})
            found = True
        if not found:
            break

    # ---- 6a NEARRAT as residual HUMPS: a heavily phase-walking tone
    # is a broad bump (width ~1-45% of its order), not a peak set; the
    # peak-cluster path below only catches the lightly-smeared case ----
    from scipy.ndimage import median_filter as _mf
    tolh = max(S.snap_tol(sp.revs), 0.004)
    res_h = np.where(sp.claimed, 0.0, sp.A)
    smh = _mf(res_h, size=max(int(0.05 / sp.do) | 1, 3))
    mban = (sp.o > 1.2) & (sp.o < 20.0)   # geared-down bearings land
    # below 1.8 (order = ratio * BPFx), big-bearing 2x reaches past 16
    hoth = (smh > 1.8 * sp.floor) & mban
    if hoth.any():
        idxh = np.flatnonzero(hoth)
        splitsh = np.flatnonzero(np.diff(idxh) > int(0.05 / sp.do))
        raw_segs = [(sp.o[g[0]], sp.o[g[-1]])
                    for g in np.split(idxh, splitsh + 1) if len(g) >= 2]
        # a diffused tone's shoulders carry much of its energy but dip
        # under any fixed threshold (and claimed-bin zeros break
        # hysteresis): integrate with PROPORTIONAL margins. Merge only
        # segments whose RAW edges nearly touch — extended-margin
        # overlap fuses distinct tones across genuinely cold valleys
        # into one centerless pseudo-hump
        merged = []                # [[lo_ext, hi_ext, [consts], raw_hi]]
        for o_lo, o_hi in raw_segs:
            o_c0 = (o_lo + o_hi) / 2
            lo_e, hi_e = o_lo - 0.08 * o_c0, o_hi + 0.08 * o_c0
            if merged and o_lo - merged[-1][3] < 0.04 * o_lo:
                merged[-1][1] = max(merged[-1][1], hi_e)
                merged[-1][2].append((o_lo, o_hi))
                merged[-1][3] = max(merged[-1][3], o_hi)
            else:
                merged.append([lo_e, hi_e, [(o_lo, o_hi)], o_hi])
        merged = [(lo, hi, consts) for lo, hi, consts, _ in merged]

        def seed_e(seg):
            i0, i1 = int(seg[0] / sp.do), int(seg[1] / sp.do) + 1
            return float(np.sum(sp.A[i0:i1] ** 2))

        cands_h = []
        for o_lo, o_hi, consts in merged:
            if o_hi - o_lo > 1.5:
                # chained too far: split back into constituents — each
                # real tone in the chain deserves its own shot (keeping
                # only the dominant one silently dropped its neighbors)
                for c in sorted(consts, key=seed_e, reverse=True)[:4]:
                    o_c0 = (c[0] + c[1]) / 2
                    cands_h.append((c[0] - 0.08 * o_c0,
                                    c[1] + 0.08 * o_c0))
            else:
                cands_h.append((o_lo, o_hi))
        for o_lo, o_hi in cands_h:
            o_c = (o_lo + o_hi) / 2
            wid = o_hi - o_lo
            if not (0.02 <= wid <= 1.5):
                continue
            near_rat = not (abs(o_c - round(o_c)) <= max(tolh, 0.01)
                            or sp.coh(o_c) >= 0.45)
            if not near_rat and wid > 0.6:
                continue                   # wide coherent stuff -> BAND
            e = sp.band(o_lo - 0.02 * o_c, o_hi + 0.02 * o_c,
                        claim=False)
            # anti-clutter gate relative to the FLOOR energy in the
            # band, not to e_tot — a small real tone on a big machine
            # (belt lattice at e ~ 0.2% of total) is still a pattern;
            # a noise blob barely over the hump threshold is not
            nb = (o_hi - o_lo + 0.04 * o_c) / sp.do
            floor_e = nb * sp.floor ** 2 / 2 / ENBW
            if e <= max(4.0 * floor_e, 0.001 * sp.e_tot):
                continue
            sp.band(o_lo - 0.02 * o_c, o_hi + 0.02 * o_c)
            members = [(round(float(o_c), 3), e)]
            if not near_rat:
                # an on-integer / shaft-coherent hump is a wander-
                # smeared LOCKED tone (vane pass on a hot record, a
                # second-shaft line) — TONE, not bearing, not floor
                pats.append({"type": "TONE",
                             "params": {"order": round(float(o_c), 3)},
                             "energy": e, "members": members})
                continue
            # 2x member: NARROW dry-run probe, claim only on acceptance
            # — an unconditional wide claim mis-centered at 2*o_c can
            # swallow a NEIGHBORING hump (a second bearing tone) whole
            w2 = min(wid, 0.06 * o_c)
            e2x = sp.band(2 * o_c - w2, 2 * o_c + w2, claim=False)
            if e2x > 0.05 * e:
                sp.band(2 * o_c - w2, 2 * o_c + w2)
                e += e2x
                members.append((round(float(2 * o_c), 3), e2x))
            pats.append({"type": "NEARRAT",
                         "params": {"order": round(float(o_c), 3)},
                         "energy": e, "members": members})

    # ---- 6 NEARRAT drifting tones (+ their 2x) ----
    tol = max(S.snap_tol(sp.revs), 0.004)
    pk = S.peak_orders(sp.A, sp.o, 20.0, 30, guard=4.0)
    clusters = []
    for oo, aa, _ in sorted(pk):
        if clusters and oo - clusters[-1][-1][0] < 0.008 * oo:
            clusters[-1].append((oo, aa))
        else:
            clusters.append([(oo, aa)])
    # slip-band grouping: a phase-walking tone fragments into several
    # sub-clusters over +/-4%; they are ONE pattern
    grouped = []
    for cl in clusters:
        a_c = sum(a for _, a in cl)
        o_c = sum(o_ * a for o_, a in cl) / (a_c + 1e-15)
        if grouped and abs(o_c / grouped[-1][-1][0] - 1) < 0.035:
            grouped[-1].extend([(m, a) for m, a in cl])
        else:
            grouped.append(list(cl))
    for cl in grouped:
        cl.sort()
        a_c = sum(a for _, a in cl)
        o_c = sum(o_ * a for o_, a in cl) / (a_c + 1e-15)
        i = int(round(o_c / sp.do))
        if sp.claimed[i] or not (1.2 < o_c < 20.0):
            continue
        width = (cl[-1][0] - cl[0][0]) / o_c if len(cl) > 1 else 0.0
        fr = Fraction(o_c).limit_denominator(8)
        snapped = (abs(float(fr) - o_c) <= tol
                   and (fr.denominator == 1 or fr.numerator <= 10))
        drifting = width > 0.004 or len(cl) >= 3 or not snapped
        if not (drifting and abs(o_c - round(o_c)) > tol
                and sp.coh(o_c) < 0.45):
            continue
        e = sp.band(cl[0][0] - 0.03 * o_c, cl[-1][0] + 0.03 * o_c)
        members = [(round(float(o_c), 3), e)]
        e2x = sp.band(2 * o_c * (1 - 0.01), 2 * o_c * (1 + 0.01),
                      claim=False)
        if e2x > 0.05 * e:
            sp.band(2 * o_c * (1 - 0.01), 2 * o_c * (1 + 0.01))
            e += e2x; members.append((round(float(2 * o_c), 3), e2x))
        if e > 0:
            pats.append({"type": "NEARRAT",
                         "params": {"order": round(float(o_c), 3)},
                         "energy": e, "members": members})

    # ---- 6b TONE: strong isolated narrow lines with no family ----
    # a lone vane-pass / second-shaft / drive line that never joined a
    # HARM run or a fan is still a NAMED pattern, not floor
    rem = sp.unclaimed_peaks(guard=5.0)
    rem.sort(key=lambda p: -p[1])
    for oo, aa in rem[:6]:
        i = int(round(oo / sp.do))
        loc = np.median(sp.A[max(i - 40, 0):i + 40]) + 1e-15
        # window adapts to wander smear: at high order a shaft-locked
        # tone (vane pass) is a narrow HUMP, not a 3-bin peak
        w = max(_tone_halfw(oo), 2.0 * sp.wander_frac * oo)
        hw = max(int(w / sp.do), 2)
        if aa < 4.0 * loc:
            continue
        seg = sp.A[max(i - hw, 0):i + hw + 1]
        dens = float(np.mean(seg ** 2))
        if dens < 6.0 * loc ** 2:
            continue
        e = sp.band(oo - w, oo + w)
        if e > 0.003 * sp.e_tot:
            pats.append({"type": "TONE",
                         "params": {"order": round(float(oo), 3)},
                         "energy": e,
                         "members": [(round(float(oo), 3), e)]})

    # ---- 7 BAND humps in the smoothed residual ----
    from scipy.ndimage import median_filter
    m = (sp.o > 0.3) & (sp.o <= sp.max_o)
    res = np.where(sp.claimed, 0.0, sp.A)
    sm = median_filter(res, size=max(int(1.0 / sp.do) | 1, 3))
    hot = (sm > 3.0 * sp.floor) & m
    if hot.any():
        idx = np.flatnonzero(hot)
        splits = np.flatnonzero(np.diff(idx) > int(0.3 / sp.do))
        segs = np.split(idx, splits + 1)
        for seg in segs:
            if len(seg) * sp.do < 0.5:
                continue
            o_lo, o_hi = sp.o[seg[0]], sp.o[seg[-1]]
            e = sp.band(o_lo, o_hi)
            if e > 0.01 * sp.e_tot:
                pats.append({"type": "BAND",
                             "params": {"lo": round(float(o_lo), 2),
                                        "hi": round(float(o_hi), 2)},
                             "energy": e,
                             "members": [(round(float((o_lo + o_hi) / 2),
                                                2), e)]})

    # ---- shares + FLOOR ----
    e_claimed = sum(p["energy"] for p in pats)
    for p in pats:
        p["share"] = float(p["energy"] / (sp.e_tot + 1e-15))
    pats.append({"type": "FLOOR",
                 "params": {"wander_frac": round(sp.wander_frac, 5)},
                 "energy": float(max(sp.e_tot - e_claimed, 0.0)),
                 "share": float(max(1.0 - e_claimed / (sp.e_tot + 1e-15),
                                    0.0)),
                 "members": []})
    pats.sort(key=lambda p: -p["share"])
    return pats, sp.e_tot


# ---------------- fixed-size feature block for the fault ML ----------

PF_COLS = ["pf_harm1", "pf_harm1_2x1x", "pf_harm1_n", "pf_harm_oth",
           "pf_half", "pf_half_n", "pf_sb_seed", "pf_sb_blind",
           "pf_sb_spacing", "pf_nearrat_top", "pf_nearrat_tot",
           "pf_nearrat_n", "pf_nearrat_frac", "pf_fixedhz", "pf_tone",
           "pf_band", "pf_floor", "pf_wander"]


def pattern_features(pats, wander_frac=None):
    """Fixed-size numeric summary of one record's decomposition, for
    the 6-way fault classifier. Shares (not raw energies) so machines
    of different overall level are comparable."""
    if wander_frac is None:
        wander_frac = next((p["params"].get("wander_frac", 0.0)
                            for p in pats if p["type"] == "FLOOR"), 0.0)
    d = dict.fromkeys(PF_COLS, 0.0)
    nr = []
    for p in pats:
        s, t = p.get("share", 0.0), p["type"]
        if t == "HARM" and abs(p["params"]["base"] - 1.0) < 0.03:
            d["pf_harm1"] += s
            d["pf_harm1_n"] = float(len(p["members"]))
            mem = {round(mo): me for mo, me in p["members"]}
            if mem.get(1, 0) > 0:
                d["pf_harm1_2x1x"] = float(
                    np.log10((mem.get(2, 0) + 1e-12) / (mem[1] + 1e-12)))
        elif t == "HARM":
            d["pf_harm_oth"] += s
        elif t == "HALFHARM":
            d["pf_half"] += s
            d["pf_half_n"] = float(len(p["members"]))
        elif t == "SIDEBAND":
            if p["params"].get("seeded"):
                d["pf_sb_seed"] += s
            else:
                d["pf_sb_blind"] += s
                d["pf_sb_spacing"] = max(d["pf_sb_spacing"],
                                         p["params"]["spacing"])
        elif t == "NEARRAT":
            nr.append((s, p["params"]["order"]))
        elif t == "FIXEDHZ":
            d["pf_fixedhz"] += s
        elif t == "TONE":
            d["pf_tone"] += s
        elif t == "BAND":
            d["pf_band"] += s
        elif t == "FLOOR":
            d["pf_floor"] = s
    if nr:
        nr.sort(reverse=True)
        d["pf_nearrat_top"] = nr[0][0]
        d["pf_nearrat_tot"] = sum(s for s, _ in nr)
        d["pf_nearrat_n"] = float(len(nr))
        o = nr[0][1]
        d["pf_nearrat_frac"] = abs(o - round(o))
    d["pf_wander"] = float(wander_frac)
    return d


# ---------------- generalized tracking ----------------

def _identity(p):
    t = p["type"]
    q = p["params"]
    if t == "HARM":
        return ("HARM", round(q["base"], 2))
    if t == "HALFHARM":
        return ("HALFHARM",)
    if t == "SIDEBAND":
        return ("SIDEBAND", round(q["carrier"], 1), round(q["spacing"], 2))
    if t == "NEARRAT":
        return ("NEARRAT", round(q["order"], 1))
    if t == "TONE":
        return ("TONE", round(q["order"], 1))
    if t == "FIXEDHZ":
        return ("FIXEDHZ", round(q["hz"], 0))
    if t == "BAND":
        return ("BAND", round((q["lo"] + q["hi"]) / 2, 0))
    return (t,)


def _match(t, a, b):
    """Slip/tolerance-aware identity match per type."""
    if t == "HARM":
        return abs(a[1] / max(b[1], 1e-9) - 1) < 0.03
    if t == "SIDEBAND":
        return (abs(a[1] / max(b[1], 1e-9) - 1) < 0.05
                and abs(a[2] / max(b[2], 1e-9) - 1) < 0.12)
    if t == "NEARRAT":
        return abs(a[1] / max(b[1], 1e-9) - 1) < 0.045
    if t == "TONE":
        return abs(a[1] / max(b[1], 1e-9) - 1) < 0.03
    if t == "FIXEDHZ":
        return abs(a[1] / max(b[1], 1e-9) - 1) < 0.015
    if t == "BAND":
        return abs(a[1] / max(b[1], 1e-9) - 1) < 0.2
    return a == b


class GeneralTracker:
    """Registry of pattern instances across records; identities are the
    per-type invariants, energies come from the isolating decompose."""

    def __init__(self, max_per_type=4):
        self.reg = []                      # {"id","type","pts","last"}
        self.max_per_type = max_per_type

    def update(self, t, patterns):
        for p in patterns:
            if p["type"] == "FLOOR" or p["energy"] <= 0:
                continue
            pid = _identity(p)
            inst = None
            for r in self.reg:
                if r["id"][0] == pid[0] and _match(pid[0], pid, r["id"]):
                    inst = r
                    break
            if inst is None:
                same = [r for r in self.reg if r["id"][0] == pid[0]]
                if len(same) >= self.max_per_type:
                    same.sort(key=lambda r: len(r["pts"]))
                    self.reg.remove(same[0])
                inst = {"id": pid, "type": pid[0], "pts": []}
                self.reg.append(inst)
            inst["pts"].append((t, np.log10(p["energy"] + 1e-15)))
            inst["id"] = pid               # slow drift of the identity

    def trends(self, adaptive=True):
        out = {}
        for r in self.reg:
            td = _trend(r["pts"], adaptive=adaptive)
            if td is not None:
                out[str(r["id"])] = td
        return out

    def groups(self):
        """Cluster instances that are kinematically ONE source: a
        bearing tone (NEARRAT), its modulation fan (SIDEBAND at the
        same carrier order or its 2x), a lone TONE at that order, and
        co-carrier fans. Returns a list of index lists into self.reg."""
        def _anchor(r):
            t, q = r["id"][0], r["id"]
            if t in ("NEARRAT", "TONE"):
                return q[1]
            if t == "SIDEBAND":
                return q[1]
            return None

        def _linked(ra, rb):
            a, b = _anchor(ra), _anchor(rb)
            if a is None or b is None or a < 0.05 or b < 0.05:
                return False
            for m in (1.0, 2.0, 0.5):
                if abs(a / (m * b) - 1) < 0.08:
                    return True
            return False

        n = len(self.reg)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                if _linked(self.reg[i], self.reg[j]):
                    parent[find(i)] = find(j)
        out = {}
        for i in range(n):
            out.setdefault(find(i), []).append(i)
        return list(out.values())

    def cusum(self, warmup=4, k_sigma=1.0, h_sigma=9.0):
        """SUPPLEMENTARY high-recall growth channel (not part of the
        certified alarm contract): one-sided CUSUM on log10-energy
        against a warmup baseline. The early-warning hypothesis was
        KILLED on dev — with honest (robust) scatter estimation the
        delay matches the trend gate, because delay is set by
        detectability, not by the trend test's 6-point need. What
        survives: dev recall 1.0 vs the trend channel's 0.667 at 5.6%
        machine-level FA. Sigma is floored by the median absolute
        successive difference over the WHOLE series — a 4-point
        baseline underestimates speed-swing scatter badly (31% FA)."""
        out = {}
        for r in self.reg:
            pts = r["pts"]
            if len(pts) <= warmup:
                continue
            base = np.array([y for _, y in pts[:warmup]])
            mu = float(np.median(base))
            sig = max(float(np.std(base)), 0.06)
            dif = np.diff([y for _, y in pts])
            if len(dif) >= 3:
                sig = max(sig, 1.4826 * float(np.median(np.abs(dif)))
                          / np.sqrt(2))
            s, fired_t, smax = 0.0, None, 0.0
            for t, y in pts[warmup:]:
                s = max(0.0, s + (y - mu) / sig - k_sigma)
                smax = max(smax, s)
                if s > h_sigma and fired_t is None:
                    fired_t = t
            out[str(r["id"])] = {"alarm": fired_t is not None,
                                 "t": fired_t, "stat": round(smax, 2)}
        return out

    def trends_grouped(self, adaptive=True):
        """Group-level view: a group alarms when ANY member alarms —
        a modulated bearing may grow in its fan while its tone sits
        still; they are the same physical source."""
        tds = [(_trend(r["pts"], adaptive=adaptive), r)
               for r in self.reg]
        out = []
        for idxs in self.groups():
            mem = [(tds[i][0], tds[i][1]) for i in idxs]
            out.append({
                "types": sorted({r["type"] for _, r in mem}),
                "ids": [str(r["id"]) for _, r in mem],
                "alarm": any(td["alarm"] for td, _ in mem
                             if td is not None),
                "n": max((td["n"] for td, _ in mem if td is not None),
                         default=0)})
        return out
