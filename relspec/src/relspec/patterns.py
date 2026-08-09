"""Pattern extraction and energy accounting.

Scope: find the dominant harmonic combs and modulation families in an order
spectrum, assign every bin's energy to at most one owner, and report the energy
each pattern carries. No diagnosis, no fault naming.

Why energy and not amplitude: a pattern is a set of bins, and the only quantity
that adds across a set without double counting is energy. Tracking a pattern
over time then means tracking one number per pattern per acquisition, and the
sum of those numbers plus the residual is exactly the spectrum's total energy.
That closure is the whole point: it makes "minimal loss" measurable as an energy
balance rather than a per-bin distance that nobody can interpret.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import maximum_filter1d
from .pipeline import CENTERS, ENV_CENTERS

# --------------------------------------------------------------- primitives
def _win(centers, order, tol):
    lo, hi = np.searchsorted(centers, order-tol), np.searchsorted(centers, order+tol)
    return lo, max(hi, lo+1)

def _tol_for(order, centers):
    """Tolerance follows the LOCAL bin width. The order axis is hybrid-Q: bins
    are 0.039 orders wide below 20x and 0.70 above it. Using the median width
    across the whole axis, as an earlier version did, is 18x too tight in the
    coarse region and every line above 20 orders is missed."""
    i = int(np.searchsorted(centers, order))
    i = min(max(i, 1), len(centers)-1)
    du = float(centers[i]-centers[i-1])
    return max(1.5*du, 0.012*order)

# ------------------------------------------------------------- comb finding
def find_combs(spec, centers=None, f_lo=0.5, f_hi=25.0, nh=10,
               min_snr=3.0, min_harmonics=5, max_combs=8, step=0.005):
    """Candidate fundamentals scored by the geometric mean of harmonic
    amplitudes over the local floor. Geometric mean, not sum: it demands that
    the whole comb be present rather than rewarding one loud line, which is what
    separates a real family from a single peak sitting near a multiple."""
    centers = CENTERS if centers is None else centers
    # A fundamental is only searchable if min_harmonics of its multiples fit on
    # the axis. The envelope rail spans 0-20 orders, so searching to 25 asked
    # for harmonics that cannot exist and the gate rejected everything: the rail
    # reported 100% residual on every acquisition.
    f_hi = min(f_hi, float(centers[-1])/max(min_harmonics, 1))
    if f_hi <= f_lo: return []
    pos = spec[spec > 0]
    floor = float(np.median(pos)) if pos.size else 1e-12
    floor = max(floor, 1e-12)
    smax = maximum_filter1d(spec, size=5, mode='nearest')
    grid = np.arange(f_lo, f_hi, step)
    logs = np.full((nh, len(grid)), -np.inf)
    ok = np.zeros((nh, len(grid)), dtype=bool)
    for h in range(1, nh+1):
        o = grid*h
        v = np.interp(o, centers, smax, left=0.0, right=0.0)
        v = np.where(o <= centers[-1], v, 0.0)
        ok[h-1] = v > 0
        logs[h-1] = np.log(np.maximum(v, 1e-12))
    n_ok = ok.sum(0)
    mean_log = np.where(ok, np.where(np.isfinite(logs), logs, -27.6), 0.0).sum(0)/np.maximum(n_ok, 1)
    # Count of harmonics genuinely above the floor is the precision gate. The
    # geometric-mean score alone accepts any fundamental whose multiples happen
    # to land on noise: on a single-comb signal that returned eight combs, four
    # of them artifacts, which then poisoned every downstream exclusion list.
    n_real = np.zeros(len(grid))
    for h in range(1, nh+1):
        o = grid*h
        v = np.interp(o, centers, smax, left=0.0, right=0.0)
        n_real += ((v > 4.0*floor) & (o <= centers[-1])).astype(float)
    score = np.where((n_ok >= 3) & (n_real >= min_harmonics),
                     np.exp(mean_log)/floor, 0.0)
    n_harm_grid = n_real

    at = lambda f: (score[int(round((f-f_lo)/step))]
                    if 0 <= int(round((f-f_lo)/step)) < len(score) else 0.0)
    out, used = [], np.zeros(len(grid), dtype=bool)
    for gi in np.argsort(-score):
        if score[gi] < min_snr: break
        if used[gi]: continue
        f = float(grid[gi])
        # Collapse to the lowest fundamental: a comb at 12x also scores at 6x,
        # 4x, 3x and 1x, and the family is named by its fundamental.
        #
        # But the test cannot be "the sub-multiple also scores well". A comb at
        # 1x makes 0.5x score well too, because every even harmonic of 0.5x is a
        # real line of 1x. Accepting that halves the fundamental and invents a
        # pattern. Require the sub-multiple's OWN harmonics, the ones the parent
        # does not supply, to be present above the floor.
        for k in (8, 6, 5, 4, 3, 2):
            g = f/k
            if g < f_lo or at(g) <= 0.75*score[gi]: continue
            extra = [g*j for j in range(1, k*3) if j % k != 0]
            if not extra: continue
            v = np.interp(extra, centers, smax, left=0.0, right=0.0)
            if np.mean(v > 3*floor) >= 0.6:
                f = g; break
        for k in range(1, nh+3):
            used |= np.abs(grid-f*k) < max(3*step, 0.02*f*k)
        # and its sub-multiples: a comb at f makes f/2 and f/3 score well from
        # the parent's own lines alone, which would report the same family twice
        for k in range(2, 7):
            used |= np.abs(grid-f/k) < max(3*step, 0.02*f/k)
        if any(abs(f-o['f0']) < 0.08*max(f, 1.0) for o in out): continue
        gj = int(round((f-f_lo)/step))
        out.append(dict(f0=round(f, 4), snr=round(float(at(f) or score[gi]), 3),
                        n_harm=int(n_harm_grid[gj]) if 0 <= gj < len(n_harm_grid) else 0))
        if len(out) >= max_combs: break

    # Exclusivity pass. A spurious fundamental scores well by parasitising the
    # lines of a stronger comb: 0.53 x 5 = 2.65 sits on top of a real line near
    # 2.6, and five such coincidences clear any harmonic-count gate. Accept a
    # comb only if a majority of its harmonics are lines no stronger comb has
    # already claimed. Without this the artifacts go on to own genuine
    # modulation sidebands and the modulation search sees one-sided pairs.
    kept, claimed = [], np.zeros(len(spec), dtype=bool)
    for c in out:
        own, tot = 0, 0
        idxs = []
        for h in range(1, nh+1):
            o = c['f0']*h
            if o > centers[-1]: break
            t = _tol_for(o, centers)
            L, H = _win(centers, o, t)
            if H <= L: continue
            seg = smax[L:H]
            if seg.max() <= 4.0*floor: continue
            tot += 1
            # exclusivity is about the PEAK, not the window. Windows of nearby
            # harmonics overlap, so "some bin here is unclaimed" is true almost
            # always and lets every parasite through. The question is whether
            # the line this harmonic is actually riding on is already spoken for.
            pk = L + int(np.argmax(seg))
            if not claimed[pk]: own += 1
            idxs.append((L, H, pk))
        if tot == 0: continue
        if kept and own/tot < 0.5:
            continue
        for L, H, pk in idxs: claimed[pk] = True
        c['exclusive'] = round(own/tot, 2)
        kept.append(c)
    return kept

# ------------------------------------------------------ comb verification
def verify_combs(spec, centers, combs, nh=10, floor_mult=4.0,
                 min_frac=0.5, min_energy_frac=0.55, keep_all=False):
    """Cloud-side second opinion on every comb the scorer proposed.

    find_combs scores a candidate on the MAXIMUM inside a tolerance window
    around each harmonic, and a window maximum is a low bar: the tail of a
    neighbouring line, one edge of a haystack, or plain noise all supply one.
    A parasitic fundamental only needs a handful of such coincidences.

    Verification asks three harder questions of every claimed harmonic:

      REAL PEAK    the window argmax must be a local maximum of the spectrum
                   itself, not the window edge riding a neighbour's slope;
      IN PLACE     that peak must sit where the harmonic predicts - within
                   half the search tolerance (or one bin, whichever is
                   looser), not merely somewhere inside the window;
      CARRYING     it must stand above the spectrum floor on its own.

    A comb survives only if at least half its claimed harmonics pass AND the
    verified harmonics carry most of the claimed energy - so a family cannot
    limp through on many empty windows, nor on one loud stolen line. The
    fundamental is then re-fit by least squares through the verified peak
    positions, which also tightens key stability across frames.

    keep_all=True returns rejected combs too (flagged), for inspection."""
    smax_pos = spec[spec > 0]
    floor = max(float(np.median(smax_pos)) if smax_pos.size else 1e-12, 1e-12)
    kept, rejected = [], []
    for c in combs:
        f0 = c['f0']
        claimed = ver = 0
        e_claim = e_ver = 0.0
        fit_num = fit_den = 0.0
        for h in range(1, nh+1):
            o = f0*h
            if o > centers[-1]: break
            tol = _tol_for(o, centers)
            L, H = _win(centers, o, tol)
            if H <= L: continue
            j = L+int(np.argmax(spec[L:H]))
            pk = float(spec[j])
            if pk <= floor_mult*floor: continue          # empty window: no claim
            claimed += 1; e_claim += pk*pk
            is_peak = (0 < j < len(spec)-1 and
                       spec[j] >= spec[j-1] and spec[j] >= spec[j+1])
            rides = ((j == L and j > 0 and spec[j-1] > spec[j]) or
                     (j == H-1 and j < len(spec)-1 and spec[j+1] > spec[j]))
            jl = min(max(j, 1), len(centers)-1)
            du = float(centers[jl]-centers[jl-1])
            centered = abs(float(centers[j])-o) <= max(0.5*tol, 1.2*du)
            if is_peak and not rides and centered:
                ver += 1; e_ver += pk*pk
                fit_num += h*float(centers[j]); fit_den += h*h
        ok = (claimed >= 3 and ver >= max(3, int(np.ceil(min_frac*claimed)))
              and e_ver >= min_energy_frac*max(e_claim, 1e-30))
        c = dict(c, ver_harm=ver, ver_claimed=claimed,
                 ver_frac=round(e_ver/max(e_claim, 1e-30), 3))
        if ok:
            if fit_den > 0:
                f0_fit = fit_num/fit_den
                if abs(f0_fit-f0) < 0.02*f0: c['f0'] = round(f0_fit, 4)
            kept.append(c)
        else:
            c['rejected'] = True; rejected.append(c)
    return (kept+rejected) if keep_all else kept

# ------------------------------------------------------- modulation finding
def find_modulation(spec, carrier, centers=None, dmin=0.15, dmax=2.2,
                    step=0.01, min_pair_frac=0.12, sb_spec=None,
                    exclude=(), floor=None, min_carrier_snr=6.0,
                    min_sb_snr=3.0):
    """sb_spec: the spectrum to look for sidebands in, which should be the
    comb-suppressed residual. exclude: spacings to reject, normally the comb
    fundamentals."""
    """Find the modulation spacing around a carrier by scoring symmetric pair
    presence directly, over a grid of candidate spacings.

    Autocorrelation was the obvious choice and it is the wrong one. Its window
    has to be several times the spacing to resolve anything, which means the
    window also contains neighbouring combs, and it locks onto whatever spacing
    dominates the window rather than the one modulating this carrier. Direct
    pair scoring asks the only question that matters: are there matched lines at
    c-kd and c+kd, for consecutive k, above a fraction of the carrier.

    It also rejects sub-multiples for free. At d/2 the k=1 pair is absent, so the
    score collapses, whereas an autocorrelation peak at d/2 looks plausible."""
    centers = CENTERS if centers is None else centers
    tol = _tol_for(carrier, centers)
    L, H = _win(centers, carrier, tol)
    # carrier amplitude comes from the full spectrum (the carrier is usually a
    # comb line), sidebands from the residual (they must be lines no comb owns)
    c_amp = float(spec[L:H].max()) if H > L else 0.0
    if c_amp <= 0: return None
    # Absolute gates. A purely relative pair test will always find something:
    # take a weak noise peak as the carrier and its neighbours clear any
    # fraction-of-carrier threshold. Both carrier and sidebands must also stand
    # above the noise floor in absolute terms.
    if floor is None:
        pos = spec[spec > 0]
        floor = float(np.median(pos)) if pos.size else 1e-12
    floor = max(floor, 1e-12)
    if c_amp < min_carrier_snr*floor: return None
    # The sideband gate must track the noise distribution's TAIL, not its
    # median. A pair test over a grid of spacings is a many-trials search, so
    # the relevant quantity is how high noise reaches somewhere, not where it
    # sits typically. Using the median made false pairs scale with bin count:
    # 5 false modulations at 512 bins became 16 at 4096, purely because finer
    # bins offer more noise excursions to pair up.
    base = sb_spec if sb_spec is not None else spec
    pos = base[base > 0]
    noise_hi = float(np.quantile(pos, 0.98)) if pos.size else floor
    sb_min = max(min_pair_frac*c_amp, min_sb_snr*floor, noise_hi)
    smax = maximum_filter1d(spec if sb_spec is None else sb_spec,
                            size=3, mode='nearest')

    def amp_at(o, dcap=None):
        """dcap caps the search window relative to the spacing under test.
        Without it the windows for adjacent sideband slots overlap: testing a
        spacing of 0.15 at order 8, where the tolerance is 0.096, lets the k=2
        slot reach a genuine sideband 0.25 away and score a wrong spacing using
        the right line. Tolerance has to shrink with the hypothesis."""
        if o <= 0 or o > centers[-1]: return 0.0
        t = _tol_for(o, centers)
        if dcap is not None: t = min(t, 0.35*dcap)
        a, b = _win(centers, o, t)
        return float(smax[a:b].max()) if b > a else 0.0

    def blocked(d):
        # A comb of spacing s produces symmetric neighbours at s around every
        # one of its own members, so without this every comb reports itself as
        # modulation of itself.
        # Only integer MULTIPLES of a comb spacing need excluding. A comb of
        # spacing s produces valid pairs at s, 2s, 3s around its own members,
        # but not at s/2: there is no line there to pair with. Excluding
        # sub-multiples as well was rejecting genuine half-spacing modulation,
        # which is one of the commonest patterns there is.
        for s in exclude:
            if s <= 0: continue
            for k in (1, 2, 3):
                if abs(d-s*k) < max(2*step, 0.05*s*k): return True
        return False

    best = None
    for d in np.arange(dmin, dmax, step):
        if blocked(d): continue
        sc, pairs, e = 0.0, 0, 0.0
        for k in range(1, 5):
            lo_a, hi_a = amp_at(carrier-k*d, d), amp_at(carrier+k*d, d)
            m = min(lo_a, hi_a)
            if m <= sb_min: break
            pairs += 1
            e += lo_a**2 + hi_a**2
            # geometric mean rewards symmetry; a strong line on one side only is
            # a separate component, not a modulation
            sc += np.sqrt(lo_a*hi_a)/c_amp / k
        if pairs == 0: continue
        if best is None or sc > best[1]:
            best = (float(d), sc, pairs, e)
    if best is None: return None
    d, sc, pairs, e = best
    # one pair is a coincidence; a modulation family has at least two
    if pairs < 2: return None
    # If the true spacing is s, a scan can lock onto 2s or 3s because those also
    # produce valid symmetric pairs. Only integer SUB-MULTIPLES of the winner
    # need re-testing. An earlier version scanned every smaller spacing and took
    # the first within 90%, which simply handed the answer to whatever noise sat
    # nearest dmin.
    def score_at(dd):
        sc2, pairs2, e2 = 0.0, 0, 0.0
        for k in range(1, 5):
            lo_a, hi_a = amp_at(carrier-k*dd, dd), amp_at(carrier+k*dd, dd)
            if min(lo_a, hi_a) <= sb_min: break
            pairs2 += 1; e2 += lo_a**2+hi_a**2
            sc2 += np.sqrt(lo_a*hi_a)/c_amp / k
        return sc2, pairs2, e2

    for div in (2, 3, 4):
        d2 = d/div
        if d2 < dmin or blocked(d2): continue
        sc2, pairs2, e2 = score_at(d2)
        if pairs2 >= 2 and sc2 >= 0.85*sc:
            d, sc, pairs, e = float(d2), sc2, pairs2, e2
            break

    return dict(carrier=round(float(carrier), 4), spacing=round(d, 4),
                n_pairs=pairs, ac_peak=round(float(sc), 4),
                carrier_amp=c_amp, sb_energy=e,
                mod_index=round(float(np.sqrt(e)/c_amp), 4))

# ------------------------------------------- isolated peaks and haystacks
def find_peaks_iso(spec, owner, centers, floor, min_prom_db=8.0, max_peaks=12):
    """Discrete lines no comb explains: a lone unbalance line, a vane or blade
    pass, a resonance tone. These carry real energy and are exactly the content
    a comb-only model throws into the residual and stops tracking."""
    v = spec.astype(np.float64)
    out = []
    for i in range(1, len(v)-1):
        if owner[i] >= 0: continue
        if not (v[i] >= v[i-1] and v[i] >= v[i+1]): continue
        nb = max(v[i-1], v[i+1])
        prom = 20*np.log10(max(v[i], 1e-12)/max(nb, 1e-12))
        snr = 20*np.log10(max(v[i], 1e-12)/floor)
        if snr < min_prom_db: continue
        out.append(dict(order=float(centers[i]), idx=i, amp=float(v[i]),
                        snr=round(snr, 2), prom=round(prom, 2)))
    out.sort(key=lambda d: -d['amp'])
    return out[:max_peaks]

def find_haystacks(spec, owner, centers, floor, min_width=6, min_rise_db=5.0,
                   max_stacks=6):
    """Broadband humps: a localised region lifted above the surrounding floor.
    Detected on a smoothed spectrum against a wide-median baseline, so a hump
    is a *region* that rose, not a line. This is the pattern that a peak-and-
    comb model is blindest to, and it is how lubrication distress, cavitation
    and early resonance excitation actually present."""
    v = np.log10(np.maximum(spec.astype(np.float64), 1e-12))
    n = len(v)
    k = max(5, n//64)
    sm = np.convolve(v, np.ones(k)/k, mode='same')
    wide = max(31, n//8)
    base = np.array([np.median(sm[max(0, i-wide//2):min(n, i+wide//2)])
                     for i in range(n)])
    rise = 20*(sm-base)
    mask = (rise > min_rise_db) & (owner < 0)
    out, i = [], 0
    while i < n:
        if not mask[i]: i += 1; continue
        j = i
        while j < n and mask[j]: j += 1
        if j-i >= min_width:
            seg = slice(i, j)
            out.append(dict(lo=float(centers[i]), hi=float(centers[j-1]),
                            centre=float(centers[(i+j)//2]), i0=i, i1=j,
                            width=j-i, rise_db=round(float(rise[seg].max()), 2),
                            energy=float((spec[seg]**2).sum())))
        i = j
    out.sort(key=lambda d: -d['energy'])
    return out[:max_stacks]

# ----------------------------------------------------- energy accounting
def account_energy(spec, combs, mods, centers=None, nh=10,
                   peaks=(), stacks=()):
    """Assign every bin's energy to at most one owner and close the balance.

    Ownership matters. Harmonics of a 1x comb land on top of harmonics of a 3x
    comb, and a modulation sideband can coincide with a harmonic. Summing per
    pattern without exclusive ownership double counts, the parts exceed the
    whole, and the residual goes negative. Bins are claimed strongest-first."""
    centers = CENTERS if centers is None else centers
    e_bin = spec.astype(np.float64)**2
    e_total = float(e_bin.sum())
    owner = np.full(len(spec), -1, dtype=np.int32)
    rows = []

    ranked = sorted(range(len(combs)), key=lambda i: -combs[i]['snr'])
    for pid in ranked:
        c = combs[pid]; claimed = []
        for h in range(1, nh+1):
            o = c['f0']*h
            if o > centers[-1]: break
            tol = _tol_for(o, centers)
            L, H = _win(centers, o, tol)
            idx = np.arange(L, min(H, len(spec)))
            idx = idx[owner[idx] < 0]
            if idx.size: owner[idx] = pid; claimed.append(idx)
        e = float(e_bin[np.concatenate(claimed)].sum()) if claimed else 0.0
        nb = int(sum(len(x) for x in claimed))
        rows.append(dict(kind='comb', key=c['f0'], f0=c['f0'], spacing=None,
                         snr=c['snr'], n_pairs=None, energy=e, n_bins=nb, pid=pid,
                         share=e/e_total if e_total > 0 else 0.0,
                         ver_harm=c.get('ver_harm'), ver_frac=c.get('ver_frac'),
                         ver_claimed=c.get('ver_claimed')))

    base = len(combs)
    for mi, m in enumerate(mods):
        pid = base+mi; claimed = []
        for k in range(1, m['n_pairs']+1):
            for sgn in (-1, 1):
                o = m['carrier']+sgn*k*m['spacing']
                if o <= 0 or o > centers[-1]: continue
                tol = _tol_for(o, centers)
                L, H = _win(centers, o, tol)
                idx = np.arange(L, min(H, len(spec)))
                idx = idx[owner[idx] < 0]
                if idx.size: owner[idx] = pid; claimed.append(idx)
        e = float(e_bin[np.concatenate(claimed)].sum()) if claimed else 0.0
        nb = int(sum(len(x) for x in claimed))
        rows.append(dict(kind='mod', key=round(m['carrier'], 2), f0=m['carrier'],
                         spacing=m['spacing'], snr=m['ac_peak'],
                         n_pairs=m['n_pairs'], energy=e, n_bins=nb, pid=pid,
                         share=e/e_total if e_total > 0 else 0.0))

    base2 = len(combs)+len(mods)
    for pi, pk in enumerate(peaks):
        i = pk['idx']
        lo, hi = max(0, i-1), min(len(spec), i+2)
        idx = np.arange(lo, hi); idx = idx[owner[idx] < 0]
        if idx.size == 0: continue
        owner[idx] = base2+pi
        e = float(e_bin[idx].sum())
        rows.append(dict(kind='peak', key=round(pk['order'], 3), f0=pk['order'],
                         spacing=None, snr=pk['snr'], n_pairs=None, energy=e,
                         n_bins=int(idx.size), pid=base2+pi,
                         share=e/e_total if e_total > 0 else 0.0))
    base3 = base2+len(peaks)
    for si, st in enumerate(stacks):
        idx = np.arange(st['i0'], st['i1']); idx = idx[owner[idx] < 0]
        if idx.size == 0: continue
        owner[idx] = base3+si
        e = float(e_bin[idx].sum())
        rows.append(dict(kind='haystack', key=round(st['centre'], 2),
                         f0=st['centre'], spacing=round(st['hi']-st['lo'], 3),
                         snr=st['rise_db'], n_pairs=st['width'], energy=e,
                         n_bins=int(idx.size), pid=base3+si,
                         share=e/e_total if e_total > 0 else 0.0))

    e_res = float(e_bin[owner < 0].sum())
    return dict(e_total=e_total, e_residual=e_res,
                residual_share=e_res/e_total if e_total > 0 else 1.0,
                patterns=rows, owner=owner)

def extract_patterns(spec, centers=None, max_combs=6, max_mods=6,
                     min_harmonics=5, f_hi=25.0, verify=True):
    """One call: combs, modulation around the strongest combs, energy balance.
    verify=True runs the cloud-side comb verification pass (argmax-coincidence
    + energy checks); the scorer alone over-reports on busy spectra."""
    centers = CENTERS if centers is None else centers
    combs = find_combs(spec, centers, f_hi=f_hi, max_combs=max_combs,
                       min_harmonics=min_harmonics)
    if verify:
        combs = verify_combs(spec, centers, combs,
                             min_frac=0.5 if min_harmonics >= 5 else 0.45)

    # Carrier candidates: harmonics of each comb up to the 6th, plus the
    # spectrum's own strongest lines. Modulation frequently sits on a mid
    # harmonic or on a resonance that no comb passes through, so probing only
    # h=1 and h=2 misses most of it.
    cands = []
    for c in combs[:4]:
        cands += [c['f0']*h for h in range(1, 7)]
    smax = maximum_filter1d(spec, size=3, mode='nearest')
    for i in np.argsort(-smax)[:24]:
        cands.append(float(centers[i]))
    # Suppress comb-owned bins before looking for modulation, so a sideband can
    # only be a line the combs did not already account for. Replace with the
    # local floor rather than zero: zeroing creates notches that the pair test
    # reads as structure.
    acc0 = account_energy(spec, combs, [], centers)
    pos = spec[spec > 0]
    floor = float(np.median(pos)) if pos.size else 1e-12
    resid_spec = np.where(acc0['owner'] >= 0, floor, spec)
    # only the dominant combs: excluding against weak noise-driven combs
    # rejects real spacings that happen to land near them
    excl = [c['f0'] for c in combs[:1]]

    seen, mods = [], []
    for car in cands:
        if car <= 0 or car > centers[-1]*0.9: continue
        if any(abs(car-s) < 0.06*max(car, 1.0) for s in seen): continue
        m = find_modulation(spec, car, centers, sb_spec=resid_spec,
                            exclude=excl, floor=floor)
        if m is None: continue
        seen.append(car)
        if any(abs(m['carrier']-x['carrier']) < 0.06*max(car, 1.0) for x in mods):
            continue
        mods.append(m)
        if len(mods) >= max_mods: break
    mods.sort(key=lambda x: -x['sb_energy'])
    # Claim what the combs and modulation did not: discrete lines first, then
    # broadband humps. Order matters, a peak inside a hump belongs to the peak.
    acc1 = account_energy(spec, combs, mods, centers)
    peaks = find_peaks_iso(spec, acc1['owner'], centers, floor)
    stacks = find_haystacks(spec, acc1['owner'], centers, floor)
    return combs, mods, account_energy(spec, combs, mods, centers,
                                       peaks=peaks, stacks=stacks)

# ------------------------------------------------- identity across time
def match_key(f0, known, tol_frac=0.04):
    """Stable pattern identity so energy can be tracked. A comb whose f0 drifts
    by a fraction of a percent between acquisitions is the same pattern; giving
    it a new key each time turns one trend into hundreds of one-point series."""
    best, err = None, 1e9
    for k in known:
        e = abs(f0-k)/max(k, 1e-9)
        if e < tol_frac and e < err: best, err = k, e
    return best
