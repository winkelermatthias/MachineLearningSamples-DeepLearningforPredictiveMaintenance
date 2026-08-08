"""Extraction v2: v1's pipeline generalised across rigs.

Differences from pipeline.extract, each forced by real data:

  - The demodulation band comes from a kurtogram, not a constant. The v1
    band (2-5 kHz) does not exist below a 5.12 kHz sample rate and misses
    MFPT's sub-2 kHz outer-race ringing. Chosen once per machine state and
    cached: band identity is an asset property, not an acquisition property.
  - Speed search runs in a window around the nominal speed when one is known
    (every real rig publishes one); the unconstrained 20-70 Hz search remains
    the fallback. This is what makes 25 Hz MFPT and 20 Hz SEU tractable with
    the same code that handles 29.95 Hz CWRU.
  - Welch nperseg scales with fs so spectral resolution is constant in Hz
    across rigs, keeping order-bin occupancy comparable.

The output is the same Extract dataclass: everything downstream (codecs,
patterns, gate) is rig-agnostic by construction.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import welch
from .pipeline import (Extract, EDGES, ENV_EDGES, NBANDS, NBINS, bin_orders,
                       moments, to_db, to_u8, coherence_limit)
from .dsp2 import kurtogram_band, envelope_banded
from .datasets import nperseg_for

_band_cache: dict = {}

def band_for(key, x, fs):
    """Kurtogram once per (machine state) key; every later acquisition of that
    state reuses the answer. Determinism matters: encoder and analyst must
    agree on what the envelope rail means."""
    if key not in _band_cache:
        _band_cache[key] = kurtogram_band(x, fs)[0]
    return _band_cache[key]

def estimate_speed2(x, fs, band, fr_nominal=None, span=0.35, nh=3):
    """Envelope-domain HPS with parabolic refinement, searched over
    [nominal*(1-span), nominal*(1+span)] when a nominal is known."""
    e = envelope_banded(x, fs, band)
    N = 1 << int(np.ceil(np.log2(len(e))))
    A = np.abs(np.fft.rfft(e*np.hanning(len(e)), N))*2/len(e)
    f = np.fft.rfftfreq(N, 1/fs)
    lo, hi = ((1-span)*fr_nominal, (1+span)*fr_nominal) if fr_nominal \
        else (20.0, 70.0)
    grid = np.arange(max(lo, 2.0), hi, 0.005)
    hps = np.ones_like(grid)
    for h in range(1, nh+1):
        hps *= np.maximum(np.interp(grid*h, f, A), 1e-12)
    hps = hps**(1/nh)
    j = int(np.argmax(hps)); fr = float(grid[j])
    rival = hps[np.abs(grid-grid[j]) > 0.5]
    conf = float(hps[j]/max(rival.max() if rival.size else 1e-12, 1e-12))
    k = int(round(fr/(f[1]-f[0])))
    if 0 < k < len(A)-1:
        y0, y1, y2 = A[k-1], A[k], A[k+1]
        den = 2*(y0-2*y1+y2)
        if abs(den) > 1e-20:
            d = (y0-y2)/den
            if -1 < d < 1: fr = float((k+d)*(f[1]-f[0]))
    return fr, conf

def acc_comb_speed(f, a, fr_nominal, span=0.35, nh=6):
    """Speed from the ACCELERATION harmonic comb. The envelope estimator is
    blind on a healthy machine - no impacts, no impact modulation, no shaft
    comb in the envelope - which is most of a fleet on most days. But every
    rotating machine drives 1x..Nx into the casing whether or not anything
    is wrong. Searched only near the nominal, with the product over six
    harmonics, so a single mount line cannot win: it would need five
    accomplices at exact multiples."""
    grid = np.arange(max(2.0, (1-span)*fr_nominal), (1+span)*fr_nominal, 0.005)
    if not len(grid): return fr_nominal, 0.0
    sc = np.ones_like(grid)
    for h in range(1, nh+1):
        sc *= np.maximum(np.interp(grid*h, f, a), 1e-12)
    sc = sc**(1/nh)
    j = int(np.argmax(sc)); fr = float(grid[j])
    if 0 < j < len(sc)-1:
        y0, y1, y2 = sc[j-1], sc[j], sc[j+1]
        den = 2*(y0-2*y1+y2)
        if abs(den) > 1e-20:
            d = (y0-y2)/den
            if -1 < d < 1: fr = float(grid[j]+d*0.005)
    # The rival exclusion zone must clear the WELCH MAIN LOBE, not a fixed
    # 0.5 Hz. At 2.93 Hz bins the peak's own shoulders extend +/-3 Hz on the
    # interpolated grid; measuring the "rival" inside the same lobe pins
    # confidence at ~1 and the estimate can never be believed.
    excl = max(2.5*(f[1]-f[0]), 0.04*fr_nominal)
    rival = sc[np.abs(grid-grid[j]) > excl]
    conf = float(sc[j]/max(rival.max() if rival.size else 1e-12, 1e-12))
    return fr, conf

def extract2(x, fs, band_key='default', fr_nominal=None,
             fr_override=None) -> Extract:
    """Rig-agnostic extraction. fr_override lets a tracker (Viterbi, order
    tracking) supply the speed; the estimator's answer is still computed so
    confidence tiers stay meaningful."""
    nps = nperseg_for(fs)
    f, p = welch(x, fs=fs, nperseg=min(nps, len(x)),
                 noverlap=min(nps, len(x))//2, window='hann',
                 scaling='spectrum', detrend='constant')
    a = np.sqrt(np.maximum(p, 0))*np.sqrt(2)

    m = (f >= 10) & (f <= 1000)
    v = a[m]*9.80665/(2*np.pi*np.maximum(f[m], 1e-9))*1000.0
    vel_rms = float(np.sqrt(0.5*np.sum(v**2)))

    band = band_for(band_key, x, fs)
    fr_e, conf_e = estimate_speed2(x, fs, band, fr_nominal)
    if fr_nominal:
        # DUAL EVIDENCE. The envelope comb is sharp but treacherous: on an
        # outer-race fault the envelope contains a BPFO-spaced comb and no
        # shaft comb at all, and BPFO/3 (1.19x for a 6205) lands inside the
        # search span and wins with GOOD confidence. A confidence gate cannot
        # catch a confidently wrong estimator; only independent evidence can.
        # The acceleration comb defines what "shaft speed" means, so the
        # envelope answer is accepted only when the acc comb corroborates it.
        fr_a, conf_a = acc_comb_speed(f, a, fr_nominal)
        agree = conf_a >= 1.3 and abs(fr_e-fr_a) < 0.03*fr_a
        if conf_e >= 1.8 and (agree or conf_a < 1.3):
            fr, conf = fr_e, conf_e
            tier = 1 if (conf_e >= 3.0 and agree) else 2
        elif conf_a >= 1.3:
            fr, conf = fr_a, conf_a
            tier = 2 if conf_a >= 3.0 else 3
        else:
            fr, conf, tier = fr_nominal, min(conf_e, conf_a), 3
    else:
        fr, conf = fr_e, conf_e
        tier = 1 if conf >= 3.0 else (2 if conf >= 1.8 else 3)
    if not (2 < fr < 300): fr, tier = (fr_nominal or 30.0), 3
    if fr_override is not None: fr = float(fr_override)

    e = envelope_banded(x, fs, band)
    fe, pe = welch(e, fs=fs, nperseg=min(nps, len(e)),
                   noverlap=min(nps, len(e))//2, window='hann',
                   scaling='spectrum', detrend='constant')
    ae = np.sqrt(np.maximum(pe, 0))*np.sqrt(2)

    acc = bin_orders(f, a, fr, EDGES)
    env = bin_orders(fe, ae, fr, ENV_EDGES)
    ar, ak, ac = moments(x); er, ek, ec = moments(e)
    bandmap = np.minimum((np.arange(NBINS)*NBANDS)//NBINS, NBANDS-1)
    band_db = np.array([to_db(np.sqrt(np.mean(acc[bandmap == b]**2)))
                        for b in range(NBANDS)])
    speed_err = 0.0005 if tier == 1 else (0.003 if tier == 2 else 0.02)
    return Extract(acc, env, to_u8(acc), to_u8(env), fr, conf, tier,
                   coherence_limit(3.0, speed_err, fr),
                   ar, vel_rms, ak, ac, er, ek, ec, band_db)
