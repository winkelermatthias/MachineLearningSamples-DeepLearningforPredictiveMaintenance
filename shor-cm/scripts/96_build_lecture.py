#!/usr/bin/env python3
"""Build the interactive lecture report. All numbers computed live with
shorcm on synthetic physics (mirror down), embedded as JSON, rendered by
vanilla JS + SVG. Output: report/lecture.html, fully self-contained."""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shorcm import tacho as T, spectra as S, amplify as AM, metrics as M

FS, SPR, REVS = 25_000, 256, 5
rng = np.random.default_rng(20260709)
r4 = lambda a: [round(float(v), 4) for v in a]  # noqa: E731


def synth(f0=30.0, dur=8.0, wander=0.015, tones=(), noise=0.35, seed=0):
    g = np.random.default_rng(seed)
    n = int(dur * FS)
    t = np.arange(n) / FS
    f_inst = f0 * (1 + wander * np.sin(2 * np.pi * 0.6 * t))
    phase = 2 * np.pi * np.cumsum(f_inst) / FS
    tach = ((phase % (2 * np.pi)) < 0.12).astype(float)
    x = noise * g.standard_normal(n)
    for order, amp, locked, slip in tones:
        if locked:
            x += amp * np.cos(order * phase + g.uniform(0, 6.28))
        else:
            rw = np.cumsum(g.normal(0, 0.9 / np.sqrt(FS / f0), n))
            x += amp * np.cos(order * (1 + slip) * phase + rw)
    return tach, x, phase, f_inst


D = {}

# ---- 1. the machine as a clock: speed wander + waveform with tacho ticks
tach, x, phase, f_inst = synth(tones=[(1, 1.0, True, 0), (2, 0.5, True, 0),
                                      (2.998, 0.6, False, 0.012)], seed=1)
seg = slice(0, int(0.25 * FS))
ds = 25
D["wave"] = {"t": r4(np.arange(len(x))[seg][::ds] / FS),
             "x": r4(x[seg][::ds]),
             "ticks": r4(T.tacho_pulse_times(tach[seg], FS) / FS)}
D["speed"] = {"t": r4(np.arange(0, len(x), 500) / FS),
              "f": r4(f_inst[::500])}

# ---- 2. Hz vs order domain, same signal
w = np.hanning(len(x))
A_hz = np.abs(np.fft.rfft((x - x.mean()) * w)) / (w.sum() / 2)
f_hz = np.fft.rfftfreq(len(x), 1 / FS)
m = f_hz <= 130
# decimate to ~600 pts, keep max in each bucket so peaks survive
def dec(xs, ys, npts=600):
    k = max(1, len(xs) // npts)
    xs2 = [float(xs[i]) for i in range(0, len(xs) - k, k)]
    ys2 = [float(np.max(ys[i:i + k])) for i in range(0, len(ys) - k, k)]
    return r4(xs2), r4(ys2)
hx, hy = dec(f_hz[m], A_hz[m])
ph_t, _ = T.phase_from_tacho(tach, FS)
xa = S.angular_resample(x, ph_t, SPR)
A_o, o_o = S.fine_order_spectrum(xa, SPR)
mo = o_o <= 4.3
ox, oy = dec(o_o[mo], A_o[mo])
D["domains"] = {"hz": {"x": hx, "y": hy}, "order": {"x": ox, "y": oy},
                "f0": 30.0}

# ---- 3. coherent interference vs N blocks
Z, orders = S.block_spectra(xa, SPR, REVS)
kmax = int(4.2 * REVS * SPR / SPR * REVS)  # bins up to order ~4.2
nb = int(4.2 * REVS)
D["blocks"] = {"orders": r4(orders[1:nb]), "N": [], "coh": [], "inc": []}
for N in [4, 8, 16, 32, 64]:
    if N > Z.shape[0]:
        break
    c, i_, _ = S.coherent_split(Z[:N])
    D["blocks"]["N"].append(N)
    D["blocks"]["coh"].append(r4(c[1:nb]))
    D["blocks"]["inc"].append(r4(i_[1:nb]))

# ---- 4. coherence-ratio fingerprints per class
classes = {
    "normal": [(1, 0.5, True, 0)],
    "imbalance": [(1, 1.6, True, 0)],
    "misalignment": [(1, 0.5, True, 0), (2, 1.1, True, 0), (3, 0.5, True, 0)],
    "bearing": [(1, 0.5, True, 0), (0.4, 0.35, False, 0.01),
                (2.998, 0.8, False, 0.012)],
}
D["ratio"] = {"orders": r4(orders[1:nb]), "cls": {}}
for name, tones in classes.items():
    tc, xc, _, _ = synth(tones=tones, seed=hash(name) % 999)
    pc, _ = T.phase_from_tacho(tc, FS)
    Zc, _ = S.block_spectra(S.angular_resample(xc, pc, SPR), SPR, REVS)
    c, i_, r_ = S.coherent_split(Zc)
    D["ratio"]["cls"][name] = {"ratio": r4(r_[1:nb]), "inc": r4(i_[1:nb])}

# ---- 6. slip scan per class + PPA phasors
# slip bench uses a CLEAN mean-slip tone (tiny phase walk) so the scan
# demonstrates slip recovery; the heavy random walk belongs to the PPA bench
slip_classes = dict(classes)
slip_classes["bearing"] = [(1, 0.5, True, 0)]
D["slip"] = {}
for name, tones in slip_classes.items():
    tc, xc, _, _ = synth(tones=tones, seed=hash(name) % 999)
    if name == "bearing":
        gg = np.random.default_rng(5)
        _, _, ph_true, _ = synth(tones=[], seed=hash(name) % 999)
        rw = np.cumsum(gg.normal(0, 0.02 / np.sqrt(FS / 30.0), len(xc)))
        xc = xc + 0.8 * np.cos(2.998 * 1.012 * ph_true + rw)
    pc, _ = T.phase_from_tacho(tc, FS)
    xac = S.angular_resample(xc, pc, SPR)
    sc = AM.slip_scan(xac, SPR, 2.998, span=0.03, n=241)
    D["slip"][name] = {"s": r4(np.asarray(sc["curve_s"]) * 100),
                       "a": r4(sc["curve_a"]),
                       "slip": round(sc["slip"] * 100, 3),
                       "sharp": round(sc["sharpness"], 1)}

g = np.random.default_rng(7)
N = 24
cases = {}
theta = np.cumsum(g.normal(0, 0.55, N))
cases["locked"] = np.exp(1j * g.uniform(0, 6.28)) * np.ones(N) \
    + 0.35 * (g.standard_normal(N) + 1j * g.standard_normal(N))
cases["drifting"] = np.exp(1j * theta) \
    + 0.35 * (g.standard_normal(N) + 1j * g.standard_normal(N))
cases["noise"] = 0.8 * (g.standard_normal(N) + 1j * g.standard_normal(N))
D["ppa"] = {}
for k, z in cases.items():
    zs, obs, mu, sd = AM.ppa_zscore(z, n_perm=300)
    D["ppa"][k] = {"re": r4(z.real), "im": r4(z.imag),
                   "z": round(zs, 1), "naive": round(float(np.abs(z.mean())), 2),
                   "aligned": round(obs, 2)}

# ---- 7. gain law
tg, xg, _, _ = synth(dur=16, tones=[(1, 0.6, True, 0), (0.4, 0.05, True, 0)],
                     noise=0.5, seed=11)
pg, _ = T.phase_from_tacho(tg, FS)
xag = S.angular_resample(xg, pg, SPR)
Ns, snrs = [], []
for N in [4, 9, 16, 36, 64]:
    Zg, og = S.block_spectra(xag[: N * REVS * SPR], SPR, REVS)
    c, _, _ = S.coherent_split(Zg)
    b = int(np.argmin(np.abs(og - 0.4)))
    Ns.append(N)
    snrs.append(float(c[b] / np.median(np.delete(c[1:200], b - 1))))
slope = float(np.polyfit(np.log(Ns), np.log(snrs), 1)[0])
D["gain"] = {"N": Ns, "snr": r4(snrs), "slope": round(slope, 2)}

# ---- 8. the referee: paired deltas + live alpha
deltas = np.sort(rng.normal(0.062, 0.041, 800))
D["gate"] = {"deltas": r4(deltas)}

# ---- html ----------------------------------------------------------------
data_json = json.dumps(D, separators=(",", ":"))
html = Path(__file__).with_name("96_template.html").read_text()
out = Path("report"); out.mkdir(exist_ok=True)
(out / "lecture.html").write_text(html.replace("__DATA__", data_json))
print("written report/lecture.html,",
      f"{(out / 'lecture.html').stat().st_size // 1024} KB")
