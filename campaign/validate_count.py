"""v2.12: validation of the interstitial winding count on controlled fields.

All tests imprint KNOWN vortex configurations (qtsim.imprint) on the
production ground state.  Ground truth for N_lat is the sum over Delaunay
triangles of |net charge of the vortices placed inside that triangle|
(opposite charges in one triangle cancel by construction of the observable);
the total vortex number is reported alongside.

Tests
  V1  random interstitial placements, N = 10 ... 280, one vortex per site
  V2  random uniform placements anywhere (inside droplets too)
  V3  close +/- pairs at separation 1, 2, 3, 5 xi
  V4  doubly occupied triangles (two same-sign vortices in one triangle)
  V5  vortex-free sound: smooth random phase imprint of increasing amplitude
  V6  sensitivity to smoothing width and peak window
  V7  after 2 tau of free evolution (cores healed), recount
"""
import sys, os, json
import numpy as np
from scipy.spatial import Delaunay
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from qtsim import EGPESolver, quasi2d_dipolar_symbol
from qtsim.diagnostics import (plaquette_charges_2d, droplet_centers,
                               interstitial_charges, link_coherence)
from qtsim.imprint import imprint
import track_a_scan as ta

meta = json.load(open("/home/claude/runs2/gs_dx05.json"))
psi0 = np.load("/home/claude/runs2/gs_dx05.npy")
lat = meta["lattice"]
grid = ta.build_grid(lat, meta["cells"], meta["dx"])
L = np.array(grid.lengths)
n0 = np.abs(psi0) ** 2
cen0 = droplet_centers(n0, grid, lat)
tiles = np.concatenate([cen0 + L * np.array([u, v]) for u in (-1, 0, 1) for v in (-1, 0, 1)])
tri = Delaunay(tiles)
tc = tiles[tri.simplices].mean(axis=1)
inside = np.all((tc >= 0) & (tc < L), axis=1)
sites = tc[inside]
rng = np.random.default_rng(12345)


def truth(xs, ys, qs):
    """Sum over triangles of |net charge| for the ground-state triangulation."""
    pts = np.mod(np.c_[xs, ys], L)
    s = tri.find_simplex(pts)
    c = np.mod(tc[s], L)
    key = np.round(c, 6)
    net = {}
    for k, q in zip(map(tuple, key), qs):
        net[k] = net.get(k, 0) + q
    return int(sum(abs(v) for v in net.values()))


def count(psi, smooth=1.5, win=None):
    n = np.abs(psi) ** 2
    q = plaquette_charges_2d(psi)
    if win is None:
        cen = droplet_centers(n, grid, lat, smooth=smooth)
    else:
        cen = droplet_centers(n, grid, lat * win / 0.6, smooth=smooth)
    w, _ = interstitial_charges(q, cen, grid)
    return int(np.abs(w).sum()), int(len(cen)), int((q != 0).sum())


def neutral_signs(N):
    qs = np.array([1] * (N // 2) + [-1] * (N // 2))
    rng.shuffle(qs)
    return qs


res = {}
# V1 interstitial placement
v1 = []
for N in (10, 40, 80, 120, 160, 200, 240, 280):
    for rep in range(2):
        idx = rng.choice(len(sites), N, replace=False)
        jit = rng.normal(0, 0.5, (N, 2))
        xs, ys = (sites[idx] + jit).T
        qs = neutral_signs(N)
        psi = imprint(psi0, grid, xs, ys, qs)
        nl, nd, raw = count(psi)
        v1.append(dict(N=N, truth=truth(xs, ys, qs), N_lat=nl, droplets=nd, raw_plaquette=raw))
res["V1_interstitial"] = v1
# V2 uniform placement anywhere
v2 = []
for N in (40, 120, 200):
    xs = rng.uniform(0, L[0], N); ys = rng.uniform(0, L[1], N); qs = neutral_signs(N)
    psi = imprint(psi0, grid, xs, ys, qs)
    nl, nd, raw = count(psi)
    v2.append(dict(N=N, truth=truth(xs, ys, qs), N_lat=nl, droplets=nd, raw_plaquette=raw))
res["V2_uniform"] = v2
# V3 close pairs (each pair alone in the box, centred on a site, along x)
v3 = []
for sep in (1.0, 2.0, 3.0, 5.0, 8.0):
    c = sites[rng.integers(len(sites))]
    xs = np.array([c[0] - sep / 2, c[0] + sep / 2]); ys = np.array([c[1], c[1]]); qs = np.array([1, -1])
    psi = imprint(psi0, grid, xs, ys, qs)
    nl, nd, raw = count(psi)
    v3.append(dict(separation=sep, truth=truth(xs, ys, qs), N_lat=nl, raw_plaquette=raw))
res["V3_close_pairs"] = v3
# V4 doubly occupied triangles: 20 triangles with two same-sign vortices (+ 20 with -2)
idx = rng.choice(len(sites), 40, replace=False)
xs, ys, qs = [], [], []
for k, i in enumerate(idx):
    sgn = 1 if k < 20 else -1
    for off in (-1.2, 1.2):
        xs.append(sites[i][0] + off); ys.append(sites[i][1]); qs.append(sgn)
xs, ys, qs = map(np.array, (xs, ys, qs))
psi = imprint(psi0, grid, xs, ys, qs)
nl, nd, raw = count(psi)
res["V4_double"] = dict(vortices=80, truth=truth(xs, ys, qs), N_lat=nl, raw_plaquette=raw)
# V5 vortex-free smooth phase (sound-like) imprint
v5 = []
kx = 2 * np.pi * np.fft.fftfreq(grid.shape[0], grid.dx[0])
ky = 2 * np.pi * np.fft.fftfreq(grid.shape[1], grid.dx[1])
K = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
for A in (0.5, 1.0, 2.0, 4.0, 8.0):
    spec = np.where((K > 0.2) & (K < 1.5), 1.0 / np.maximum(K, 1e-9), 0.0)
    ph = np.real(np.fft.ifft2(spec * np.exp(2j * np.pi * rng.random(K.shape))))
    ph *= A / ph.std()
    psi = psi0 * np.exp(1j * ph)
    nl, nd, raw = count(psi)
    lc = link_coherence(psi, droplet_centers(np.abs(psi) ** 2, grid, lat), grid)
    v5.append(dict(phase_rms=A, N_lat=nl, raw_plaquette=raw, coherence=lc["cos"]))
res["V5_sound_no_vortices"] = v5
# V6 sensitivity of one N=160 interstitial configuration
idx = rng.choice(len(sites), 160, replace=False)
xs, ys = (sites[idx] + rng.normal(0, 0.5, (160, 2))).T; qs = neutral_signs(160)
psi = imprint(psi0, grid, xs, ys, qs)
tr = truth(xs, ys, qs)
v6 = []
for sm in (1.0, 1.5, 2.0, 2.5):
    for win in (0.5, 0.6, 0.7):
        nl, nd, raw = count(psi, smooth=sm, win=win)
        v6.append(dict(smooth=sm, window=win, N_lat=nl, droplets=nd))
res["V6_sensitivity"] = dict(truth=tr, rows=v6)
# V7 after 2 tau of real-time evolution
s = EGPESolver(grid, eps_dd=meta["eps"], gamma=meta["gamma"], Dk=quasi2d_dipolar_symbol(grid, meta["l_z"]))
s.psi = psi.copy()
before = count(s.psi)[0]
s.step_real(0.005, 400)
after = count(s.psi)[0]
res["V7_after_2tau"] = dict(truth_initial=tr, N_lat_initial=before, N_lat_after_2tau=after)
json.dump(res, open("/home/claude/runs2/validate_count.json", "w"), indent=1)
print(json.dumps(res, indent=1))
