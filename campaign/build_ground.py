"""v2.12: production ground state of the corrected quasi-2D model.

eps_dd = 2.10, omega_z = 2pi x 103 Hz, n_peak = 2e21 m^-3 (164Dy constants),
triangular crystal at the stress-free spacing a = s * d_rot (s = 0.94 from
spacing_scan.py) in a commensurate cells x cells box.  Saves psi and the
parameters; reports the multi-start comparison against noise seeds.
"""
import sys, os, json, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from qtsim import EGPESolver, quasi2d_dipolar_symbol
from qtsim.lhy import q2d_parameters
from qtsim.minimize import minimize_energy
from qtsim.diagnostics import droplet_centers
import track_a_scan as ta
from regime_scan import roton, leggett

ap = argparse.ArgumentParser()
ap.add_argument("--eps", type=float, default=2.10)
ap.add_argument("--omega_z", type=float, default=103.0)
ap.add_argument("--n_peak", type=float, default=2e21)
ap.add_argument("--s", type=float, default=0.94)
ap.add_argument("--cells", type=int, default=12)
ap.add_argument("--dx", type=float, default=0.5)
ap.add_argument("--out", type=str, required=True)
ap.add_argument("--noise_seeds", type=int, nargs="*", default=[])
a = ap.parse_args()

p = q2d_parameters(a.eps, a.omega_z, a.n_peak)
kr, lmin = roton(a.eps, p["l_z"], p["gamma"])
d_rot = 4 * np.pi / (np.sqrt(3) * kr)
lat = a.s * d_rot
grid = ta.build_grid(lat, a.cells, a.dx)
Dk = quasi2d_dipolar_symbol(grid, p["l_z"])
Nt = float(grid.integrate(np.ones(grid.shape)))


def relax(seed):
    s = EGPESolver(grid, eps_dd=a.eps, gamma=p["gamma"], Dk=Dk)
    if seed == "lattice":
        s.psi = ta.lattice_seed(grid, lat, a.cells).astype(complex)
    else:
        rng = np.random.default_rng(seed)
        s.psi = (1 + 0.05 * rng.standard_normal(grid.shape)).astype(complex)
    s.psi *= np.sqrt(Nt / s.norm())
    s.step_imag(0.002, 2000, norm_target=Nt)
    info = minimize_energy(s, Nt, maxiter=3000, verbose=False)
    n = np.abs(s.psi) ** 2
    cen = droplet_centers(n, grid, lat)
    return s, dict(seed=str(seed), E=s.energy()["total"] / Nt, mu=info["mu"],
                   residual=float(info["residual"]), droplets=int(len(cen)),
                   contrast=float((n.max() - n.min()) / n.mean()),
                   nmin=float(n.min() / n.mean()), fs_x=leggett(n, 0), fs_y=leggett(n, 1))


s, r = relax("lattice")
np.save(a.out + ".npy", s.psi)
rows = [r]
for sd in a.noise_seeds:
    rows.append(relax(sd)[1])
meta = dict(eps=a.eps, omega_z=a.omega_z, n_peak=a.n_peak, **p, k_rot=kr, Lambda_min=lmin,
            d_rot=d_rot, s=a.s, lattice=lat, cells=a.cells, dx=a.dx, grid=list(grid.shape),
            box=list(grid.lengths), states=rows)
json.dump(meta, open(a.out + ".json", "w"), indent=1)
print(json.dumps(meta, indent=1))
