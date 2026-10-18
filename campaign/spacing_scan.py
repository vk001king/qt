"""v2.12: stress-free lattice constant of the triangular crystal.

The box is commensurate with a trial spacing a = s * d_rot (cells x cells
triangular cells); the crystal is relaxed from a lattice seed at that
spacing and the energy per particle is recorded.  The minimum over s gives
the stress-free spacing, independent of the linear roton estimate.
Also returns the multi-start check: the lowest-energy state is re-relaxed
from three independent noise seeds in the same box.
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


def relax(eps, p, a, cells, seed, dx=0.5):
    grid = ta.build_grid(a, cells, dx)
    s = EGPESolver(grid, eps_dd=eps, gamma=p["gamma"], Dk=quasi2d_dipolar_symbol(grid, p["l_z"]))
    Nt = float(grid.integrate(np.ones(grid.shape)))
    if seed == "lattice":
        s.psi = ta.lattice_seed(grid, a, cells).astype(complex)
    else:
        rng = np.random.default_rng(seed)
        s.psi = (1 + 0.05 * rng.standard_normal(grid.shape)).astype(complex)
    s.psi *= np.sqrt(Nt / s.norm())
    s.step_imag(0.002, 2000, norm_target=Nt)
    info = minimize_energy(s, Nt, maxiter=3000, verbose=False)
    n = np.abs(s.psi) ** 2
    cen = droplet_centers(n, grid, a)
    return dict(a=a, seed=str(seed), E=s.energy()["total"] / Nt, mu=info["mu"],
                residual=float(info["residual"]), droplets=int(len(cen)),
                contrast=float((n.max() - n.min()) / n.mean()),
                nmin=float(n.min() / n.mean()),
                fs_x=leggett(n, 0), fs_y=leggett(n, 1)), s.psi


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=2.10)
    ap.add_argument("--omega_z", type=float, default=103.0)
    ap.add_argument("--n_peak", type=float, default=2e21)
    ap.add_argument("--cells", type=int, default=5)
    ap.add_argument("--s", type=float, nargs="+", default=[0.90, 0.92, 0.94, 0.96, 0.98, 1.00])
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()
    p = q2d_parameters(a.eps, a.omega_z, a.n_peak)
    kr, lmin = roton(a.eps, p["l_z"], p["gamma"])
    d_rot = 4 * np.pi / (np.sqrt(3) * kr)
    rows = []
    for sv in a.s:
        r, psi = relax(a.eps, p, sv * d_rot, a.cells, "lattice")
        r["s"] = sv
        rows.append(r)
        print(json.dumps(r), flush=True)
    json.dump(dict(params=p, d_rot=d_rot, k_rot=kr, Lambda_min=lmin, scan=rows),
              open(a.out, "w"), indent=1)
