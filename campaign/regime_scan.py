"""v2.12 regime scan: ground states of the corrected quasi-2D model.

For each eps_dd at fixed physical (omega_z, n_peak), derive (l_z, gamma) with
qtsim.lhy.q2d_parameters, find the roton, build a commensurate 5-cell box,
relax from a lattice seed (imaginary time + L-BFGS-B) and from a uniform
seed with noise, and report: energy per particle of the modulated and
uniform states, contrast, n_min/<n>, and Leggett bounds on the superfluid
fraction along x and y.
"""
import sys, os, json, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qtsim import EGPESolver, quasi2d_dipolar_symbol, quasi2d_dipolar_profile
from qtsim.lhy import q2d_parameters
from qtsim.minimize import minimize_energy
sys.path.insert(0, os.path.dirname(__file__))
import track_a_scan as ta


def roton(eps, l, gam):
    k = np.linspace(1e-5, 6, 60000)
    lam = k ** 2 + 2 * (1 + eps * quasi2d_dipolar_profile(k, l)) + 3 * gam
    i = int(np.argmin(lam[1:])) + 1
    return float(k[i]), float(lam[i])


def leggett(n, axis):
    """Upper and lower Leggett bounds for superflow along `axis` (0=x, 1=y)."""
    other = 1 - axis
    nbar = n.mean()
    # upper: average over the transverse direction first, harmonic mean along
    # the flow; lower: harmonic mean along the flow first, then average over
    # the transverse direction (Leggett; Chauveau et al. 2023)
    up = 1.0 / (nbar * np.mean(1.0 / n.mean(axis=other)))
    lo = np.mean(1.0 / np.mean(1.0 / n, axis=axis)) / nbar
    return float(up), float(lo)


def ground(eps, wz, npk, cells=5, dx=0.5, seed="lattice", maxiter=2500):
    p = q2d_parameters(eps, wz, npk)
    kr, lmin = roton(eps, p["l_z"], p["gamma"])
    d = 4 * np.pi / (np.sqrt(3) * kr)
    grid = ta.build_grid(d, cells, dx)
    s = EGPESolver(grid, eps_dd=eps, gamma=p["gamma"], Dk=quasi2d_dipolar_symbol(grid, p["l_z"]))
    Nt = float(grid.integrate(np.ones(grid.shape)))
    if seed == "lattice":
        s.psi = ta.lattice_seed(grid, d, cells).astype(complex)
    else:
        rng = np.random.default_rng(1)
        s.psi = (1 + 0.05 * rng.standard_normal(grid.shape)).astype(complex)
    s.psi *= np.sqrt(Nt / s.norm())
    s.step_imag(0.002, 2000, norm_target=Nt)
    info = minimize_energy(s, Nt, maxiter=maxiter, verbose=False)
    n = np.abs(s.psi) ** 2
    E = s.energy()["total"] / Nt
    psi_mod = s.psi.copy()
    s.psi = np.ones(grid.shape, complex) * np.sqrt(Nt / float(grid.integrate(np.ones(grid.shape))))
    Eu = s.energy()["total"] / Nt
    ux, lx = leggett(n, 0)
    uy, ly = leggett(n, 1)
    return dict(eps=eps, omega_z=wz, n_peak=npk, seed=seed, **p, k_rot=kr, Lambda_min=lmin,
                d_rot=d, mu=info["mu"], residual=float(info["residual"]),
                E_mod=E, E_uniform=Eu, contrast=float((n.max() - n.min()) / n.mean()),
                nmin_over_mean=float(n.min() / n.mean()),
                leggett_x=(ux, lx), leggett_y=(uy, ly)), psi_mod, grid


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, nargs="+", default=[1.90, 1.95, 2.00, 2.05, 2.10])
    ap.add_argument("--omega_z", type=float, default=103.0)
    ap.add_argument("--n_peak", type=float, default=2e21)
    ap.add_argument("--out", type=str, default="regime_scan.json")
    a = ap.parse_args()
    rows = []
    for e in a.eps:
        for sd in ("lattice", "uniform"):
            r, psi, grid = ground(e, a.omega_z, a.n_peak, seed=sd)
            np.save(os.path.splitext(a.out)[0] + "_eps%.2f_%s.npy" % (e, sd), psi)
            from qtsim.diagnostics import droplet_centers
            cen = droplet_centers(np.abs(psi) ** 2, grid, r["d_rot"])
            r["n_droplets"] = int(len(cen))
            rows.append(r)
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)
