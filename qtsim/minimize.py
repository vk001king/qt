"""Direct energy minimisation for eGPE ground states.

WHY THIS EXISTS
---------------
Imaginary-time gradient flow is robust but it is only steepest descent.
For a dipolar droplet crystal the energy landscape has many nearby minima
separated by small barriers, and gradient flow from a single seed lands in
whichever basin it happens to start in.  Measured symptoms (v1.5):

  * residual plateaus at 2.3e-2 and will not improve with 24000 further
    steps at dtau down to 5e-5;
  * a perfect-lattice seed reaches mu = 5.233 while a noise seed reaches
    5.509, so neither is the minimum;
  * in real time the droplets keep their shape (contrast stable to 1.3e-3,
    peak density fixed) and barely translate (0.35 percent of a lattice
    constant) but the Bragg amplitudes change by 47 percent -- the
    droplets are rearranging, i.e. hunting for a better arrangement.

So the obstacle is optimisation, not physics.  This module minimises the
energy functional directly, under the norm constraint, with L-BFGS-B, and
supports a multi-start ensemble that keeps the lowest chemical potential.

FORMULATION
-----------
Minimise E[psi] subject to  int |psi|^2 dV = N.  The constraint is handled
by projection: the objective is evaluated on the rescaled field
    psi_hat = sqrt(N / ||psi||^2) psi,
so E is a function on the constraint manifold and the optimiser can move
freely.  The gradient of E with respect to psi* is
    dE/dpsi* = H[psi] psi = (-Lap + V + n + eps D[n] + gamma n^{3/2}) psi,
i.e. the same operator the solver already applies.  For the projected
objective the chain rule adds the radial component removal
    grad_proj = (grad - (Re<psi_hat, grad> / N) psi_hat) * sqrt(N)/||psi||,
which is the tangential part.  Real and imaginary parts are packed into a
single real vector for scipy.

BACKEND.  `scipy.optimize.minimize` is CPU-only, so the packed vector
handed to it is always a host (NumPy) array.  Everything expensive --
FFTs, the nonlinear terms, the energy functional -- runs through
`solver.g.xp`/`solver.g.fft_mod`, i.e. on the GPU when the solver's grid
was built with `backend="gpu"`.  `_unpack` moves the host vector onto the
solver's device before evaluating the objective; `_pack` moves the
resulting device gradient back to host.  This means one host<->device
transfer per L-BFGS-B iteration, which is now the throughput floor for
this minimiser on GPU -- stated plainly rather than hidden, since it means
the GPU speedup here is real but smaller than for step_real/step_imag,
which never leave the device.
"""
from __future__ import annotations

import numpy as np

from .backend import asnumpy
from .kernels import Grid
from .solver import EGPESolver


def _pack(psi):
    """Device array -> packed real NumPy vector (host), for scipy."""
    psi_h = asnumpy(psi)
    return np.concatenate([psi_h.real.ravel(), psi_h.imag.ravel()])


def _unpack(x, shape, xp):
    """Packed real NumPy vector (host) -> complex array on backend `xp`."""
    half = x.size // 2
    psi_h = (x[:half].reshape(shape) + 1j * x[half:].reshape(shape))
    return xp.asarray(psi_h)


def energy_and_gradient(solver: EGPESolver, psi, N_target: float):
    """Projected energy and its gradient in packed real form.

    Returns (E, grad_packed) with the norm constraint handled by rescaling.
    `psi` is expected to already be on the solver's backend (device); the
    returned gradient is packed to host by `_pack`.
    """
    g = solver.g
    xp = g.xp
    nrm2 = float(g.integrate(xp.abs(psi) ** 2))
    scale = np.sqrt(N_target / nrm2)
    ph = psi * scale                      # on the constraint manifold

    saved = solver.psi
    solver.psi = ph
    E = solver.energy()["total"]
    Hp = solver.H_psi()                   # = dE/dpsi*
    solver.psi = saved

    # tangential projection, then chain rule for the rescaling
    overlap = float(xp.real(g.integrate(xp.conj(ph) * Hp)))
    grad = (Hp - (overlap / N_target) * ph) * scale
    return float(E), _pack(grad * g.dV * 2.0)


def minimize_energy(solver: EGPESolver, N_target: float | None = None,
                    maxiter: int = 2000, ftol: float = 1e-14,
                    gtol: float = 1e-12, verbose: bool = True):
    """L-BFGS-B minimisation of E[psi] at fixed norm.

    The solver's `psi` is replaced by the minimiser's result, normalised to
    `N_target`.  Returns a dict with the final energy, chemical potential,
    stationarity residual, iteration count and scipy's message.
    """
    try:
        from scipy.optimize import minimize as _sp_min
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "minimize_energy requires scipy. Install the project "
            "dependencies with `pip install -r requirements.txt` "
            "(or `pip install scipy`)."
        ) from exc

    g = solver.g
    xp = g.xp
    if N_target is None:
        N_target = solver.norm()
    shape = g.shape

    def fun(x):
        psi = _unpack(x, shape, xp)
        E, grad = energy_and_gradient(solver, psi, N_target)
        return E, grad

    x0 = _pack(solver.psi)
    res = _sp_min(fun, x0, jac=True, method="L-BFGS-B",
                  options=dict(maxiter=maxiter, maxfun=4 * maxiter,
                               ftol=ftol, gtol=gtol))

    psi = _unpack(res.x, shape, xp)
    psi = psi * xp.sqrt(N_target / float(g.integrate(xp.abs(psi) ** 2)))
    solver.psi = psi
    mu, residual = solver.mu_and_residual()
    out = dict(energy=solver.energy()["total"], mu=mu, residual=residual,
               nit=int(res.nit), nfev=int(res.nfev),
               success=bool(res.success), message=str(res.message))
    if verbose:
        print(f"  L-BFGS-B: {out['nit']} iters, E={out['energy']:.6f}, "
              f"mu={out['mu']:.6f}, residual={out['residual']:.3e}")
    return out


def multistart_ground_state(grid: Grid, eps_dd: float, gamma: float,
                            Dk, N_target: float, n_seeds: int = 6,
                            lattice_seed=None, presmooth: int = 1500,
                            maxiter: int = 2000, seed0: int = 0,
                            verbose: bool = True):
    """Multi-start search: relax briefly, polish with L-BFGS, keep lowest mu.

    A single seed is not enough (measured: noise gives mu = 5.509, a
    perfect lattice gives 5.233).  Each start is pre-smoothed with a short
    imaginary-time run to remove the highest-k noise, then polished by
    L-BFGS-B.  The lowest-mu result is returned.

    Parameters
    ----------
    lattice_seed : optional callable(grid) -> ndarray giving an initial
        density guess (e.g. an ideal triangular lattice).  Used as seed 0
        when supplied, since a physically motivated start usually wins.
    """
    best = None
    for i in range(n_seeds):
        s = EGPESolver(grid, eps_dd=eps_dd, gamma=gamma, Dk=Dk)
        if i == 0 and lattice_seed is not None:
            seed_host = np.asarray(lattice_seed(grid), dtype=complex)
            s.psi = grid.asarray(seed_host)
            label = "lattice"
        else:
            rng = np.random.default_rng(seed0 + 1000 * i)
            amp = 0.02 * (1 + i)          # vary the seed strength too
            seed_host = (np.ones(grid.shape, dtype=complex)
                        + amp * (rng.standard_normal(grid.shape)
                                + 1j * rng.standard_normal(grid.shape)))
            s.psi = grid.asarray(seed_host)
            label = f"noise(a={amp:.2f})"
        s.psi = s.psi * np.sqrt(N_target / s.norm())
        s.step_imag(0.002, presmooth, norm_target=N_target)
        info = minimize_energy(s, N_target, maxiter=maxiter, verbose=False)
        if verbose:
            print(f"  seed {i} [{label:14s}] mu={info['mu']:.6f} "
                  f"residual={info['residual']:.2e} nit={info['nit']}")
        if best is None or info["mu"] < best[1]["mu"]:
            best = (s, info, label)
    if verbose:
        print(f"  best: {best[2]}  mu={best[1]['mu']:.6f}")
    return best[0], best[1]
