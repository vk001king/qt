"""Imprinting prescribed vortex configurations in a periodic 2D box (v2.12).

A neutral set of point vortices {(x_j, y_j, q_j)}, sum q_j = 0, in a periodic
box Lx x Ly is imprinted with the phase

    theta(x, y) = sum_j q_j arg theta1( pi (zeta - zeta_j) / Lx | tau ) + K y,
    zeta = x + i y,   tau = i Ly / Lx,

where theta1 is the Jacobi theta function.  Near each vortex theta1(z) ~ z,
so the winding of vortex j is exactly q_j.  Periodicity:
  * x -> x + Lx  (z -> z + pi): theta1 -> -theta1, phase change pi * sum q_j = 0;
  * y -> y + Ly  (z -> z + pi tau): theta1 -> -q^{-1} e^{-2iz} theta1, which
    changes the phase by 2 pi sum_j q_j x_j / Lx (a constant); the uniform
    term K y with K = 2 pi (m - sum_j q_j x_j / Lx) / Ly, m the nearest
    integer, cancels it.  The result is single-valued on the torus.
The amplitude is multiplied by prod_j r_j / sqrt(r_j^2 + r_c^2) (minimum-image
distances) so that each core starts with a density zero of width r_c.
"""
from __future__ import annotations
import numpy as np


def _theta1(z, q, nterms=12):
    out = np.zeros_like(z, dtype=complex)
    for n in range(nterms):
        out += (-1) ** n * q ** ((n + 0.5) ** 2) * np.sin((2 * n + 1) * z)
    return 2.0 * out


def vortex_phase(X, Y, Lx, Ly, xs, ys, qs):
    """Periodic phase field of a neutral point-vortex configuration.

    X, Y: coordinate arrays measured from the box corner (0 <= X < Lx).
    """
    xs, ys, qs = (np.asarray(v, dtype=float) for v in (xs, ys, qs))
    if abs(qs.sum()) > 1e-12:
        raise ValueError("configuration must be neutral (sum q = 0) in a periodic box")
    tau = 1j * Ly / Lx
    qn = np.exp(1j * np.pi * tau)
    th = np.zeros(X.shape)
    for x0, y0, q0 in zip(xs, ys, qs):
        z = np.pi * ((X - x0) + 1j * (Y - y0)) / Lx
        th += q0 * np.angle(_theta1(z, qn))
    shift = np.sum(qs * xs) / Lx
    K = 2 * np.pi * (np.round(shift) - shift) / Ly
    return th + K * Y


def core_profile(X, Y, Lx, Ly, xs, ys, rc=1.0):
    f = np.ones(X.shape)
    for x0, y0 in zip(xs, ys):
        dx = (X - x0 + Lx / 2) % Lx - Lx / 2
        dy = (Y - y0 + Ly / 2) % Ly - Ly / 2
        r2 = dx * dx + dy * dy
        f *= np.sqrt(r2 / (r2 + rc * rc))
    return f


def imprint(psi, grid, xs, ys, qs, rc=1.0):
    """Return psi * core_profile * exp(i theta) on the grid's backend.

    Coordinates are taken from the box corner, i.e. x in [0, Lx), matching
    qtsim.diagnostics.droplet_centers / interstitial_charges.
    """
    Lx, Ly = grid.lengths
    X = grid.to_host(grid.X[0]) + Lx / 2
    Y = grid.to_host(grid.X[1]) + Ly / 2
    th = vortex_phase(X, Y, Lx, Ly, xs, ys, qs)
    amp = core_profile(X, Y, Lx, Ly, xs, ys, rc)
    return psi * grid.asarray(amp * np.exp(1j * th))
