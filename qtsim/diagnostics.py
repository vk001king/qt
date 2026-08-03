"""Turbulence diagnostics: vortices, circulation, Helmholtz decomposition.

2D plaquette winding is exact (+/-2pi per elementary cell containing a
vortex).  The Helmholtz split acts on w = sqrt(n) v (Phase 5B III.E) and is
orthogonal on the periodic grid by Parseval, so
E_kin,total(w) = E_i + E_c holds to round-off (asserted in tests).
"""
from __future__ import annotations
import numpy as np
from .kernels import Grid

_fft, _ifft = np.fft.fftn, np.fft.ifftn


def _wrap(a):
    """Wrap angle array to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def plaquette_charges_2d(psi):
    """Integer winding number on each grid plaquette (periodic).

    Returns int array of shape psi.shape with entries in {-1, 0, +1}
    (higher charges would appear as adjacent unit charges at our
    resolutions; multiply-quantized cores are unstable anyway).
    """
    th = np.angle(psi)
    d1 = _wrap(np.roll(th, -1, 0) - th)                       # right edge up
    d2 = _wrap(np.roll(np.roll(th, -1, 0), -1, 1)
               - np.roll(th, -1, 0))                          # top edge
    d3 = _wrap(np.roll(th, -1, 1) - np.roll(np.roll(th, -1, 0), -1, 1))
    d4 = _wrap(th - np.roll(th, -1, 1))
    # Orientation verified by a minimal single-vortex probe (see
    # VALIDATION_REPORT): the edge terms d1..d4 equal the hand-computed
    # CCW differences (B-A, C-B, D-C, A-D) exactly, so the raw sum is
    # already counter-clockwise / right-handed.  (An earlier negation
    # here, added on a wrong clockwise diagnosis, inverted all charges
    # and was removed after the probe.)
    w = (d1 + d2 + d3 + d4) / (2 * np.pi)
    return np.rint(w).astype(int)


def circulation_loop_2d(psi, grid: Grid, center, radius):
    """Circulation (units of 2*pi) around a square loop of half-side
    `radius` centered at `center` (grid units), via wrapped phase sums."""
    th = np.angle(psi)
    i0 = int(round((center[0] + grid.lengths[0] / 2) / grid.dx[0]))
    j0 = int(round((center[1] + grid.lengths[1] / 2) / grid.dx[1]))
    r = int(round(radius / grid.dx[0]))
    n0, n1 = grid.shape
    path = []
    for j in range(-r, r):   path.append(((i0 - r) % n0, (j0 + j) % n1))
    for i in range(-r, r):   path.append((((i0 + i) % n0), (j0 + r) % n1))
    for j in range(r, -r, -1): path.append(((i0 + r) % n0, (j0 + j) % n1))
    for i in range(r, -r, -1): path.append((((i0 + i) % n0), (j0 - r) % n1))
    tot = 0.0
    for (a, b), (c, d) in zip(path, path[1:] + path[:1]):
        tot += _wrap(th[c, d] - th[a, b])
    # The four-segment path (bottom→right→top→left) traverses CLOCKWISE
    # in the (i,j)=(x,y) convention used by arctan2.  Single-vortex probe
    # confirmed: raw sum = -1 for a known +1 vortex.  Negate to match
    # the plaquette detector's verified CCW convention.
    return -tot / (2 * np.pi)


def helmholtz_split_2d(psi, grid: Grid, floor=1e-12):
    """Split w = sqrt(n) v into incompressible/compressible parts (periodic).

    Returns (Ei, Ec, Etot) kinetic energies with Etot = Ei + Ec (Parseval).
    v = 2*Im(psi* grad psi)/n in these units (since v = 2*grad(phase) when
    lengths are in xi and the kinetic operator is -Lap).
    """
    n = np.abs(psi) ** 2
    ws = []
    psik = _fft(psi)
    for Ki in grid.K:
        dpsi = _ifft(1j * Ki * psik)
        # sqrt(n) v_i = 2*Im(conj(psi) dpsi)/sqrt(n)
        ws.append(2.0 * np.imag(np.conj(psi) * dpsi)
                  / np.sqrt(np.maximum(n, floor)))
    wk = [_fft(w) for w in ws]
    k2 = np.maximum(grid.k2, 1e-30)
    kdotw = sum(Ki * wki for Ki, wki in zip(grid.K, wk))
    Ei = Ec = 0.0
    for Ki, wki in zip(grid.K, wk):
        wc = Ki * kdotw / k2          # longitudinal (compressible) part
        wc[grid.k2 == 0] = 0.0
        wi = wki - wc
        Ei += float(np.sum(np.abs(wi) ** 2))
        Ec += float(np.sum(np.abs(wc) ** 2))
    scale = grid.dV / grid.Ntot       # Parseval normalization
    return Ei * scale, Ec * scale, (Ei + Ec) * scale
