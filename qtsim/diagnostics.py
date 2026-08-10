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


def _annulus_kernel(grid: Grid, r_in: float, r_out: float):
    """Normalized annulus mask in Fourier space, for fast local averaging."""
    xs = [np.fft.fftshift(np.arange(n) - n // 2) * d
          for n, d in zip(grid.shape, grid.dx)]
    R = np.sqrt(sum(x ** 2 for x in np.meshgrid(*xs, indexing="ij")))
    m = ((R >= r_in) & (R <= r_out)).astype(float)
    tot = m.sum()
    if tot == 0:
        raise ValueError("annulus contains no grid points; widen r_in..r_out")
    return _fft(m / tot)


def plaquette_charges_2d_masked(psi, grid: Grid, r_in: float = 0.5,
                                r_out: float = 2.0, frac: float = 0.3,
                                return_stats: bool = False):
    """Vortex charges with near-vacuum false positives removed.

    WHY THIS IS NEEDED.  The raw plaquette winding is exact wherever the
    phase is meaningful, but in a droplet crystal the inter-droplet regions
    are essentially vacuum (measured n_min ~ 1e-6 n_mean).  Phase there is
    numerical noise, and the detector happily finds random +/-2pi windings
    in it.  On a stirred quasi-2D crystal the raw count was 874 while
    masking plaquettes whose corners fell below 1 percent of the mean
    density left 459 -- a factor of two.

    A plain density mask is the WRONG fix, because a genuine vortex core
    also has n -> 0 at its centre.  The discriminator is the density in an
    ANNULUS around the candidate: a real vortex sits in bulk fluid and has
    a substantial annulus mean, whereas a spurious detection in a void has
    an annulus mean near zero.

    Parameters
    ----------
    r_in, r_out : annulus radii in units of xi.  Defaults skip the core
        (radius ~ xi) and sample the surrounding fluid.
    frac : keep a candidate when the annulus mean density exceeds
        frac * (mean density of the whole field).

    ACCURACY, measured on a fluid/void control (real fluid with one true
    vortex on one side, near-vacuum with random phase on the other):
    the defaults reject 97.4 percent of void false positives while keeping
    the real vortex.  The residual 2.6 percent sit on the fluid/void
    INTERFACE, where the annulus straddles both and the classification is
    genuinely ambiguous -- no parameter choice removes them.

    SYSTEMATIC UNCERTAINTY ON L.  On a stirred quasi-2D droplet crystal
    the count was 874 raw, 724 with this annulus mask (-17 percent), and
    459 with a cruder all-corners density mask (-47 percent).  Vortex
    line density in a droplet crystal therefore carries a METHOD-DEPENDENT
    SYSTEMATIC of order 20-50 percent.  Since L is the primary observable
    for the decay-law and avalanche analyses, that systematic must be
    quoted alongside L rather than a single number being reported.

    Returns
    -------
    q_masked, or (q_masked, stats) when return_stats is True.
    """
    if grid.dim != 2:
        raise ValueError("plaquette_charges_2d_masked requires a 2D grid")
    q = plaquette_charges_2d(psi)
    n = np.abs(psi) ** 2
    ann = _annulus_kernel(grid, r_in, r_out)
    n_ann = _ifft(ann * _fft(n)).real          # annulus-averaged density
    keep = n_ann > frac * n.mean()
    q_m = np.where(keep, q, 0)
    if not return_stats:
        return q_m
    raw = int((q != 0).sum())
    kept = int((q_m != 0).sum())
    stats = dict(raw=raw, kept=kept,
                 rejected=raw - kept,
                 reject_fraction=(raw - kept) / raw if raw else 0.0,
                 net_charge=int((q_m == 1).sum() - (q_m == -1).sum()))
    return q_m, stats
