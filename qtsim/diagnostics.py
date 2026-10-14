"""Turbulence diagnostics: vortices, circulation, Helmholtz decomposition.

2D plaquette winding is exact (+/-2pi per elementary cell containing a
vortex).  The Helmholtz split acts on w = sqrt(n) v (Phase 5B III.E) and is
orthogonal on the periodic grid by Parseval, so
E_kin,total(w) = E_i + E_c holds to round-off (asserted in tests).

BACKEND.  Functions that receive a `grid` argument use `grid.xp` and
`grid.fft_mod`, matching kernels.py and solver.py.  A few functions here
(`plaquette_charges_2d`, `circulation_loop_2d`'s angle step) take only
`psi` with no grid, since they are called that way throughout the
notebook and campaign scripts; for those, `_xp_of(psi)` duck-types the
array's own module so the right backend is used without changing any
call site.  On the default CPU backend this is identical to before the
GPU port -- `_xp_of` returns plain NumPy for a NumPy array.
"""
from __future__ import annotations
import numpy as np
from .kernels import Grid


def _xp_of(arr):
    """Return (xp, fft_module) matching the array's own backend.

    Duck-types on the array's module name rather than importing CuPy
    unconditionally, so this file has no hard CuPy dependency.
    """
    mod = type(arr).__module__.split(".")[0]
    if mod == "cupy":
        import cupy as cp
        return cp, cp.fft
    return np, np.fft


def _wrap(a, xp=np):
    """Wrap angle array to (-pi, pi]."""
    return (a + xp.pi) % (2 * xp.pi) - xp.pi


def plaquette_charges_2d(psi):
    """Integer winding number on each grid plaquette (periodic).

    Returns int array of shape psi.shape with entries in {-1, 0, +1}
    (higher charges would appear as adjacent unit charges at our
    resolutions; multiply-quantized cores are unstable anyway).
    """
    xp, _ = _xp_of(psi)
    th = xp.angle(psi)
    d1 = _wrap(xp.roll(th, -1, 0) - th, xp)                   # right edge up
    d2 = _wrap(xp.roll(xp.roll(th, -1, 0), -1, 1)
               - xp.roll(th, -1, 0), xp)                      # top edge
    d3 = _wrap(xp.roll(th, -1, 1) - xp.roll(xp.roll(th, -1, 0), -1, 1), xp)
    d4 = _wrap(th - xp.roll(th, -1, 1), xp)
    # Orientation verified by a minimal single-vortex probe (see
    # VALIDATION_REPORT): the edge terms d1..d4 equal the hand-computed
    # CCW differences (B-A, C-B, D-C, A-D) exactly, so the raw sum is
    # already counter-clockwise / right-handed.  (An earlier negation
    # here, added on a wrong clockwise diagnosis, inverted all charges
    # and was removed after the probe.)
    w = (d1 + d2 + d3 + d4) / (2 * xp.pi)
    return xp.rint(w).astype(int)


def circulation_loop_2d(psi, grid: Grid, center, radius):
    """Circulation (units of 2*pi) around a square loop of half-side
    `radius` centered at `center` (grid units), via wrapped phase sums."""
    xp, _ = _xp_of(psi)
    th = xp.angle(psi)
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
        tot += float(_wrap(th[c, d] - th[a, b], xp))
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
    xp = grid.xp
    fft, ifft = grid.fft_mod.fftn, grid.fft_mod.ifftn
    n = xp.abs(psi) ** 2
    ws = []
    psik = fft(psi)
    for Ki in grid.K:
        dpsi = ifft(1j * Ki * psik)
        # sqrt(n) v_i = 2*Im(conj(psi) dpsi)/sqrt(n)
        ws.append(2.0 * xp.imag(xp.conj(psi) * dpsi)
                  / xp.sqrt(xp.maximum(n, floor)))
    wk = [fft(w) for w in ws]
    k2 = xp.maximum(grid.k2, 1e-30)
    kdotw = sum(Ki * wki for Ki, wki in zip(grid.K, wk))
    Ei = Ec = 0.0
    for Ki, wki in zip(grid.K, wk):
        wc = Ki * kdotw / k2          # longitudinal (compressible) part
        wc = xp.where(grid.k2 == 0, 0.0, wc)
        wi = wki - wc
        Ei += float(xp.sum(xp.abs(wi) ** 2))
        Ec += float(xp.sum(xp.abs(wc) ** 2))
    scale = grid.dV / grid.Ntot       # Parseval normalization
    return Ei * scale, Ec * scale, (Ei + Ec) * scale


def _annulus_kernel(grid: Grid, r_in: float, r_out: float):
    """Normalized annulus mask in Fourier space, for fast local averaging.

    Built on the host (one-time per grid, cheap) then moved to the grid's
    backend, matching the pattern used for kernel construction in
    kernels.py.
    """
    xs = [np.fft.fftshift(np.arange(n) - n // 2) * d
          for n, d in zip(grid.shape, grid.dx)]
    R = np.sqrt(sum(x ** 2 for x in np.meshgrid(*xs, indexing="ij")))
    m = ((R >= r_in) & (R <= r_out)).astype(float)
    tot = m.sum()
    if tot == 0:
        raise ValueError("annulus contains no grid points; widen r_in..r_out")
    m_dev = grid.asarray(m / tot)
    return grid.fft_mod.fftn(m_dev)


def _dilate_sign(q, s, radius, xp):
    """Boolean map: True within `radius` plaquettes (disc, periodic) of a
    charge of sign s.  Rolls only, so it runs unchanged on NumPy or CuPy.
    Cost: about pi*radius^2 rolls of a boolean array per call."""
    src = q == s
    out = src.copy()
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if (dx or dy) and dx * dx + dy * dy <= radius * radius:
                out |= xp.roll(xp.roll(src, dx, 0), dy, 1)
    return out


def persistent_charges(q, history, radius=2):
    """Keep only charges confirmed in EVERY earlier frame of `history`.

    v2.8.  A charge of sign s at plaquette p survives if each frame in
    `history` (charge maps from the previous samples, same grid) holds a
    charge of the same sign within `radius` plaquettes of p (periodic).
    Pass radius so that it exceeds the distance a real vortex moves
    between samples (vortex speed < c ~ 1.4 xi/tau, so < 0.7 xi per 0.5
    tau; radius = 1 xi is the default in track_a_scan).

    WHY.  On the Track A data the masked count went UP in 47-53 percent of
    its nonzero sample-to-sample steps and jumped by 6+ vortices in 25-40
    percent of samples, with a dominant period of 1.1-2.4 tau.  True vortex
    number cannot rise in free decay, so most of that is detection flicker
    (candidates at the droplet/void interface moving in and out of the
    annulus mask as droplets breathe).  Flicker that is uncorrelated from
    one sample to the next is removed by requiring persistence; a real
    vortex, which moves continuously, is kept.  An annihilating pair
    disappears from the persistent count at the first sample in which it
    is gone, so annihilation timing is not delayed (only the APPEARANCE
    of a vortex is delayed by len(history) samples).

    Returns the filtered charge map (same dtype as q).
    """
    xp, _ = _xp_of(q)
    keep = q != 0
    for h in history:
        for s in (1, -1):
            keep &= ~((q == s) & ~_dilate_sign(h, s, radius, xp))
    return xp.where(keep, q, 0)


def droplet_centers(n, grid: Grid, d: float, smooth: float = 1.5):
    """Droplet positions (physical coords from the box corner, host NumPy).

    Density is Gaussian-smoothed (sigma `smooth` xi, periodic, via FFT),
    then local maxima over a window of ~0.6 d that exceed the mean density
    are kept.  Runs on the host: one 2D array transfer per call.
    """
    from scipy.ndimage import maximum_filter
    from .backend import asnumpy
    nh = asnumpy(n).astype(float)
    k = [np.fft.fftfreq(s, dd) * 2 * np.pi for s, dd in zip(nh.shape, grid.dx)]
    K2 = k[0][:, None] ** 2 + k[1][None, :] ** 2
    ns = np.fft.ifft2(np.fft.fft2(nh) * np.exp(-0.5 * K2 * smooth ** 2)).real
    w = [max(3, int(round(0.6 * d / dd)) | 1) for dd in grid.dx]
    peak = (ns == maximum_filter(ns, size=w, mode="wrap")) & (ns > ns.mean())
    idx = np.argwhere(peak)
    return idx * np.array(grid.dx, dtype=float)


def interstitial_charges(q, centers, grid: Grid):
    """Net winding around every interstitial triangle of droplets.

    v2.8.  The plaquette charges are summed over the Voronoi cell of each
    interstitial site (the centroid of a Delaunay triangle of droplets).
    For a triangular droplet lattice those Voronoi cells ARE the droplet
    triangles, so by Stokes the sum is the phase winding around a loop
    through three droplets, where the density is high and the phase is
    well defined.  Defects that flicker in and out inside a void come in
    +/- pairs and cancel; only a winding that encloses the void counts.

    WHY.  Track A snapshots showed masked plaquette 'vortices' jumping a
    median 2-7 xi between samples 0.5 tau apart (impossible for a vortex:
    c ~ 1.4 xi/tau), with core density 0.01-0.05 of the mean -- the core
    sits in a near-vacuum interstitial and its plaquette position is not
    defined.  The interstitial winding is (Alana et al. also place
    vortices at interstitial sites).

    Returns (w, sites): integer winding per site and site coordinates.
    Periodic: droplets are tiled into the 8 neighbouring images before the
    triangulation, and only triangles whose centroid lies in the box kept.
    """
    from scipy.spatial import Delaunay, cKDTree
    from .backend import asnumpy
    L = np.array(grid.lengths, dtype=float)
    if len(centers) < 3:
        return np.zeros(0, int), np.zeros((0, 2))
    tiles = np.concatenate([centers + L * np.array([a, b])
                            for a in (-1, 0, 1) for b in (-1, 0, 1)])
    tri = Delaunay(tiles)
    cen = tiles[tri.simplices].mean(axis=1)
    inside = np.all((cen >= 0) & (cen < L), axis=1)
    sites = cen[inside]
    qh = asnumpy(q)
    nz = np.argwhere(qh != 0)
    if len(nz) == 0 or len(sites) == 0:
        return np.zeros(len(sites), int), sites
    pos = np.mod((nz + 0.5) * np.array(grid.dx, dtype=float), L)
    # exact: the triangle that CONTAINS each charge, mapped to its periodic
    # image whose centroid is in the box (a distorted lattice is handled)
    s = tri.find_simplex(pos)
    ok = s >= 0
    c = np.mod(cen[s[ok]], L)
    _, j = cKDTree(sites, boxsize=L).query(c)
    nz = nz[ok]
    w = np.bincount(j, weights=qh[nz[:, 0], nz[:, 1]], minlength=len(sites))
    return np.rint(w).astype(int), sites


def link_coherence(psi, centers, grid: Grid):
    """Phase coherence across nearest-neighbour droplet links (v2.9).

    Links are the edges of the periodic Delaunay triangulation of the
    droplet centres; the phase of each droplet is read at its centre
    (where the density, and so the phase, is best defined).  Returns
    dict(cos=<cos dphi>  [1 = phase-locked crystal, 0 = random phases],
         frac_gt_half_pi = fraction of links with |dphi| > pi/2,
         median_abs = median |dphi| in rad, n_links).
    A ground state gives cos = 1 exactly.  After stirring, the Track A
    snapshots gave median |dphi| ~ 1.1-1.3 rad, i.e. weak links that
    support frequent phase slips (the cause of nv_lat fluctuating).
    """
    from scipy.spatial import Delaunay
    from .backend import asnumpy
    L = np.array(grid.lengths, dtype=float)
    nan = float("nan")
    if len(centers) < 3:
        return dict(cos=nan, frac_gt_half_pi=nan, median_abs=nan, n_links=0)
    ph_all = np.angle(asnumpy(psi))
    idx = np.rint(centers / np.array(grid.dx, dtype=float)).astype(int) \
        % np.array(grid.shape)
    ph = ph_all[idx[:, 0], idx[:, 1]]
    n0 = len(centers)
    tiles = np.concatenate([centers + L * np.array([a, b])
                            for a in (-1, 0, 1) for b in (-1, 0, 1)])
    tri = Delaunay(tiles)
    s = tri.simplices
    e = np.concatenate([s[:, [0, 1]], s[:, [1, 2]], s[:, [2, 0]]]) % n0
    e = np.unique(np.sort(e, axis=1), axis=0)
    e = e[e[:, 0] != e[:, 1]]
    dphi = np.angle(np.exp(1j * (ph[e[:, 0]] - ph[e[:, 1]])))
    a = np.abs(dphi)
    return dict(cos=float(np.mean(np.cos(dphi))),
                frac_gt_half_pi=float(np.mean(a > np.pi / 2)),
                median_abs=float(np.median(a)), n_links=int(len(e)))


def measured_spacing(n_droplets, area):
    """Triangular-lattice spacing implied by the droplet count (v2.9):
    area per droplet = (sqrt 3 / 2) a^2.  The roton estimate d assumed by
    the commensurate box was 10.3 xi; a no-stir 5-cell crystal measured
    9.2 xi nearest-neighbour spacing with 28 droplets instead of 25."""
    if not n_droplets:
        return float("nan")
    return float(np.sqrt(2.0 * area / (np.sqrt(3.0) * n_droplets)))


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

    SYSTEMATIC UNCERTAINTY ON L, measured on a real stirred campaign run
    (not just the fluid/void control): rejection fraction was 90%, 75%,
    54%, 35% at four successive times as the tangle grew, and raw and
    masked counts grow at DIFFERENT RATES (2.8x vs 18x over that run), so
    a raw L(t) has the wrong SHAPE, not merely an offset.  L in a droplet
    crystal therefore carries a large, time-dependent, method-dependent
    systematic that must be quoted, never treated as a clean number.

    MASK QUALITY GATE.  True total circulation is exactly zero in a
    periodic box, and the raw plaquette count respects that identically.
    Because the mask is spatial, it can clip one sign preferentially near
    a fluid/void interface, breaking neutrality; the residual net charge
    on the masked count is therefore the only internal check available on
    the mask itself, and is returned as `charge_imbalance` /
    `mask_trustworthy`.  Measured: imbalance fell 14% -> 11% -> 2% -> 0.8%
    as nv grew 14 -> 250 -- the mask is LEAST trustworthy exactly where
    there are fewest vortices, which is the R>1 side of the H4' scan.

    Returns
    -------
    q_masked, or (q_masked, stats) when return_stats is True.
    """
    if grid.dim != 2:
        raise ValueError("plaquette_charges_2d_masked requires a 2D grid")
    xp = grid.xp
    fft, ifft = grid.fft_mod.fftn, grid.fft_mod.ifftn
    q = plaquette_charges_2d(psi)
    n = xp.abs(psi) ** 2
    ann = _annulus_kernel(grid, r_in, r_out)
    n_ann = ifft(ann * fft(n)).real          # annulus-averaged density
    keep = n_ann > frac * n.mean()
    q_m = xp.where(keep, q, 0)
    if not return_stats:
        return q_m
    raw = int((q != 0).sum())
    kept = int((q_m != 0).sum())
    net = int((q_m == 1).sum() - (q_m == -1).sum())
    imbalance = abs(net) / kept if kept else 0.0
    stats = dict(raw=raw, kept=kept,
                 rejected=raw - kept,
                 reject_fraction=(raw - kept) / raw if raw else 0.0,
                 net_charge=net,
                 charge_imbalance=imbalance,
                 mask_trustworthy=bool(imbalance < 0.05))
    return q_m, stats
