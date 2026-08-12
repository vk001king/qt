"""Grids and dipolar interaction kernels (dimensionless units of Phase 5B, Eq. 10).

Units: length xi = hbar/sqrt(2 m g n0), time hbar/(g n0), energy g n0.
Governing equation:  i dpsi/dt = [ -Lap + V + |psi|^2 + eps_dd*D[|psi|^2]
                                   + gamma*|psi|^3 ] psi
Fourier symbol of D (bare, physical for periodic/bulk emulation):
    D(k) = 3*(khat . ehat)^2 - 1 ,  D(0) := 0        [Phase 5C Eq. (13)]
Ronen-truncated kernel (isolated systems; verified formula, PRA 74, 013623):
    D_R(k) = (3 cos^2 th_k - 1) * B(kR),
    B(x)   = 1 + 3 cos(x)/x^2 - 3 sin(x)/x^3         [Phase 5C Eq. (14)]
Small-x series (derived and checked in Phase 5C):
    B(x) = x^2/10 - x^4/280 + O(x^6)  -> B(0) = 0 (regular; removes k=0
    direction dependence).
"""
from __future__ import annotations
import numpy as np

from .backend import get_backend, asnumpy, device_report


class Grid:
    """Uniform periodic Cartesian grid in d dimensions.

    Parameters
    ----------
    shape : tuple[int]   grid points per dimension
    lengths : tuple[float]  box lengths per dimension (units of xi)
    backend : "cpu" (default, NumPy) or "gpu" (CuPy).  See backend.py for
        why this is explicit rather than auto-detected.  X, K, k2 live on
        the selected backend; everything built from them (solver state,
        diagnostics) follows automatically since they inherit dtype/device
        from array operations on these fields.
    """

    def __init__(self, shape, lengths, backend: str = "cpu"):
        assert len(shape) == len(lengths)
        self.shape = tuple(shape)
        self.lengths = tuple(float(L) for L in lengths)
        self.dim = len(shape)
        self.dx = [L / n for n, L in zip(shape, lengths)]
        self.dV = float(np.prod(self.dx))
        self.Ntot = int(np.prod(shape))

        self.xp, self.fft_mod, self.backend = get_backend(backend)
        if self.backend == "gpu":
            print(device_report(self.xp, self.backend))

        # Coordinate construction uses plain NumPy (cheap, one-off, and
        # fftfreq/linspace semantics are identical across backends) then
        # moves to the selected device in one transfer.
        xs = [np.linspace(-L / 2, L / 2, n, endpoint=False)
              for n, L in zip(shape, lengths)]
        ks = [2 * np.pi * np.fft.fftfreq(n, d=L / n)
              for n, L in zip(shape, lengths)]
        X_np = np.meshgrid(*xs, indexing="ij")
        K_np = np.meshgrid(*ks, indexing="ij")
        self.X = [self.xp.asarray(x) for x in X_np]
        self.K = [self.xp.asarray(k) for k in K_np]
        self.k2 = sum(k * k for k in self.K)

    # --- integration helpers -------------------------------------------
    def integrate(self, f):
        """Riemann integral of real/complex field f over the box."""
        return f.sum() * self.dV

    def asarray(self, x):
        """Move a host (NumPy) array onto this grid's backend."""
        return self.xp.asarray(x)

    def to_host(self, x):
        """Move an array on this grid's backend to a NumPy host array."""
        return asnumpy(x)


def bare_dipolar_symbol(grid: Grid, ehat) -> np.ndarray:
    """Bare (periodic) dipolar Fourier symbol D(k)=3 cos^2(th_k)-1, D(0)=0.

    ehat : polarization unit vector, len == grid.dim (for dim==2 the
           in-plane projection convention is the caller's responsibility;
           production quasi-2D kernels use the reduced form -- see FLAGS.md).

    Built on the HOST (NumPy) and moved to the grid's backend in one
    transfer at the end.  Kernel construction is a one-time cost per run
    (unlike the FFT steps, which dominate runtime), so correctness and
    avoiding scipy/CuPy interop issues matter far more here than speed.
    """
    e = np.asarray(ehat, dtype=float)
    if e.size != grid.dim:
        raise ValueError(f"ehat has {e.size} components but grid.dim="
                         f"{grid.dim}; supply an in-plane axis for 2D "
                         "(e.g. (0,1)), not a truncated 3D vector.")
    nrm = np.linalg.norm(e)
    if nrm == 0.0:
        raise ValueError("ehat is the zero vector; cannot define a "
                         "polarization axis. (Truncating (0,0,1) to 2D "
                         "gives (0,0) -- this was a live bug.)")
    e = e / nrm
    K_host = [grid.to_host(Ki) for Ki in grid.K]
    k2_host = grid.to_host(grid.k2)
    kdote = sum(Ki * ei for Ki, ei in zip(K_host, e))
    with np.errstate(invalid="ignore", divide="ignore"):
        cos2 = np.where(k2_host > 0, (kdote * kdote) / k2_host, 0.0)
    D = 3.0 * cos2 - 1.0
    D[k2_host == 0] = 0.0  # convention: uniform shift absorbed into mu
    return grid.asarray(D)


def truncation_bracket(x: np.ndarray) -> np.ndarray:
    """B(x)=1+3cos(x)/x^2-3sin(x)/x^3 with a cancellation-safe series branch.

    For x < 1e-2 uses B ~ x^2/10 - x^4/280 (rel. error < ~4e-9 at the
    switch point, dominated by the x^6/15120 term).
    """
    x = np.asarray(x, dtype=float)
    small = x < 1e-2
    xs = np.where(small, 1.0, x)  # avoid 0-division in the large branch
    large_val = 1.0 + 3.0 * np.cos(xs) / xs**2 - 3.0 * np.sin(xs) / xs**3
    series = x**2 / 10.0 - x**4 / 280.0
    return np.where(small, series, large_val)


def truncated_dipolar_symbol(grid: Grid, ehat, R: float) -> np.ndarray:
    """Ronen-truncated symbol D_R(k) (Phase 5C Eq. 14); D_R(0)=0 exactly.

    Built on the host for the same reason as bare_dipolar_symbol above.
    """
    D_host = grid.to_host(bare_dipolar_symbol(grid, ehat))
    x = np.sqrt(grid.to_host(grid.k2)) * R
    return grid.asarray(D_host * truncation_bracket(x))


# ---------------------------------------------------------------------------
# Quasi-2D projected dipolar kernel (pancake trap, dipoles along z).
#
# DERIVATION.  The 3D symbol for dipoles polarized along z is
#     U3(k) = g_dd (3 k_z^2 / k^2 - 1),        g_dd = 4 pi hbar^2 a_dd / m.
# Factorize psi(r,z) = psi_2D(r) phi(z) with a normalized Gaussian
#     phi(z) = (pi l_z^2)^{-1/4} exp(-z^2 / 2 l_z^2),
# whose density |phi|^2 has Fourier transform exp(-k_z^2 l_z^2 / 4).
# Projecting out z:
#     U_2D(k) = int dk_z/(2pi) U3(k, k_z) exp(-k_z^2 l_z^2 / 4)
#             = g_dd [ 2/(sqrt(pi) l_z)
#                      - (3k/2) exp(k^2 l_z^2/4) erfc(k l_z/2) ],
# using   int dk_z exp(-a k_z^2)/(k_z^2+k^2) = (pi/k) e^{a k^2} erfc(sqrt(a) k).
# The contact coupling projects to  g_2D = g / (sqrt(2 pi) l_z)  because
# int |phi|^4 dz = 1/(sqrt(2 pi) l_z).  Dividing:
#
#     U_2D(k) / g_2D = eps_dd * D(k),
#     D(k) = 2 sqrt(2) - 3 sqrt(2 pi) u erfcx(u),      u = k l_z / 2,
#
# with erfcx(u) = exp(u^2) erfc(u) evaluated in scaled form for stability.
#
# LIMITS (both verified numerically to 1e-15 against direct quadrature):
#     D(0)      = +2 sqrt(2) = +2.828   repulsive at long wavelength
#     D(k->inf) = -  sqrt(2) = -1.414   ATTRACTIVE at short wavelength
# That sign change is the origin of the roton minimum, and it is exactly
# what the bare periodic kernel lacks -- which is why the bare kernel
# produced only a box-scale mode rather than a droplet crystal.
# ---------------------------------------------------------------------------

def _erfcx(x):
    """exp(x^2) erfc(x), overflow-safe.  Uses scipy when available."""
    x = np.asarray(x, dtype=float)
    try:
        from scipy.special import erfcx as _sp_erfcx
        return _sp_erfcx(x)
    except Exception:
        # asymptotic series for large x, series-free small-x fallback
        from math import erfc as _erfc
        out = np.empty_like(x)
        small = x < 25.0
        xs = x[small]
        out[small] = np.exp(np.minimum(xs * xs, 700.0)) * np.vectorize(_erfc)(xs)
        xl = x[~small]
        # erfcx(x) ~ 1/(x sqrt(pi)) (1 - 1/(2x^2) + 3/(4x^4))
        out[~small] = (1.0 / (xl * np.sqrt(np.pi))
                       * (1.0 - 0.5 / xl**2 + 0.75 / xl**4))
        return out


def quasi2d_dipolar_profile(k, l_z: float) -> np.ndarray:
    """Dimensionless quasi-2D dipolar symbol D(k) (see derivation above).

    Parameters
    ----------
    k    : in-plane wavenumber magnitude (units of xi)
    l_z  : Gaussian axial confinement length (units of xi)
    """
    u = 0.5 * np.asarray(k, dtype=float) * float(l_z)
    return 2.0 * np.sqrt(2.0) - 3.0 * np.sqrt(2.0 * np.pi) * u * _erfcx(u)


def quasi2d_dipolar_symbol(grid: Grid, l_z: float) -> np.ndarray:
    """Quasi-2D projected dipolar Fourier symbol on a 2D grid.

    Unlike `bare_dipolar_symbol`, this carries a roton structure and can
    therefore support a genuine droplet crystal.  Requires a 2D grid.
    Built on the host, then moved to the grid's backend once.
    """
    if grid.dim != 2:
        raise ValueError("quasi2d_dipolar_symbol requires a 2D grid; "
                         f"got dim={grid.dim}")
    if l_z <= 0:
        raise ValueError("l_z must be positive")
    k2_host = grid.to_host(grid.k2)
    return grid.asarray(quasi2d_dipolar_profile(np.sqrt(k2_host), l_z))


def bogoliubov_omega(grid: Grid, n0: float, eps_dd: float, gamma: float,
                     Dk) -> np.ndarray:
    """Bogoliubov frequency omega(k) about a uniform state of density n0.

    In the dimensionless units of this package (kinetic operator -Lap):
        omega(k)^2 = k^2 ( k^2 + 2 n0 [1 + eps_dd D(k)] + 3 gamma n0^{3/2} )
    The LHY contribution follows from 2 n0 d(mu_LHY)/dn = 3 gamma n0^{3/2}
    with mu_LHY = gamma n^{3/2}.  Returns NaN where omega^2 < 0 (dynamically
    unstable), which is the signature of a roton instability.

    Diagnostic-only (never in the per-step hot path): always computed and
    returned on the HOST as a plain NumPy array, regardless of the grid's
    backend, so it can be plotted directly.
    """
    k2 = grid.to_host(grid.k2)
    Dk_h = grid.to_host(Dk)
    inside = k2 + 2.0 * n0 * (1.0 + eps_dd * Dk_h) + 3.0 * gamma * n0 ** 1.5
    w2 = k2 * inside
    out = np.full_like(w2, np.nan)
    ok = w2 >= 0
    out[ok] = np.sqrt(w2[ok])
    return out


def roton_wavevector(grid: Grid, n0: float, eps_dd: float, gamma: float,
                     Dk):
    """Locate the roton: (k_rot, omega_rot, is_unstable).

    k_rot is the wavenumber minimizing omega(k) over k > 0.  When
    omega^2 < 0 somewhere, k_rot is taken at the most negative omega^2 and
    is_unstable is True -- the uniform state then decays into a crystal
    with lattice period ~ 2 pi / k_rot.  Diagnostic-only: computed on the
    host regardless of backend.
    """
    k2 = grid.to_host(grid.k2)
    Dk_h = grid.to_host(Dk)
    inside = k2 + 2.0 * n0 * (1.0 + eps_dd * Dk_h) + 3.0 * gamma * n0 ** 1.5
    w2 = k2 * inside
    mask = k2 > 0
    if np.nanmin(w2[mask]) < 0:
        idx = np.argmin(np.where(mask, w2, np.inf))
        unstable = True
    else:
        idx = np.argmin(np.where(mask, w2, np.inf))
        unstable = False
    i = np.unravel_index(idx, w2.shape)
    k_rot = float(np.sqrt(k2[i]))
    w2r = float(w2[i])
    om = float(np.sqrt(w2r)) if w2r >= 0 else float('nan')
    return k_rot, om, unstable
