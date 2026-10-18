"""Lee-Huang-Yang auxiliary integral Q5 and dimensionless LHY strength.

Q5(eps) = 1/2 * int_0^pi sin(a) [1 + eps*(3 cos^2 a - 1)]^{5/2} da
        = int_0^1 [1 + eps*(3 u^2 - 1)]^{5/2} du            (u = cos a; even)

Verified analytic anchors used as unit tests:
  Q5(0) = 1
  Q5(1) = int_0^1 (3u^2)^{5/2} du = 3^{5/2}/6 = 2.598076211...
Convention (documented in Phase 5B, corroborated in retrieved literature):
for eps_dd > 1 the radicand is negative near u=0; Q5 is evaluated with a
complex power and the REAL PART is retained.

gamma_tilde = (32/(3 sqrt(pi))) * sqrt(n0 a_s^3) * Q5(eps_dd)
(dimensionless LHY strength; see Phase 5B Eq. 11).
"""
from __future__ import annotations
import numpy as np


def Q5(eps_dd: float, nquad: int = 96) -> float:
    """Gauss-Legendre evaluation of Q5 on u in [0,1]; Re-part convention."""
    nodes, weights = np.polynomial.legendre.leggauss(nquad)
    u = 0.5 * (nodes + 1.0)          # map [-1,1] -> [0,1]
    w = 0.5 * weights
    radicand = 1.0 + eps_dd * (3.0 * u * u - 1.0)
    vals = np.power(radicand.astype(complex), 2.5)
    return float(np.sum(w * vals.real))


def gamma_tilde(eps_dd: float, n0_as3: float) -> float:
    """Dimensionless LHY coefficient gamma~ (Phase 5B Eq. 11).

    n0_as3 : gas parameter n0 * a_s^3 (dimensionless, ~1e-4..1e-5 for Dy).
    """
    return (32.0 / (3.0 * np.sqrt(np.pi))) * np.sqrt(n0_as3) * Q5(eps_dd)


# ---------------------------------------------------------------------------
# v2.12: quasi-2D projection and a single source of consistent parameters.
#
# With Psi = psi_2(x,y) phi(z), phi Gaussian of width l (|phi|^2 =
# exp(-z^2/l^2)/(sqrt(pi) l)), the LHY energy (2/5) g_QF n^{5/2} projects with
#     int |phi|^5 dz = (pi l^2)^{-5/4} l sqrt(2 pi / 5).
# In the units E0 = g_2D nbar (g_2D = g/(sqrt(2 pi) l)) the in-plane LHY
# coefficient is
#     gamma_2D = (2/sqrt5) * gamma_tilde(eps, n_p a_s^3),
# where n_p = nbar/(sqrt(pi) l) is the PEAK 3D density of the reference
# state.  The same reference fixes the length unit:
#     E0 = g n_p / sqrt2,   xi = hbar / sqrt(2 m E0),   l/xi = sqrt(sqrt2 g n_p / (hbar w_z)).
# ---------------------------------------------------------------------------
LHY_Q2D_FACTOR = 2.0 / np.sqrt(5.0)

_HBAR = 1.054571817e-34
_A0 = 5.29177210903e-11
_AMU = 1.66053906660e-27


def q2d_parameters(eps_dd: float, omega_z_hz: float, n_peak: float,
                   a_dd_a0: float = 130.8, mass_amu: float = 163.929175):
    """All dimensionless inputs of the quasi-2D eGPE from physical ones.

    Parameters: eps_dd, axial trap frequency (Hz), peak 3D density (m^-3),
    dipolar length (a0, default 164Dy) and mass (amu).
    Returns dict(l_z, gamma, xi_m, tau_s, a_s_a0, n_peak_as3).
    """
    m = mass_amu * _AMU
    a_s = a_dd_a0 * _A0 / eps_dd
    g = 4 * np.pi * _HBAR ** 2 * a_s / m
    E0 = g * n_peak / np.sqrt(2.0)
    xi = _HBAR / np.sqrt(2 * m * E0)
    l = np.sqrt(_HBAR / (m * 2 * np.pi * omega_z_hz))
    gas = n_peak * a_s ** 3
    return dict(l_z=float(l / xi),
                gamma=float(LHY_Q2D_FACTOR * gamma_tilde(eps_dd, gas)),
                xi_m=float(xi), tau_s=float(_HBAR / E0),
                a_s_a0=float(a_s / _A0), n_peak_as3=float(gas))
