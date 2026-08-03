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
