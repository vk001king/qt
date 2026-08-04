"""Split-step Fourier eGPE solver (Phase 5C pseudocode, faithfully coded).

Dimensionless equation (Phase 5B Eq. 10):
    i dpsi/dt = [ -Lap + V(x,t) + n + eps_dd * D[n] + gamma * n^{3/2} ] psi,
    n = |psi|^2.

Algorithm: Strang splitting (2nd order):
    half-kinetic (exact, k-space) -> full nonlinear (exact, x-space)
    -> half-kinetic.  See Phase 5C IV.B for the exactness proofs.
Complexity: 4 FFTs/step without dipolar term, 6 with; O(N log N).
Memory: ~6 complex arrays of grid size.
"""
from __future__ import annotations
import warnings

import numpy as np

from .kernels import Grid

_fft, _ifft = np.fft.fftn, np.fft.ifftn


class EGPESolver:
    """Minimal, dependency-free reference implementation (CPU / NumPy).

    Parameters
    ----------
    grid : Grid
    V : ndarray or None      static external potential on grid (units g*n0)
    eps_dd : float           dipolar strength eps_dd = a_dd/a_s
    gamma : float            dimensionless LHY strength gamma~
    Dk : ndarray or None     dipolar Fourier symbol on grid (bare/truncated)
    """

    def __init__(self, grid: Grid, V=None, eps_dd: float = 0.0,
                 gamma: float = 0.0, Dk=None):
        self.g = grid
        self.V = np.zeros(grid.shape) if V is None else np.asarray(V, float)
        self.eps_dd = float(eps_dd)
        self.gamma = float(gamma)
        self.Dk = Dk
        if self.eps_dd != 0.0 and Dk is None:
            raise ValueError("eps_dd != 0 requires a dipolar symbol Dk")
        self.psi = np.zeros(grid.shape, dtype=complex)

    # ------------------------------------------------------------------
    def _Phi_dd(self, n):
        if self.eps_dd == 0.0:
            return 0.0
        return _ifft(self.Dk * _fft(n)).real

    def _W(self, n):
        """Local effective potential of the nonlinear substep."""
        return (self.V + n + self.eps_dd * self._Phi_dd(n)
                + self.gamma * n ** 1.5)

    # ------------------------------------------------------------------
    # Timestep stability guard.
    #
    # The kinetic substep applies exp(-i k^2 dt) exactly, so there is no
    # CFL-type *stability* limit in the linear problem.  But the highest
    # representable mode advances k_max^2 * dt radians per step, and once
    # that approaches pi those modes are unresolved in time; coupling
    # through the nonlinear term then pumps them and the field detonates
    # after a few tens of time units with the norm still conserved to
    # round-off (so a norm check will NOT catch it).
    #
    # Empirically verified on a 2D vortex pair:
    #   6.32 rad/step -> blows up near t = 35
    #   0.79 rad/step -> stable to t = 200, dE/E ~ 1e-8
    # Hence: warn above WARN, refuse above HARD.
    PHASE_WARN = 1.0     # rad per step: above this, accuracy degrades
    PHASE_HARD = 2.0     # rad per step: above this, expect instability

    def max_phase_per_step(self, dt: float) -> float:
        """k_max^2 * dt: phase advance of the highest grid mode per step."""
        return float(self.g.k2.max()) * float(dt)

    def suggested_dt(self, phase: float = 0.8) -> float:
        """Largest dt whose highest-mode phase advance is `phase` radians."""
        return phase / float(self.g.k2.max())

    def _check_dt(self, dt: float, strict: bool = True) -> None:
        ph = self.max_phase_per_step(dt)
        if ph <= self.PHASE_WARN:
            return
        msg = (f"dt={dt:g} gives {ph:.2f} rad/step for the highest grid "
               f"mode (k_max^2*dt). Recommended dt <= "
               f"{self.suggested_dt():.4g} on this grid.")
        if ph > self.PHASE_HARD and strict:
            raise ValueError(
                "Unstable timestep: " + msg +
                " Pass strict=False to override deliberately.")
        warnings.warn("Marginal timestep: " + msg, RuntimeWarning,
                      stacklevel=3)

    # ------------------------------------------------------------------
    def step_real(self, dt: float, nsteps: int = 1, strict: bool = True):
        """Real-time Strang steps (norm-conserving to round-off).

        Raises ValueError when dt is large enough to cause temporal
        aliasing of the highest grid modes (see the guard above).
        """
        self._check_dt(dt, strict=strict)
        halfK = np.exp(-1j * self.g.k2 * dt / 2.0)
        for _ in range(nsteps):
            self.psi = _ifft(halfK * _fft(self.psi))
            n = np.abs(self.psi) ** 2
            self.psi = self.psi * np.exp(-1j * self._W(n) * dt)
            self.psi = _ifft(halfK * _fft(self.psi))

    def step_imag(self, dtau: float, nsteps: int = 1, norm_target=None):
        """Imaginary-time gradient flow with per-step renormalization."""
        if norm_target is None:
            norm_target = self.norm()
        halfK = np.exp(-self.g.k2 * dtau / 2.0)
        for _ in range(nsteps):
            self.psi = _ifft(halfK * _fft(self.psi))
            n = np.abs(self.psi) ** 2
            self.psi = self.psi * np.exp(-self._W(n) * dtau)
            self.psi = _ifft(halfK * _fft(self.psi))
            self.psi *= np.sqrt(norm_target / self.norm())

    # ------------------------------------------------------------------
    def norm(self) -> float:
        return float(self.g.integrate(np.abs(self.psi) ** 2))

    def energy(self) -> dict:
        """Energy functional terms (Phase 5B Eq. 9, dimensionless).

        E = int [ |grad psi|^2 + V n + n^2/2 + (eps/2) Phi n
                  + (2/5) gamma n^{5/2} ] dV
        Kinetic term evaluated spectrally (Parseval).
        """
        g = self.g
        psik = _fft(self.psi)
        Ekin = (g.dV / g.Ntot) * float(np.sum(g.k2 * np.abs(psik) ** 2))
        n = np.abs(self.psi) ** 2
        Epot = float(g.integrate(self.V * n))
        Eint = 0.5 * float(g.integrate(n * n))
        Edd = (0.5 * self.eps_dd * float(g.integrate(self._Phi_dd(n) * n))
               if self.eps_dd else 0.0)
        Elhy = (0.4 * self.gamma * float(g.integrate(n ** 2.5))
                if self.gamma else 0.0)
        tot = Ekin + Epot + Eint + Edd + Elhy
        return dict(kin=Ekin, pot=Epot, int=Eint, dd=Edd, lhy=Elhy,
                    total=tot)

    def H_psi(self):
        """Apply the (nonlinear) Hamiltonian to psi (for residuals)."""
        n = np.abs(self.psi) ** 2
        lap = _ifft(self.g.k2 * _fft(self.psi))
        return lap + self._W(n) * self.psi

    def mu_and_residual(self):
        """Chemical potential and stationarity residual ||(H-mu)psi||/||psi||."""
        Hp = self.H_psi()
        N = self.norm()
        mu = float(np.real(self.g.integrate(np.conj(self.psi) * Hp))) / N
        r = Hp - mu * self.psi
        num = np.sqrt(float(self.g.integrate(np.abs(r) ** 2)))
        den = np.sqrt(N)
        return mu, num / den
