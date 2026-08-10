# Open items

## Fixed in v1.1
- [x] Temporal-aliasing blow-up: solver now rejects dt with
      k_max^2*dt > 2 rad/step (T7 regression tests).
- [x] Notebook test harness no longer shortens loops, so long-time
      instabilities are actually exercised.
- [x] Ground-state modulation test now reports wavelength, not just
      contrast, and flags box-scale artifacts.
- [x] Stirring protocol rewritten: v1.0 produced zero vortices because
      the obstacle was 0.22*mu and the flow was Mach 0.35.  Now V0 = 3*mu
      at Mach 1.1, verified to nucleate a saturating tangle (~200 vortices)
      that then decays.  Drive evaluated at the substep midpoint.

## Known accuracy gap
- [ ] Dipolar ground-state stationarity residual is ~1e-2, not the 1e-8
      target.  Expected to be a symptom of the box-scale mode; recheck
      after the quasi-2D kernel lands (V4).

## v1.3 progress on validation rung V4
- [x] Quasi-2D projected dipolar kernel DERIVED and implemented:
      D(k) = 2*sqrt(2) - 3*sqrt(2*pi)*u*erfcx(u),  u = k*l_z/2,
      from integrating g_dd(3k_z^2/k^2 - 1) against the Gaussian axial
      density and dividing by g_2D = g/(sqrt(2*pi)*l_z).
      Verified against direct quadrature to 5e-15; both limits exact
      (D(0)=+2sqrt2 repulsive, D(inf)=-sqrt2 attractive).  The sign
      change is the roton, absent from the bare kernel.
- [x] Bogoliubov spectrum and roton locator added.
- [x] Confirmed a genuine triangular DROPLET CRYSTAL forms, contrast
      ~19.7, with period matching the roton prediction to 1.8 percent
      in a commensurate box (7.00 vs 7.12 xi).
- [ ] **OPEN: the crystal is not a converged stationary state.**
      Imaginary-time residual plateaus at 2.3e-2 (res/mu 4e-3) and does
      not improve with 24k further steps at dtau down to 5e-5.  Real-time
      evolution shows density drift of order the contrast itself
      (16 relative to mean over t=5), so the configuration is
      metastable/defected, not the ground state.  Candidate causes to
      investigate, in order: (i) insufficient annealing -- try a slow
      eps_dd ramp instead of noise seeding; (ii) l_z = 6 xi sits at the
      edge of quasi-2D validity, so the frozen-axial-mode assumption may
      be breaking; (iii) droplet collapse dynamics needing a stronger LHY
      term or three-body loss; (iv) box aspect ratio still frustrating
      the triangular lattice despite commensuration.
- [ ] Only after the above: compare the lattice constant to the published
      164Dy measurements.  That comparison is the actual content of V4
      and has NOT been done.

## Code
- [ ] Quasi-2D projected dipolar kernel (pancake geometry).  The current 2D
      kernel is the bare periodic symbol: fine for demonstration, but the
      demo ground state stays uniform rather than crystalline because of it.
      Required before validation rung V4.
- [ ] 3D vortex line tracking via Re(psi)=Im(psi)=0 zero-crossings,
      oriented by pseudo-vorticity.
- [ ] CuPy GPU backend (drop-in for the FFT calls in solver.py).
- [ ] Checkpoint/restart stress test across a real Colab disconnect.
- [ ] Truncation radius selection for strongly non-cubic domains
      (cylindrical-cutoff kernel as fallback).
- [ ] Adaptive embedded Runge-Kutta cross-check integrator.

## References (resolve at the publisher pages before submission)
| Ref | Issue |
|---|---|
| Barenghi, AVS Quantum Sci. 5, 025601 (2023) | DOI not captured |
| Saint-Michel (SHREK), arXiv:1401.7117 | published venue unconfirmed |
| Wlazlowski, PNAS Nexus 3, pgae160 (2024) | full author list |
| PRL 133, 243402 (2024) inverse cascade | first author to confirm |
| Nat. Phys. (2026) Kelvin-wave dispersion | author list |
| arXiv:2401.03548 | author list |
| Tang, PNAS 118, e2021957118 (2021) | partial author list |
| arXiv:0904.3440 dipolar PGPE methods | published venue |
| JLTP mutual friction review (2023) | author list |
