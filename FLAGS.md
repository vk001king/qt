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

## Fixed in v1.4 (data-loss bug)
- [x] `mount_drive` silently fell back to ./drive_local when the Drive
      mount failed or the popup was not approved.  A full Colab session
      (validation + vortex pair + a 200 s crystal relaxation) was written
      to ephemeral disk and would have been destroyed on disconnect, with
      no warning in the output.  Mount failure inside Colab now RAISES
      by default (`require_drive=True`); the fallback must be requested
      explicitly.
- [x] Notebook cell 1 asserts `archive.persistent` before any simulation
      runs, so a non-persistent archive stops the notebook immediately.
- [x] Notebook cell 6 printed a hardcoded 'Drive location: MyDrive/...'
      even when output had gone to ephemeral disk -- it now prints the
      real path and the persistence flag.

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
- [x] DIAGNOSED (v1.5) why the crystal is not stationary.  Measured, not
      guessed:
        energy drift 1e-10, norm 1e-13 over t=6  -> solver is sound
        n_peak constant 19.6, contrast stable to 3.6e-3 -> droplets keep
          their shape; NOT droplet collapse, so LHY is adequate
        net crystal slide 0.058 xi = 0.8 percent of a -> NOT a rigid
          Goldstone translation either
        Bragg AMPLITUDES ||dA||/||A|| grow to 0.46 over t=6 -> individual
          droplets REARRANGE positions
      A perfect-triangular-lattice seed reaches mu = 5.233 versus 5.509
      from a noise seed, proving the noise-seeded state was defected and
      that neither is the true minimum.
      CONCLUSION: the energy landscape has many nearby minima and plain
      imaginary-time gradient flow from a single seed lands in the wrong
      one.  This is an OPTIMISATION problem, not a physics or solver bug.
      Next attempt should be a proper minimiser (conjugate-gradient or
      L-BFGS on the energy functional) and/or a multi-seed ensemble
      keeping the lowest mu, rather than more gradient-flow steps.
- [x] ADDRESSED in v1.6: direct energy minimisation implemented
      (`qtsim/minimize.py`).  L-BFGS-B on the energy functional with the
      norm handled by projection, plus a multi-start ensemble keeping the
      lowest mu.  Measured on the crystal:
        gradient flow, 24000 steps : mu = 5.509289, residual 2.28e-02
        single-start L-BFGS-B      : mu = 5.466331, residual 4.74e-03
        multi-start, lattice seed  : mu = 5.189268, residual 4.73e-03
      On a simple trapped problem the same minimiser drives the residual
      from 5.4e-02 to 7.5e-07 in 93 iterations (test T10b), so the
      optimiser is not the limitation.
- [x] Interpretation bug found and fixed: `a_measured = 2*pi/k_dominant`
      is the density MODULATION WAVELENGTH, not the triangular lattice
      constant.  The first Bragg vector of a triangular lattice has
      |k| = 4*pi/(a*sqrt3), so a = 2*lambda/sqrt(3) -- a factor 1.1547.
      Published 164Dy numbers quote the lattice constant, so conflating
      them would have corrupted the V4 comparison.  Both are now reported.
- [ ] **STILL OPEN: prove the global minimum.**  Even multi-start L-BFGS
      plateaus near residual 5e-3 on the crystal rather than the 1e-8
      target, and different seeds converge tightly into different basins
      (one reached residual 5.2e-05 at a HIGHER mu of 5.397).  So low
      residual does not imply low mu: the landscape is genuinely rugged.
      Next options: many more seeds; simulated annealing over eps_dd;
      constraining the unit cell and varying it explicitly; or comparing
      against a published lattice constant to decide which basin is
      physical.  Until one of these lands, V4 is not finished.
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
