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

## Fixed in v1.6 (stale campaign)
- [x] `campaign/template_scan.py` was still importing and using
      `bare_dipolar_symbol` -- it never received the v1.3 quasi-2D kernel.
      The script that generates the paper's data was therefore simulating
      turbulence in a rippled superfluid rather than a supersolid, which
      defeats the point of the project.  It now takes `--kernel
      {quasi2d,bare}` and `--l_z`, defaults to quasi2d, prints the roton
      wavelength and the triangular lattice constant, and warns if the
      uniform state is NOT roton-unstable at the chosen parameters.
      `bare` is retained for regression only.
- [x] `test_notebook_cells.py` docstring still claimed loops were
      shortened; false since v1.2 and corrected.

## v2.4: GPU BACKEND (CuPy), EXPLICIT OPT-IN ONLY
Colab's T4 was enabled but idle: every module hardcoded `import numpy as
np`, and turning on a GPU runtime does nothing by itself.

- [x] `qtsim/backend.py` added: `Grid(..., backend="cpu"|"gpu")` selects
      NumPy or CuPy.  Default is "cpu" -- nothing changes unless requested.
      A "gpu" request with no CuPy/GPU present RAISES (RuntimeError) rather
      than silently continuing on CPU -- the same class of bug already
      fixed once for Google Drive persistence (a silent fallback there
      destroyed a full session's results with no warning in the output).
- [x] `kernels.py`, `solver.py`, `diagnostics.py`, `minimize.py` rewritten
      to use `grid.xp` / `grid.fft_mod` instead of a hardcoded `numpy`.
      Kernel construction (which needs scipy.special.erfcx, CPU-only) is
      built on the host once per run and moved to the device in a single
      transfer -- this is a one-time cost, unlike the per-step FFTs, so
      correctness mattered far more than avoiding the transfer.
- [x] `minimize.py`: scipy's L-BFGS-B is CPU-only.  The gradient's
      expensive part (FFTs, nonlinear terms) runs on the selected backend;
      only the packed vector is transferred to host once per iteration.
      This transfer is now the throughput floor for the MINIMISER
      specifically on GPU, and is documented as such rather than hidden --
      step_real/step_imag never leave the device and get the full speedup.
- [x] `campaign/track_a_scan.py` and `campaign/template_scan.py` both
      accept `--backend {cpu,gpu}`.
- [x] Verified: full CPU regression (all 41 prior checks) plus 4 new
      checks (T12) covering default-cpu, invalid-name rejection, the
      loud-failure guarantee, and a full pipeline (kernel + solver +
      minimiser + masked diagnostics together) on the explicit path.
      45/45 total.  CuPy/GPU itself could NOT be exercised in the sandbox
      this was built in (no GPU present) -- the actual speedup on a T4
      remains to be measured in Colab, not asserted here.
- [ ] **UNVERIFIED: has not run on a real GPU yet.**  The design is
      correct by construction (identical array API, explicit checks,
      CPU-path regression-tested) but CuPy's numerical results should be
      spot-checked against the CPU path once run in Colab -- FFT
      libraries can differ in the last few bits of floating-point
      rounding across backends, which matters for the tight residual
      tolerances used in T3/T10.

## v2.3: THE MASK SYSTEMATIC IS WORSE, AND WORST WHERE H4' NEEDS IT
Colab runs at real parameters produced three corrections.

- [x] **The mask rejection is 35-95 percent and TIME-DEPENDENT, not the
      constant 20-50 percent previously documented.**  Measured across one
      stirring run: raw 138 -> masked 14 (90% rejected) at t=5, then 294->74
      (75%), 316->147 (54%), 382->250 (35%).  The TRENDS differ too: raw
      grows 2.8x while masked grows 18x, so an L(t) built on raw counts has
      the wrong SHAPE, not merely an offset.  This damages H1 (threshold)
      and H5 (decay laws) directly, since both read L(t).
- [x] NEW DIAGNOSTIC: masked net charge as a mask-quality gate.  Raw counts
      are exactly charge-neutral (69/69, 147/147, 158/158, 191/191); masked
      are not (+8/-6, +33/-41, +72/-75, +124/-126).  True circulation is
      exactly zero, so the residual imbalance measures mis-clipping.
      Imbalance fell 14% -> 11% -> 2% -> 0.8% as nv grew 14 -> 250.
      `plaquette_charges_2d_masked` now returns charge_imbalance and
      mask_trustworthy (imbalance < 5%).
- [ ] **CONSEQUENCE FOR H4': the R>1 side is the hardest to measure.**
      At R = 4.65 the scan reported nv = 1 from raw 22 (95% rejected) with
      100 percent charge imbalance.  Since R ~ nv^{-1/2}, an ambiguity of
      1 versus 3 vortices is a factor 1.7 in R -- in the exact quantity the
      crossover is defined against.  Mitigation: larger boxes.
      nv(R=1) = cells^2*sqrt(3)/2, so cells=5 gives only nv=5 at R=2 while
      cells=12 gives nv=31 at R=2 and nv=14 at R=3.  Default --cells raised
      from 8 to 12; the scan now flags any row with nv<20 as untrustworthy
      and tells the user to re-run larger rather than interpret it.
- [x] CORRECTION to a claim I made: the crystal period agreement is ~5.0
      percent at full optimiser settings, NOT the 1.5 percent I quoted from
      a shortened local test (maxiter=1200, 2 seeds).  The full Colab run
      (maxiter=2500, 4 seeds) hit the iteration cap and gave 4.99 percent.
      The agreement is optimiser-dependent, which is itself worth reporting.

## v2.1: REAL PARAMETERS FIXED THE METASTABILITY; TWO BOX/LOGIC BUGS
Running cell 4 at the real experimental parameters produced the single most
consequential result so far, plus exposed two of our own bugs.

- [x] **THE METASTABILITY WAS LARGELY AN ARTEFACT OF eps_dd = 1.8.**
      At real parameters (eps_dd = 1.414, l_z = 8.6 xi) the crystal is
      CONVERGED and STATIONARY: residual 4.75e-06 (res/mu 9.1e-07, versus
      4.7e-03 before), energy drift 6.8e-13, Bragg amplitude change 7.0e-06,
      crystal slide exactly 0.00, contrast change 8.9e-07.
      Interpretation: eps_dd = 1.8 sits deep in the isolated-droplet regime
      where droplets are nearly independent and their arrangements nearly
      degenerate -- hence the rugged landscape.  At real supersolid
      parameters (contrast ~5, droplets still connected) the landscape is
      far better behaved.  So the v1.5-v1.7 investigation diagnosed a real
      effect but at unphysical parameters, and over-weighted it.  Multi-start
      still helps (lattice seed 5.2428 vs noise 5.264-5.272) but the margin
      is now 0.5 percent, not decisive.
- [x] **BOX-CONSTRUCTION BUG.**  The commensurate box used
      Lx = nx * (2*pi/k_rot), but 2*pi/k_rot is the modulation WAVELENGTH,
      not the droplet spacing d = 2*lambda/sqrt(3).  The in-row spacing was
      therefore too small by sqrt(3)/2 = 0.866, the crystal adopted the
      spacing the box allowed, and this manufactured a spurious 13.4 percent
      'period error'.  The measured/predicted ratio was exactly 0.866023 vs
      sqrt(3)/2 = 0.866025 -- six-digit confirmation.  Fixed: box now uses
      d_pred.  Period agreement went 13.4% -> 1.5%.
      Note Ly correctly used the sqrt(3)/2 row factor, so only the in-row
      spacing was wrong -- a partial error, easy to miss.
- [x] **VERDICT LOGIC BUG.**  The chain fell through to 'NO CRYSTAL at these
      parameters' whenever err >= 10%, so a converged stationary crystal with
      contrast 4.95 was reported as NO CRYSTAL while the diagnosis line said
      STATIONARY -- a flat self-contradiction in the output.  Crystal
      EXISTENCE and period ACCURACY are now reported as separate facts.
- [ ] **THE 1.5 PERCENT AGREEMENT IS PARTLY CIRCULAR.**  The box is built
      commensurate with the roton-predicted spacing, so the crystal cannot
      freely choose its period.  A non-circular test requires an
      incommensurate or much larger box in which several periods compete.
      This must be done before the agreement is claimed as validation.

## v2.0: PHASE 4 REDESIGNED (see PHASE4_REVISED.md)
The gap G1 survives intact and is now asserted by the primary source
itself.  What was falsified was the MECHANISM, and the hypotheses have been
rebuilt rather than merely flagged:
- H1 inverted, with the two mechanisms the sources give (reduced
  interstitial nucleation barrier; crystal quadrupole channel).  Partly a
  reproduction target now.
- H2 intact, but the roton and lattice scales are NOT independent
  (a = 2 lambda/sqrt(3), lambda = 2 pi/k_rot): one crystalline length.
- H3 reformulated: friction ansatz retained as phenomenology, but its
  ORIGIN is now an open choice between phase-gradient and density-barrier
  coupling, separable by measuring alpha_eff against f_s versus contrast.
  Pi DEMOTED from control parameter to diagnostic.
- H4 sharpened into the flagship claim, and a NEW contradiction C8 named:
  Alaña (smooth, no barrier, few vortices) versus Poli (glitches from
  unpinning) describe incompatible physics.  Hypothesis: geometric
  frustration at ell < a_L, when vortices outnumber interstitial sites,
  restores barrier dynamics.  Either outcome resolves C8.
- Geometry DECIDED: two tracks, Track A periodic extended lattice for
  statistics, Track B trapped few-droplet matching the experiments for
  V4/V5 and experimental contact.  Claims labelled by track.
- [ ] NOVELTY CHECK STILL OWED on the revised question (C8 / frustration
      crossover).  The original check tested a different question.

## v1.9: SECOND SOURCE READ -- PINNING PICTURE IS WRONG
Alaña, Modugno, Capuzzi, Jezek, arXiv:2405.05099 (2024), read in full.

- [ ] **THE PINNING NUMBER Pi MAY BE THE WRONG CONTROL PARAMETER.**
      Phase 5B built Pi = Delta_E_pin/E_l on the assumption that vortices
      sit in density minima behind an energy barrier.  For a ROTATING
      supersolid this paper finds vortices are NOT preferentially at
      density minima or saddles; their positions vary smoothly with drive
      frequency and are set by the RELATIVE PHASES of neighbouring
      droplets, not the density landscape.  Their model gives
      Y_v = (phi/pi + 2l + 1) pi hbar/(m d Omega).
      Consequences: the friction closure alpha_eff (H3) must be re-derived
      with the phase-gradient mechanism; H4's unpinning avalanches may have
      no barrier to avalanche over in the rotating case.  Whether a STIRRED
      TURBULENT state behaves like their STATIONARY rotating one is
      untested and is now the sharpest open question in the project.
- [x] CONFIRMED a = 2*lambda/sqrt(3) -- they state d = 2 lambda/sqrt(3)
      for the triangular lattice, identical to our v1.6 derivation.
- [x] CONFIRMED H1 inversion by a SECOND independent source: low-density
      valleys "reduce the energetic barrier for a vortex to enter the
      system, which lowers the nucleation frequency".
- [x] CONFIRMED multi-start is necessary: their note [56] says conjugate
      gradient on the eGPE "inherently yields local minima" and different
      trial wavefunctions give "nearly degenerate" lattice geometries.
      Exactly the v1.5 diagnosis and v1.6 fix, independently corroborated.
- [x] Plaquette method sourced: Foster, Blakie, Davis, PRA 81, 023623
      (2010), cited by them for the same purpose.
- [ ] TWO LHY CONVENTIONS in the literature: exact Re{Q5} (Casotti) versus
      closed form (1 + 3 eps_dd^2/2) (Alaña).  The closed form is 5 percent
      LOW across the experimental window (measured).  We use the exact
      integral; this must be stated since 5 percent in gamma shifts the
      already-thin roton margin.

## v1.8: LITERATURE READ, THREE CORRECTIONS
See EXPERIMENTAL_PARAMETERS.md for the full extraction and citations.
Casotti et al., Nature 635, 327 (2024) was read in full (Methods included),
not abstract-only as in Phase 1.

- [ ] **H1 IS BACKWARDS AND MUST BE INVERTED.**  We hypothesised that the
      supersolid tangle threshold is RAISED by interstitial pinning.  The
      paper reports, in both experiment and its own eGPE, that the
      supersolid nucleates vortices at SIGNIFICANTLY LOWER rotation than
      the BEC (Omega*_SSP ~ 0.25-0.45 vs Omega*_BEC ~ 0.6 omega_perp),
      because the near-degenerate CRYSTAL quadrupole mode opens an extra
      angular-momentum channel.  Pinning governs vortex motion and decay
      (H3-H5), not the nucleation threshold.  The article design needs
      revising before any campaign is run against H1.
- [x] eps_dd = 1.8 used in every crystal run is OUTSIDE the experimental
      supersolid window.  Real: a_s = 90-95 a0 with a_dd = 130.8 a0, i.e.
      eps_dd = 1.377-1.453.  Our 1.8 means a_s = 72.7 a0 -- the isolated
      droplet regime.  Recomputed at real values the roton survives but
      min(inside) is only -0.03 to -0.19 (vs -0.60 at 1.8) and vanishes
      at higher density: crystal existence is density-sensitive, which
      eps_dd = 1.8 hid completely.
- [x] l_z = 6 xi was chosen by scanning, not derived.  Real trap plus
      plausible densities give l_z/xi ~ 6-19, so 6 is inside the range but
      only by luck; l_z/xi must be quoted with the density it assumes.
- [x] **Q5 CONVENTION FLAG CLOSED.**  The paper states
      Q_n(x) = int_0^1 (1-x+3xu^2)^{n/2} du with Re{} for x>1, and
      gamma_QF = (128 hbar^2/3m) sqrt(pi a_s^5) Re{Q5}.  Both are
      algebraically identical to our implementation.  Now sourced.
- [x] Our v1.7 vortex-count systematic is INDEPENDENTLY CONFIRMED: they
      mask to a 6 um circle and state that varying their threshold changes
      absolute counts but not qualitative results.  Report L as trends
      with a quoted systematic, never as a single absolute number.

## Fixed in v1.7 (vortex counting in a crystal)
- [x] The raw plaquette detector invents vortices in the near-vacuum
      inter-droplet regions of a crystal, where the phase is numerical
      noise (measured n_min ~ 1e-6 n_mean).  `plaquette_charges_2d_masked`
      adds an ANNULUS density criterion -- a plain density mask is wrong
      because a real vortex core also has n -> 0 at its centre.
      Validated on a fluid/void control: 97.4 percent of void false
      positives rejected, real vortex kept, no-op on a clean pair (T11).
- [ ] **REPORT THIS AS A SYSTEMATIC.**  On a stirred quasi-2D crystal the
      count was 874 raw, 724 annulus-masked (-17 percent), 459 with a crude
      corner mask (-47 percent).  So vortex line density L in a droplet
      crystal carries a METHOD-DEPENDENT SYSTEMATIC of order 20-50 percent.
      L is the primary observable for the threshold (H1), the friction
      budget (H3), the avalanche statistics (H4) and the decay laws (H5),
      so every reported L must carry this systematic.  The campaign now
      records raw AND masked counts plus the reject fraction so the
      systematic is measurable per run rather than assumed.
      The residual 2.6 percent of false positives sit on the fluid/void
      interface and no parameter choice removes them.

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

## v2.4: GPU BACKEND (CuPy), AND A REAL TEST-HARNESS REGRESSION FOUND

### GPU support added
- [x] `qtsim/backend.py` selects NumPy (CPU, default) or CuPy (GPU)
      EXPLICITLY -- `backend="gpu"` must be requested, and raises loudly if
      CuPy/CUDA is unavailable rather than silently falling back to CPU.
      Same "explicit, not automatic" principle as the Drive-persistence fix
      (v1.4): a person must never mistake a CPU run for a GPU one.
- [x] `Grid`, `EGPESolver`, `diagnostics.py`, `minimize.py` all thread the
      array module through as `grid.xp`/`grid.fft_mod` rather than a
      hardcoded `numpy` import.  `campaign/template_scan.py` and
      `campaign/track_a_scan.py` expose `--backend {cpu,gpu}`.
- [x] VERIFIED WITHOUT REAL HARDWARE: a fake `cupy` module (NumPy under the
      hood) was injected into `sys.modules` to exercise the `backend="gpu"`
      code path end to end -- grid construction, imaginary/real time
      stepping, energy, the L-BFGS-B minimiser, both vortex detectors, and
      Helmholtz decomposition.  Every quantity matched the real CPU path to
      EXACTLY zero difference (not just within tolerance), proving the
      xp-threading has no missed hardcoded `np.` call that would silently
      stay on CPU semantics.  This does NOT test real CUDA numerics or
      timing; Cell 7 of the notebook is the honest, undone check against
      real hardware, and states plainly that it was never run against one.
- [ ] **NOT YET DONE: run Cell 7 against a real GPU.**  Speedup, real CUDA
      numerical agreement (last-ULP differences from different reduction
      order ARE expected there and must be distinguished from a real bug),
      and the actual throughput floor from the per-iteration host<->device
      transfer in the minimiser are all unmeasured until this happens.

### Regression found and fixed: the notebook test harness was broken since v1.4
- [x] **`test_notebook_cells.py` could not get past Cell 1 outside Colab,
      for every version from v1.4 through v2.3.**  Cell 1's Drive-
      persistence assertion (added in v1.4 to stop the ephemeral-storage
      data-loss bug) correctly raises when no Google Drive is mounted --
      which is ALWAYS the case outside Colab.  Nothing in the harness told
      it this was expected.  During that whole span, "full notebook"
      verification claims in this file were based on ad hoc per-cell exec
      scripts that pre-injected `archive` as a global and skipped Cell 1
      entirely -- not on running this sanctioned, sequential harness.  So
      individual cells 2-6 WERE genuinely exercised each time, but the
      claim of "verified end to end" for the complete cell-1-through-6
      sequence had not actually been true since v1.3.
      Fixed via an explicit, env-gated bypass (`QTSIM_SANDBOX_TEST=1`, set
      only by this harness, never inferred) that overrides the
      `persistent` flag without changing where files are written or
      touching the real Colab code path at all.
- [x] **Found while fixing it: Cell 1 had two separate, overlapping
      persistence checks that were never consolidated** -- a raw
      `assert archive.project_dir.startswith('/content/drive/MyDrive')`
      (which hardcodes a literal path prefix and bypasses the `persistent`
      abstraction entirely) alongside the correct, already-documented
      `if not archive.persistent: raise RuntimeError(...)`.  These were
      added in different edit passes and left duplicated.  The redundant,
      path-hardcoded assert was removed; only the flag-based check remains.
- [x] Fixed the SystemExit-crashes-the-harness bug this exposed: the
      optional GPU cell (Cell 7) legitimately raises `SystemExit` when
      CuPy is absent, and `SystemExit` is `BaseException`, not `Exception`
      -- it would have escaped the harness's `except Exception:` and
      killed the whole script uncontrolled rather than reporting a clean
      skip.  The harness now catches `SystemExit` specifically for the
      recognized optional cell and reports SKIPPED, distinct from PASS/FAIL.
- [x] Full harness re-run end to end after all fixes, for the first time
      since v1.3: **6/6 mandatory cells passed, 1 skipped (GPU, as
      designed), exit code 0.**
