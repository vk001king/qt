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
