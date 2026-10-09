# Roadmap to publication

Current position (v2.6): **Track A 12-cell scan complete (24/24 clean
points) with the surrogate null test; crossover gap (R 0.8-1.3) and long
decay still to run.**  Earlier: **Step 1 complete** — solver built, now
50/50 validation checks passing on CPU and on a real Colab T4.

## Step 1.5 — DESIGN REPAIR (new, blocking)   ~1-2 weeks
The literature reading invalidated two hypotheses and the pinning control
parameter.  PHASE4_REVISED.md rebuilds the design.  Remaining:
- [ ] Novelty check on the REVISED question (C8 / frustration crossover)
- [ ] Obtain Poli et al. PRL 131, 223401 full text (other half of C8)
- [ ] Obtain a measured droplet spacing to close V4
- [ ] Implement Track B trapped few-droplet geometry

## Step 1.97 — Track A analysis fixes (v2.6)
- [x] Resume key on the full physical-parameter set (finding F2)
- [x] F1 confirmed: the two 8-cell rows are `--quick` smoke-test runs
- [x] `--report`: filter on stored params, box check, CSV
- [x] Formal null test: phase-randomised + IAAFT surrogates, p per run
- [x] MAD == 0 guard in `detect_avalanches` (finding F3)
- [x] Colab T4: the two missing 12-cell points (Ma 0.6 / 2.4, seed 0)
      re-run (2026-10-09, 3.9 min, not skipped by the new key); 24-row
      table rebuilt: 0 of 24 runs with p_phase(theta=3) < 0.05
- [ ] Colab Cell 8: gap-fill drives 0.65-0.80 x 3 seeds (R 0.8-1.3)
- [ ] Colab Cell 8: T_decay = 600 for Ma 0.6, 0.9, 1.3 + gap drives

## Step 1.95 — track_a_scan.py GPU port (v2.5, fixed on a real crash)
- [x] Five host/device mixing bugs found and fixed in the campaign script
      itself (v2.4 only ported the qtsim package, not the scripts)
- [x] Confirmed on a real Colab T4 (2026-10-09): `--quick --backend gpu`
      and two production 12-cell points ran clean (dE/E <= 1e-7)

## Step 1.9 — GPU backend (v2.4, done except real-hardware check)
- [x] CuPy backend implemented, explicit opt-in, verified against a fake
      GPU module to prove the abstraction has no gaps
- [x] Fixed a real regression: the sequential notebook test harness had
      been silently broken since v1.4 outside Colab
- [ ] Run Cell 9 (was Cell 7 before v2.6) on real Colab GPU hardware;
      record a controlled CPU-vs-GPU speedup.  Indirect evidence only so
      far: stored Track A wall times are ~1280 s (4 runs) versus ~120 s
      (18 runs) per 12-cell point, i.e. ~10.7x, but v2.5 runs did not
      record their backend, so the split is inferred from timing
- [x] Track A run at `--backend gpu` on a T4 (~2 min per 12-cell point)

## Step 1.75 — TRACK A: the H4' scan (READY TO RUN)   ~2-4 weeks
`campaign/track_a_scan.py` implements the frustration-crossover experiment
that decides contradiction C8.  Calibrated and verified at the default box (cells=12, 4 box-scaled
obstacles): drives 0.4 -> 4.2 span R from above 2 down to ~0.5, with masked
vortex counts of 28 to 391 and mask charge imbalance mostly under 4 percent.

    python campaign/track_a_scan.py --quick     # smoke test, ~1 min
    python campaign/track_a_scan.py             # full scan, 8 drives x 3 seeds

Resumable: completed parameter points are skipped, so a Colab disconnect
costs only the run in progress.  Every run records raw AND masked vortex
counts, avalanche statistics at three detection thresholds, and energy/norm
drift as a health check.  Rows flagged not-conservative must be discarded.

- [ ] Run the full scan at cells=12, then repeat at cells=8 and 16 for
      finite-size scaling (avalanche exponents are meaningless without it)
- [ ] Recalibrate the drive list for each box size -- vortex yield depends
      on drives, V0_factor, n_stir AND cells jointly, and does not transfer.
      Yield is also non-monotonic at high drive (Ma=3.2 -> nv=391 but
      Ma=4.2 -> nv=174), so read R from the table, never infer it.
- [ ] Discard any row with nv < 20 on the R>1 side: the masked count is not
      trustworthy there and R scales as nv^{-1/2}.
- [ ] Decide C8 from the event-count-versus-R trend
- [ ] Test an INCOMMENSURATE box: the 1.5% period agreement is currently
      partly circular because the box imposes the roton spacing

## Step 2 — Platform validation (V3–V5)   ~3–5 weeks
- [ ] V3  Code-to-code comparison against published dipolar GPE suites
- [ ] V4  Reproduce the 164Dy supersolid transition point and lattice constant
- [ ] V5  Reproduce glitch spin-up phenomenology (Poli et al., PRL 2023)
- [x] Implement the quasi-2D projected dipolar kernel (done in v1.3,
      verified to 5e-15, produces a droplet crystal at the roton period)
- [x] Diagnose the metastable crystal (v1.5): ruled out solver error,
      droplet collapse and Goldstone translation by measurement; it is
      droplet rearrangement, i.e. an optimisation problem
- [x] Implement direct minimisation (v1.6): multi-start L-BFGS-B lowers
      mu from 5.509 to 5.189
- [ ] Prove the global minimum (residual still ~5e-3, rugged landscape) --
      see FLAGS.md for the shortlist of next attempts

## Step 3 — Turbulence regression (V6)   ~1 week
- [ ] eps_dd -> 0, gamma -> 0 must recover Kolmogorov k^-5/3 and Vinen L ~ t^-1

## Step 4 — Quasi-2D production campaign   ~8–12 weeks GPU
- [ ] ~8 eps_dd values x ~5 drive amplitudes x 10–20 seeds
- [ ] Collect L(t), spectra, energy channels, avalanche time series
- [ ] Figures 1–5

## Step 5 — Three-dimensional confirmation   ~4–8 weeks GPU
- [ ] 2–3 representative parameter points at 256^3
- [ ] 3D vortex line tracking (Re/Im zero-crossings)
- [ ] Figure 7

## Step 6 — Analysis and hypothesis testing   ~3–4 weeks
- [ ] Test H1–H5 with bootstrap error bars
- [ ] Extract alpha_eff(eps_dd, f_s)
- [ ] Avalanche statistics with finite-size scaling

## Step 7 — Manuscript completion   ~3 weeks
- [ ] Fill every Abstract and Section V placeholder from data
- [ ] Discussion and Conclusion
- [ ] Resolve the nine reference flags in FLAGS.md
- [ ] LaTeX typesetting (PRX format)

## Step 8 — Release and submission   ~2 weeks
- [ ] Archive code and data on Zenodo with DOI
- [ ] Cover letter and reviewer suggestions
- [ ] Submit

Publication readiness: **52%**.  The campaign in Step 4 is the single
largest remaining block and gates everything after it.
