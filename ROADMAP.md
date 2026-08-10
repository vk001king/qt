# Roadmap to publication

Current position: **Step 1 complete** — solver built, 38/38 validation checks
passing, all notebook cells execution-tested.

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
