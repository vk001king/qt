# Experimental parameters and what they corrected

Source, read in full (not abstract-only):
**E. Casotti, E. Poli, L. Klaus, A. Litvinov, C. Ulm, C. Politi, M. J. Mark,
T. Bland, F. Ferlaino, "Observation of vortices in a dipolar supersolid",
Nature 635, 327-331 (2024)**, DOI 10.1038/s41586-024-08149-7,
arXiv:2403.18510 (Methods section).

## The numbers

| quantity | experimental value | source |
|---|---|---|
| isotope | 164Dy | Methods |
| dipolar length | a_dd = 130.8 a0 (exact, stated) | Methods |
| supersolid scattering length | a_s = 90-95 a0 | Methods |
| **supersolid interaction ratio** | **eps_dd >= 1.37** (1.377-1.453) | Methods |
| BEC/SSP boundary | eps_dd ~ 1.3 | main text |
| radial trap | omega_perp = 2pi x 50.3(2) Hz | Methods |
| axial trap | omega_z = 2pi x 103-135 Hz, aspect 2-3 | figure captions |
| atom number | N = 3e4 to 7e4 | Methods |
| field tilt | theta = 30 deg | Methods |
| three-body loss | L3 = 1.2e-41 m^6 s^-1 | Methods |
| thermal noise | T = 20 nK, Bogoliubov modes with eps_n <= 2 k_B T | Methods |
| droplet number | typically 4 (sometimes 3) | Methods |
| BEC nucleation threshold | Omega*_BEC ~ 0.6 omega_perp | main text |
| SSP nucleation thresholds | weak resonance ~0.25-0.35, threshold response from ~0.45 omega_perp | main text |

## Correction 1 (MAJOR): hypothesis H1 was backwards

Our design stated H1 as: *a tangle is sustainable above a critical drive,
with a threshold ELEVATED relative to the unmodulated superfluid, because
interstitial pinning suppresses free vortex transport.*

The paper reports the opposite, in both experiment and its own eGPE
simulations: the supersolid is **more** susceptible to vortex creation,
nucleating at **significantly lower** rotation frequency than the BEC
(Omega*_SSP ~ 0.25-0.45 omega_perp versus Omega*_BEC ~ 0.6 omega_perp).
Their stated mechanism is not pinning at all: a two-dimensional supersolid
has three quadrupole modes -- one from broken phase symmetry and one from
each broken translational direction -- and the near-degenerate *crystal*
quadrupole resonance opens an extra channel for angular-momentum transfer,
giving a monotonic, rigid-body-like increase of vortex number with Omega.

H1 must therefore be inverted, and its mechanism replaced.  Pinning is
still expected to matter, but for vortex *motion and decay* (H3-H5), not
for the nucleation threshold.

## Correction 2: eps_dd = 1.8 was outside the supersolid phase

All our crystal runs used eps_dd = 1.8, which corresponds to a_s = 72.7 a0
-- far below the experimental supersolid window of 90-95 a0.  That is the
isolated-droplet regime, not a supersolid.

Recomputed at the real values (eps_dd = 1.377-1.453, l_z/xi from the real
trap), the roton instability survives but is much weaker:

| eps_dd | a_s/a0 | l_z/xi | min(inside) | unstable | lambda/xi |
|---|---|---|---|---|---|
| 1.453 | 90.0 | 8.46 | -0.188 | yes | 8.77 |
| 1.414 | 92.5 | 8.58 | -0.108 | yes | 8.90 |
| 1.377 | 95.0 | 8.69 | -0.031 | yes | 9.02 |

compared with min(inside) = -0.60 at the unphysical eps_dd = 1.8.  So a
crystal does still form at experimental parameters, but the margin is thin
and it disappears entirely at higher peak density (n0 = 5e21 m^-3 gives
min(inside) > 0).  Crystal existence is therefore density-sensitive in a
way our eps_dd = 1.8 runs completely hid.

Predicted modulation wavelength at real parameters: lambda ~ 8.8-9.0 xi,
i.e. ~0.79-0.81 um for xi ~ 90 nm, giving a triangular lattice constant
a = 2 lambda / sqrt(3) ~ 0.91-0.94 um.  **The paper does not state a
droplet spacing numerically**, so this prediction is not yet checkable
against it; extracting the spacing from their Fig. 2 or the Zenodo dataset
(doi 10.5281/zenodo.10695943) is the remaining step for V4.

## Correction 3: l_z = 6 xi was lucky, not derived

We chose l_z = 6 xi by scanning until a roton appeared.  With the real trap
(omega_z = 2pi x 103 Hz) and plausible peak densities, l_z/xi ranges from
about 6 at n0 = 5e20 m^-3 to 19 at n0 = 5e21 m^-3.  So 6 does sit inside
the physical range -- but by luck.  The honest statement is that l_z/xi is
set by omega_z and n0 together and must be quoted with the density.

## Confirmed correct

* **Q5 and the LHY coefficient.**  The paper gives
  Q_n(x) = int_0^1 du (1 - x + 3 x u^2)^{n/2} and
  gamma_QF = (128 hbar^2 / 3m) sqrt(pi a_s^5) Re{Q5(eps_dd)}.
  Our Q5 integrand [1 + eps(3u^2 - 1)]^{5/2} is algebraically identical,
  and our gamma = (32/(3 sqrt pi)) g a_s^{3/2} Q5 reduces to their
  prefactor exactly.  **The Re{} convention for eps_dd > 1 is now sourced**
  -- that flag is closed.
* **Vortex-count systematics.**  They restrict counting to a circle of
  radius 6 um (a spatial mask, like our annulus mask) and state that
  varying their detection threshold from 0.34 to 0.42 "modifies the
  absolute vortex count of each individual image but not the overall
  qualitative result".  So the experimentalists also treat absolute counts
  as unreliable and argue from trends.  This independently validates the
  20-50 percent systematic we measured in v1.7, and tells us how to report
  L: as trends with a quoted systematic, never as a single absolute number.
* **The competing-length-scales motivation.**  Their conclusion names
  exactly our organising axis -- vortex separation, crystal wavelength, and
  core diameter -- and anticipates "constrained motion and pinning to
  avalanche escape" as unique to supersolids.  The gap is real and stated
  by the source itself.

## Still not obtained

* Experimental droplet spacing as a number (needed to finish V4).
* Poli et al. PRL 131, 223401 (2023) glitch parameters (needed for V5).
* Whether a periodic extended crystal is the right idealisation at all:
  the experiment has FOUR droplets, a finite mesoscopic crystal, whereas
  our periodic box models an extended lattice.  These may not be the same
  physics and the discrepancy is not yet assessed.
* Three-body loss (L3) and the T = 20 nK Bogoliubov noise prescription are
  in the paper and used by them; our solver has L3 as an unused optional
  switch and seeds noise with an arbitrary amplitude instead.

---

# Second and third sources read in full

## Alaña, Modugno, Capuzzi, Jezek, "Phase-induced vortex pinning in
## rotating supersolid dipolar systems", arXiv:2405.05099 (2024)

Read in full (HTML).  Parameters: N = 1.1e5 of 162Dy, omega = 2pi x {60,120}
Hz, a_dd = 130 a0, a_s = 92 a0 fixed, T = 0, box 10x10x12 um on a
{256,256,64} grid, energy minimised by conjugate gradient.

### CORRECTION 4 (MAJOR): our pinning picture is wrong

Our Phase 5B introduced a pinning number Pi = Delta_E_pin / E_l, with
Delta_E_pin the energy difference between a vortex placed interstitially
and on a droplet -- i.e. we assumed vortices sit in DENSITY MINIMA held by
an energy barrier, and that turbulence would be governed by hopping over
that barrier.

This paper states the opposite for a ROTATING supersolid.  Vortices are
"not only pinned at local density minima, but instead their coordinates are
smooth functions of the rotation frequency"; explicitly, "the pinning at
the saddle points ... and density minima do not seem to be favored with
respect to other points along such paths.  Instead, the vortex position
smoothly changes as a function of the rotation frequency.  As a matter of
fact, in a rotating supersolid, the slow variation of the vortex location
arises from the imprinted velocity field on the droplets, rather than from
density holes that typically pin vortices in non-rotating systems."

Their quantitative model sets the vortex position from the RELATIVE PHASES
of neighbouring droplets:
    Y_v = (phi/pi + 2l + 1) pi hbar / (m d Omega)          (two droplets)
    sqrt(N0/N1) exp[d(d - sqrt3 Y_v)/(2a^2)] + 2 cos(m d Omega Y_v / 2 hbar) = 0
                                                            (three droplets)
with d the inter-droplet distance.  Three neighbours are needed near a
lattice vertex; two suffice near a saddle.

Consequence for us: Pi is at best incomplete and at worst the wrong
control parameter.  Any friction closure alpha_eff built on a
density-barrier picture (H3) must be re-derived with the phase-gradient
mechanism included, and H4's "unpinning avalanches" need re-examination --
if vortex position is a smooth function of drive, there may be no barrier
to avalanche over in the rotating case.  Whether a stirred TURBULENT state
behaves like their stationary rotating one is untested and is now the
sharpest open question in the project.

### CONFIRMATION 1: lattice constant conversion is right

They state d = 2 lambda / sqrt(3) for the triangular droplet lattice.  This
is exactly the conversion we derived independently in v1.6 after finding we
had been conflating modulation wavelength with lattice constant.  Verified.

### CONFIRMATION 2: H1 inversion, second independent source

"given that droplets are separated by low density valleys, the barrier
required for the nucleation of vortices is reduced with respect to the
superfluid case" and "low-density regions reduce the energetic barrier for
a vortex to enter the system, which lowers the nucleation frequency".
So both Casotti et al. (experiment + eGPE) and Alaña et al. (theory) agree
the supersolid nucleates vortices MORE easily.  H1 as originally written is
definitively wrong.

### CONFIRMATION 3: multi-start minimisation is necessary, not optional

Their note [56]: "the conjugate gradient approach employed to minimize the
eGP energy functional inherently yields local minima.  In this context,
employing different trial wave functions can generate alternative lattice
geometry configurations that are nearly degenerate in energy."

This is precisely the failure we diagnosed in v1.5 and fixed in v1.6 with
multi-start L-BFGS.  Independent confirmation that the rugged landscape is
a real property of the eGPE functional and not a bug in our solver.  They
also use a gradient-based minimiser rather than imaginary time, as we now
do.

### CONFIRMATION 4: the plaquette method is standard, with a citation

They "extract the positions of any vortices present in the system using a
plaquette method", citing C. J. Foster, P. B. Blakie, M. J. Davis, "Vortex
pairing in two-dimensional Bose gases", Phys. Rev. A 81, 023623 (2010).
Our detector matches standard practice and now has a source.

### DISCREPANCY 1: two LHY conventions are in use

Alaña et al. use the closed form
    gamma_LHY = (128 sqrt(pi) hbar^2 a_s^{5/2} / 3m) (1 + 3 eps_dd^2 / 2),
whereas Casotti et al. use the exact Re{Q5(eps_dd)}.  Measured difference:

| eps_dd | Re Q5 exact | 1 + 3 eps^2/2 | ratio |
|---|---|---|---|
| 1.000 | 2.59808 | 2.50000 | 0.962 |
| 1.377 | 4.05017 | 3.84419 | 0.949 |
| 1.414 | 4.21746 | 3.99909 | 0.948 |
| 1.453 | 4.39868 | 4.16681 | 0.947 |

The closed form is 5 percent LOW across the experimental window.  We use
the exact integral (as Casotti et al. do).  This must be stated, because a
5 percent shift in gamma moves the roton margin, which we already showed is
thin at real parameters.

## Poli et al., "Glitches in rotating supersolids", PRL 131, 223401 (2023)

Only the abstract and a figure caption were retrievable; the full text is
paywalled and the arXiv PDF did not extract.  From the caption of Fig. S1:
N = 3e5, a_s = 90 a0, omega = 2pi x (50, 130) Hz, and the figure shows the
"pinning force felt by a single vortex in a supersolid" with vortex
equilibrium positions and saddle points marked.  That figure is exactly the
quantity H3 needs, and we do not have it.

**This is the single most valuable missing item.**  If you can obtain the
full text of PRL 131, 223401 (2023) -- or its Supplemental Material -- that
would let us (a) reproduce their pinning-force map as validation rung V5,
and (b) settle whether the density-barrier or phase-gradient picture governs
the non-rotating/stirred case.

## Still needed

1. **Poli et al. PRL 131, 223401 (2023)** full text + Supplemental. (V5)
2. **Norcia et al., Nature 596, 357 (2021)**, "Two-dimensional supersolidity
   in a dipolar quantum gas" -- expected to state the measured droplet
   spacing, which is what V4 needs and which Casotti et al. do not give.
3. **Bland et al., PRL 128, 195302 (2022)**, "Two-dimensional supersolid
   formation in dipolar condensates" -- lattice geometry and spacing.
4. **Casotti et al. Zenodo dataset**, doi 10.5281/zenodo.10695943 -- may
   contain the density profiles from which spacing can be measured directly.
5. **Poli et al., "Synchronization in rotating supersolids"**,
   arXiv:2412.11976 -- crystal-vortex coupling, directly relevant to H3.
