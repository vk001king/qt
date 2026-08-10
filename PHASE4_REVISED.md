# Phase 4, revised against the literature actually read

Written after reading Casotti et al. (Nature 635, 327, 2024) and Alaña et
al. (arXiv:2405.05099) in full.  See EXPERIMENTAL_PARAMETERS.md for the
extraction.  This supersedes the original Phase 4 hypotheses.

## 1. The gap survives unchanged

G1 rested on four evidence points and all four still hold:

| claim | source | status |
|---|---|---|
| vortices exist in a dipolar supersolid, seeding differently from unmodulated fluids | #26 Casotti | CONFIRMED, read in full |
| wave turbulence exists across the SF-SS transition; the vortex sector is untouched | #27 Bougas | holds |
| pinning/unpinning characterised only for few-vortex glitch events | #44 Poli, #45 Bland | holds |
| no statistical tangle, spectra, decay laws, or lattice dissipation channel | corpus + occupancy check | holds |

The gap is now also asserted by the primary source itself.  Casotti et al.
close with: a "fascinating interplay of competing length scales emerges.
These include the separation between vortices, the wavelength of the
self-forming crystal, and the diameter of the vortex core.  This
competition has the potential to lead to intriguing dynamics, ranging from
constrained motion and pinning to avalanche escape.  These phenomena are
genuinely unique to supersolids."  That is our organising axis and our
target phenomenology, named by the experiment and left unstudied.

**What changed is the MECHANISM, not the gap.**  The original hypotheses
assumed a density-barrier picture of pinning that the literature
contradicts.  Below they are restated with the mechanisms the sources
support.

## 2. A NEW contradiction in the literature, which the project can resolve

This is the most valuable thing the reading produced.

* **Alaña et al. (rotating, stationary, few vortices, lattice symmetry
  imposed):** vortices are NOT preferentially at density minima or saddles.
  Their positions vary smoothly with drive frequency and are set by the
  relative PHASES of neighbouring droplets:
  Y_v = (phi/pi + 2l + 1) pi hbar/(m d Omega).  There is no barrier and
  hence nothing to unpin from.
* **Poli et al. (spin-down, glitches):** vortex UNPINNING produces discrete
  glitch events, i.e. there IS a barrier and vortices escape it abruptly.

Both are eGPE studies of dipolar supersolids and they describe incompatible
physics.  Label this **C8**.

Plausible reconciliation, and the reason our regime matters: Alaña et al.
study a stationary state with FEWER vortices than interstitial sites, where
each vortex can sit at its phase-preferred position.  Poli et al. drive the
system out of that state.  A dense turbulent tangle has ell << a_L, i.e.
MORE vortices than interstitial sites -- geometric frustration.  When
vortices cannot all occupy phase-preferred positions, barrier-like
behaviour and avalanches may be restored.

**This makes C8 a falsifiable, mechanistically sharp question that only a
turbulent-tangle study can answer.**  It is a stronger scientific hook than
the original "pinning constrains turbulence" framing, because it resolves a
live disagreement rather than confirming an assumption.

## 3. Revised hypotheses

### H1' (INVERTED) -- threshold is LOWERED, not raised
The supersolid sustains a vortex tangle at LOWER drive than the unmodulated
superfluid at matched sound speed, for two reasons the sources give:
(a) low-density interstitial paths reduce the nucleation barrier
    [Alaña: "low-density regions reduce the energetic barrier for a vortex
    to enter the system, which lowers the nucleation frequency"];
(b) a 2D supersolid has three quadrupole modes, and the near-degenerate
    CRYSTAL quadrupole resonance opens an extra angular-momentum channel,
    giving a monotonic rigid-body-like rise in vortex number
    [Casotti: Omega*_SSP ~ 0.25-0.45 vs Omega*_BEC ~ 0.6 omega_perp].
*Test:* steady-state L versus drive amplitude, scanned across eps_dd
through the SF-SS transition at matched sound speed.
*Falsifier:* threshold equal or higher in the supersolid.
*Note:* this is now partly a REPRODUCTION target, not a discovery -- the
few-vortex version is established.  Our contribution is whether it survives
into the many-vortex turbulent regime, where the extra channel may saturate.

### H2 (largely intact, one simplification)
A quasiclassical range appears only while ell > a_L; for ell <~ a_L the
spectrum crosses to Vinen/strong-QT form with a feature at the crystal
scale.
*Simplification found:* the roton scale and the lattice scale are NOT
independent.  With a = 2 lambda/sqrt(3) and lambda = 2 pi/k_rot, there is
ONE crystalline length, set by the roton.  The original design implicitly
treated them separately.  R_ell = ell/a_L remains the organising parameter.

### H3' (REFORMULATED) -- friction ansatz survives, its origin does not
The question is no longer "how high is the hopping barrier" but: **in a
dense tangle, does the crystal absorb vortex energy, and can that transfer
be written as a friction?**  Two candidate mechanisms, now to be
distinguished rather than assumed:
(i) *phase-gradient coupling* [Alaña]: vortex positions are tied to droplet
    phases, so moving vortices must reorganise those phases, exchanging
    energy with the crystal;
(ii) *density-barrier coupling* [Poli]: vortices interact with the density
    landscape and dissipate on unpinning.
The friction ansatz f = -D(v_L - v_lat)_perp - D_t s' x (v_L - v_lat) with
alpha_eff = D/(rho_s kappa) is retained -- it is a phenomenological form,
agnostic to origin -- but alpha_eff must be MEASURED, and the two
mechanisms give different predicted scalings: (i) should scale with the
phase stiffness and hence with superfluid fraction f_s, (ii) with the
density contrast.  Measuring alpha_eff against BOTH f_s and contrast
separates them.
*Retired:* Pi = Delta_E_pin/E_ell as the primary control parameter.  It
presumes mechanism (ii).  It is demoted to a diagnostic reported alongside
a phase-stiffness analogue, not used to organise the results.

### H4' (SHARPENED, now the flagship claim)
At low vortex density the smooth phase-set positioning of Alaña et al.
should hold and there should be NO avalanches.  As R_ell decreases below
unity, geometric frustration sets in -- more vortices than interstitial
sites -- and we hypothesise a crossover to barrier-dominated, avalanching
dynamics with heavy-tailed event sizes.
*Test:* avalanche statistics as a function of R_ell, with finite-size
scaling; the crossover location is the headline number.
*Falsifier:* smooth dynamics at all R_ell (Alaña wins, glitches are a
spin-down-only phenomenon) or avalanches at all R_ell (Poli wins,
Alaña's smoothness is an artefact of imposed lattice symmetry).
*Either outcome resolves C8.*  This is the strongest part of the revised
design because it cannot fail to be informative.

### H5' (intact)
L(t) departs from the canonical t^-1 / t^-3/2 forms, with plateaus
punctuated by avalanche drops, reverting to canonical behaviour deep in the
superfluid phase.  Now conditional on H4': if there are no avalanches,
H5' predicts smooth but modified decay, and the modification is then
attributable to friction alone.

## 4. Geometry: a decision, finally made

The experiments are FINITE: Casotti et al. have typically four droplets,
Alaña et al. seven in a trap.  Our periodic box models an extended lattice.
These are different objects and the difference was flagged once and ignored.

**Decision: two tracks, with claims explicitly labelled.**
* **Track A -- periodic extended lattice.**  Clean statistics, well-defined
  L, spectra, finite-size scaling.  This is where H2, H4', H5' are tested.
  Claims from Track A are about supersolid turbulence AS A PHASE OF MATTER,
  not about any current experiment.
* **Track B -- trapped few-droplet, matching Casotti (4 droplets, omega =
  2pi x (50, 103-135) Hz, N = 3-7e4, a_s = 90-95 a0) and Alaña (7 droplets,
  2pi x (60,120) Hz, N = 1.1e5, a_s = 92 a0).**  This is where H1' and H3'
  make experimentally checkable predictions, and where V4/V5 validation
  happens.
A tangle with ~200 vortices is meaningless in a four-droplet system.  Track
B will therefore study the FEW-vortex-per-site regime and the onset of
frustration, not fully developed turbulence.  Saying so plainly is better
than pretending one geometry does both.

## 5. Consequences for the article

* The title's "pinning avalanches" is now a QUESTION, not a premise.
  Candidate revision: "Quantum turbulence in a dipolar supersolid: does
  vortex frustration restore pinning?"  Decide after Track A results.
* The novelty claim needs re-checking against the REVISED question.  The
  original check asked whether anyone had studied a supersolid tangle.  The
  revised question -- does frustration at ell < a_L restore barrier
  dynamics, resolving C8 -- has not been checked at all.
* H1' is partly a reproduction target.  The paper's novelty now rests
  primarily on H4'/C8 and on H3's friction closure, not on H1.

## 6. Immediate next actions, in order

1. Novelty check on the revised question (C8 / frustration crossover).
2. Obtain Poli et al. PRL 131, 223401 full text -- it is the other half of
   C8 and we only have its abstract.
3. Obtain a measured droplet spacing (Norcia Nature 596, 357 (2021), or the
   Casotti Zenodo dataset) to close V4.
4. Only then: implement Track B trapped geometry and run V4/V5.
5. Track A campaign scanning R_ell through unity -- the H4' test.
