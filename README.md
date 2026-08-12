# QT — Quantum Turbulence in a Dipolar Supersolid

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vk001king/qt/blob/main/notebooks/qt_colab.ipynb)

A validated pseudo-spectral solver for the **extended Gross–Pitaevskii equation** (eGPE) with dipolar interactions and Lee–Huang–Yang corrections, built to study vortex-tangle quantum turbulence in dipolar supersolids.

**Status:** 41/41 validation checks passing. All 6 notebook cells verified by execution at **full length** (no shortened loops).

**v1.1** fixes a temporal-aliasing blow-up in v1.0. See *Bug history* below.

---

## Run it in one click

Click the **Open in Colab** badge above, then choose **Runtime → Run all**.

Nothing needs editing. The notebook clones this repo, imports the package, mounts your Google Drive, runs the validation suite, and executes three physics demos — saving every output to a timestamped folder.

Total run time: roughly 6–10 minutes.

---

## Where your results go

```
MyDrive/
└── Research/
    └── Quantum_Turbulence/
        ├── _ARCHIVE_INDEX.json                    master log of every run
        ├── 2026-08-03_1554_validation-suite/
        │   ├── RUN_INFO.json
        │   └── logs/validation-output.txt
        ├── 2026-08-03_1555_vortex-pair-dynamics/
        │   ├── RUN_INFO.json
        │   ├── diagnostics/diagnostics.json
        │   └── figures/pair-initial.png
        │                energy-channels.png
        └── 2026-08-03_1555_dipolar-groundstate/
            ├── RUN_INFO.json
            ├── checkpoints/ckpt_step00000800_t000000.00_155529.npz
            ├── diagnostics/diagnostics.json
            └── figures/groundstate-density.png
```

Folders are checked before creation, never overwritten. Each run is stamped with date, time, and title. `RUN_INFO.json` records the parameters, start and end times, duration, status, and every file written.

---

## Repository layout

```
qt/
├── qtsim/                    the solver package
│   ├── __init__.py           public API + selfcheck()
│   ├── kernels.py            Grid; bare and Ronen-truncated dipolar symbols
│   ├── lhy.py                Lee–Huang–Yang Q5 integral, gamma_tilde
│   ├── solver.py             EGPESolver: split-step real/imaginary time
│   ├── diagnostics.py        vortex detection, circulation, Helmholtz split
│   └── drive_io.py           Archive: structured timestamped output
├── tests/
│   └── run_validation.py     18-check validation suite
├── notebooks/
│   └── qt_colab.ipynb        Colab notebook (all cells execution-tested)
├── campaign/
│   └── template_scan.py      parameter-scan runner with checkpointing
├── test_notebook_cells.py    harness that executes every notebook cell
├── VALIDATION_RESULTS.json   machine-readable validation record
├── ROADMAP.md                remaining steps to publication
├── FLAGS.md                  open reference and code items
└── LICENSE                   MIT
```

---

## Run it locally instead

```bash
git clone https://github.com/vk001king/qt.git
cd qt
pip install numpy matplotlib
python tests/run_validation.py          # expect: 41/41 checks passed
python test_notebook_cells.py           # expect: 6/6 cells executed
```

A campaign run:

```bash
python campaign/template_scan.py \
    --eps_dd 1.4 --Ma 0.5 --seed 0 \
    --N 256 --L 128 \
    --T_relax 100 --T_stir 200 --T_decay 500 \
    --checkpoint_every 500 --use_archive
```

Drop `--use_archive` to write into a plain `./results` directory.

---

## Using the package directly

```python
import sys; sys.path.insert(0, '/content/qt')
from qtsim import Grid, EGPESolver, Archive, bare_dipolar_symbol, gamma_tilde

archive = Archive()                                  # builds the folder tree
run = archive.new_run('my experiment', params={'eps_dd': 1.4})

grid = Grid((256, 256), (64.0, 64.0))                # lengths in healing lengths
Dk = bare_dipolar_symbol(grid, (0, 1))               # 2D: polarize IN-PLANE
gam = gamma_tilde(1.4, 5e-5)

solver = EGPESolver(grid, eps_dd=1.4, gamma=gam, Dk=Dk)
solver.psi = ...                                     # your initial condition
solver.step_imag(0.005, 4000)                        # relax to ground state

print(solver.suggested_dt())                         # largest safe timestep
solver.step_real(0.01, 10000)                        # evolve (guarded)

run.save_checkpoint(solver.psi, step=5000, t=100.0)
run.finish('completed')
```

---

## Dimensionless units

| Quantity | Unit | Definition |
|---|---|---|
| length | ξ | ħ / √(2 m g n₀) |
| time | τ | ħ / (g n₀) |
| speed | c₀ | √(g n₀ / m) |
| field | √n₀ | reference density |
| energy | g n₀ | per particle |

Governing equation solved:

```
i ∂ψ/∂t = [ −∇² + V + |ψ|² + ε_dd·D[|ψ|²] + γ·|ψ|³ ] ψ
```

with `D` the dipolar convolution (Fourier symbol `3cos²θ_k − 1`), `ε_dd = a_dd/a_s`, and `γ` the dimensionless Lee–Huang–Yang strength.

---

## Validation

Run `python tests/run_validation.py`. Forty-one checks, all passing:

| Check | Measured | Criterion |
|---|---|---|
| T1 plane-wave phase evolution | 2.08e-13 | < 1e-10 |
| T2 Strang temporal order | ratio 5.000 | 5.0 ± 0.5 |
| T3 stationarity residual | 3.69e-06 | < 1e-5 |
| T3 density vs Thomas–Fermi | 1.39e-03 | < 3e-2 |
| T3 chemical potential | 2.66e-03 | < 5e-2 |
| T4 Q₅(0) = 1 | 2.2e-16 | < 1e-13 |
| T4 Q₅(1) analytic | 7.2e-15 | < 1e-6 |
| T4 Q₅(1.5) real-part convention | 4.6238 | finite, increasing |
| T5a kernel series branch | 4.1e-09 | < 1e-6 |
| T5b kernel bounds | [−1.000, 2.000] | exact |
| T5c FFT vs direct sum | 2.9e-14 | < 1e-10 |
| T5d Parseval | 1.6e-16 | < 1e-12 |
| T5e orientation signs | −10.30 / +43.00 | prolate negative |
| T6 vortex pair detection | (1, 1) | exactly (1, 1) |
| T6 circulation winding | 1.0000 | 1.00 ± 0.02 |
| T6 Helmholtz split | Ei 88.1 > Ec 14.4 | incompressible dominant |
| T6 norm conservation | 5.8e-14 | < 1e-12 |
| T6 energy conservation | 2.5e-09 | < 1e-4 |
| T7a guard rejects unsafe dt | 6.32 rad/step, rejected | phase > 2 and rejected |
| T7b guarded dt phase advance | 0.790 rad/step | < 1.0 |
| T7c energy drift to t=60 | 1.26e-08 | < 1e-5 |
| T7d norm drift to t=60 | 5.8e-13 | < 1e-10 |
| T7e vortex count to t=60 | 2 | exactly 2 |

## Bug history

**v1.0 -> v1.1: temporal aliasing.** The kinetic substep applies
`exp(-i k^2 dt)` exactly, so there is no CFL stability limit. But the
highest grid mode advances `k_max^2 * dt` radians per step, and once that
approaches pi those modes are unresolved in time; the nonlinear term then
pumps them until the field detonates. The v1.0 demo used a 256^2 grid with
`dt = 0.02`, giving **6.3 rad/step**, and diverged near `t = 35`: two
vortices became 21,000 and the energy grew by a factor of 6000 — all
**with the norm conserved to 1e-13**, so a norm check does not catch it.

Two things were wrong, and both are fixed:

1. *The solver allowed it.* `step_real` now computes `k_max^2 * dt` and
   raises `ValueError` above 2 rad/step, warns above 1. `suggested_dt()`
   returns the largest safe value for any grid.
2. *The test suite missed it.* The v1.0 notebook harness shortened the
   demo loop to `t = 12`, never reaching the failure at `t = 35`. The
   harness now runs every loop at full length, and T7 pins the behaviour
   down permanently.

**Also corrected in v1.1:** the ground-state cell called a single stripe
filling the box "supersolid-like" on the basis of density contrast alone.
It now measures the dominant wavelength as well and reports
`BOX-SCALE ARTIFACT` unless at least two periods fit inside the box.

**Also corrected in v1.1: the stirring nucleated nothing.** A Colab run
showed `n_vortex = 0` at every step of a "stirred tangle" campaign while
compressible energy climbed steadily — the drive was pumping pure sound.
Vortex shedding needs two conditions and v1.0 met neither: the obstacle
must pierce the condensate (`V0 >~ mu`, but v1.0 used `V0 = Ma**2 = 0.25`
against `mu = 1.16`, i.e. 22%), and the local flow must exceed the
critical velocity (`Ma >~ 0.5`, but v1.0 gave 0.35). With `V0 = 3 mu` and
`Ma = 1.1` the count now rises 0 → 36 → 94 → 198 and saturates, then
decays to ~124 in free decay while `E_i` falls and `E_c` rises. The drive
is also now evaluated at the substep midpoint, preserving second order.

**Open accuracy note.** The dipolar ground state converges only to a
stationarity residual of ~1e-2, far short of the 1e-8 target in the
methodology. That is consistent with the state being a marginal box-scale
mode rather than a true minimum, and should resolve once the quasi-2D
projected kernel gives a physical modulation wavelength (rung V4).

Higher validation rungs (comparison against published ¹⁶⁴Dy results, glitch dynamics, turbulence regression) are listed in `ROADMAP.md` as the next work item.

---

## v1.8: real experimental parameters, and H1 inverted

Casotti et al., *Nature* **635**, 327 (2024) was read in full including
Methods. Three corrections followed; see `EXPERIMENTAL_PARAMETERS.md`.

**H1 was backwards.** We hypothesised that interstitial pinning *raises*
the vortex nucleation threshold in a supersolid. The paper reports the
opposite in both experiment and its own eGPE: the supersolid nucleates at
`Omega ~ 0.25-0.45 omega_perp` against `~0.6` for the BEC, because a 2D
supersolid's near-degenerate **crystal** quadrupole mode opens an extra
angular-momentum channel. Pinning governs vortex motion and decay, not the
threshold. H1 must be inverted before any campaign tests it.

**eps_dd = 1.8 was outside the supersolid phase.** Real window is
`a_s = 90-95 a0` with `a_dd = 130.8 a0`, i.e. `eps_dd = 1.377-1.453`. Our
1.8 means `a_s = 72.7 a0` -- isolated droplets. At real parameters the roton
survives but `min(inside)` is only `-0.03` to `-0.19` versus `-0.60` at 1.8,
and vanishes at higher density: crystal existence is density-sensitive, which
the wrong value hid entirely. Defaults are now `eps_dd = 1.414`, `l_z/xi = 8.6`.

**Confirmed:** our `Q5` and LHY prefactor are algebraically identical to
theirs, and the `Re{}` convention for `eps_dd > 1` is now sourced -- that
flag is closed. Our v1.7 vortex-count systematic is independently
corroborated: they mask to a 6 um circle and state that varying their
detection threshold changes absolute counts but not qualitative results.

## v1.8: real experimental parameters, and H1 inverted

Casotti et al., *Nature* **635**, 327 (2024) was read in full including
Methods. Three corrections followed; see `EXPERIMENTAL_PARAMETERS.md`.

**H1 was backwards.** We hypothesised that interstitial pinning *raises*
the vortex nucleation threshold in a supersolid. The paper reports the
opposite in both experiment and its own eGPE: the supersolid nucleates at
`Omega ~ 0.25-0.45 omega_perp` against `~0.6` for the BEC, because a 2D
supersolid's near-degenerate **crystal** quadrupole mode opens an extra
angular-momentum channel. Pinning governs vortex motion and decay, not the
threshold. H1 must be inverted before any campaign tests it.

**eps_dd = 1.8 was outside the supersolid phase.** Real window is
`a_s = 90-95 a0` with `a_dd = 130.8 a0`, i.e. `eps_dd = 1.377-1.453`. Our
1.8 means `a_s = 72.7 a0` -- isolated droplets. At real parameters the roton
survives but `min(inside)` is only `-0.03` to `-0.19` versus `-0.60` at 1.8,
and vanishes at higher density: crystal existence is density-sensitive, which
the wrong value hid entirely. Defaults are now `eps_dd = 1.414`, `l_z/xi = 8.6`.

**Confirmed:** our `Q5` and LHY prefactor are algebraically identical to
theirs, and the `Re{}` convention for `eps_dd > 1` is now sourced -- that
flag is closed. Our v1.7 vortex-count systematic is independently
corroborated: they mask to a 6 um circle and state that varying their
detection threshold changes absolute counts but not qualitative results.

## v1.7: vortex counting in a droplet crystal

The inter-droplet regions of a crystal are near-vacuum (`n_min ~ 1e-6
n_mean`), the phase there is numerical noise, and the raw plaquette
detector finds random windings in it. A plain density mask is the wrong fix
because a real vortex core also has `n -> 0` at its centre, so
`plaquette_charges_2d_masked` applies an **annulus** density criterion
instead. Validated on a fluid/void control: 97.4% of void false positives
rejected, the real vortex kept, and a no-op on a clean pair.

**This is a systematic you must quote.** On a stirred quasi-2D crystal:

| method | count |
|---|---|
| raw plaquette | 874 |
| annulus mask (default) | 724 (−17%) |
| crude corner mask | 459 (−47%) |

**v2.3 correction — the systematic is worse than this table suggests.** In a
stirred run the rejection fraction was measured at 90%, 75%, 54%, 35% at
successive times, and the raw and masked counts grow at different rates
(2.8× versus 18×). So raw `L(t)` has the wrong *shape*, not just an offset.
The masked net charge is now returned as a quality gate: true circulation is
exactly zero, so residual imbalance measures mis-clipping, and it reaches
100% when only one vortex survives masking. This makes the low-vortex
(`R > 1`) side of the H4′ crossover the hardest region to measure, and is
why the Track A default box is now 12 cells rather than 5.

Vortex line density in a droplet crystal therefore carries a
**method-dependent systematic of order 20–50%**. `L` is the primary
observable for the threshold, friction, avalanche and decay analyses, so
every reported `L` must carry it. The campaign now records raw and masked
counts plus the reject fraction, so the systematic is measured per run.
The residual 2.6% of false positives sit on the fluid/void interface, where
the classification is genuinely ambiguous; no parameter choice removes them.

## v1.6: direct energy minimisation

Imaginary-time gradient flow is only steepest descent, and on the droplet
crystal it plateaued. `qtsim/minimize.py` adds L-BFGS-B minimisation of the
energy functional (norm handled by projection) and a multi-start ensemble
keeping the lowest chemical potential:

| method | mu | residual |
|---|---|---|
| gradient flow, 24000 steps | 5.509289 | 2.28e-02 |
| single-start L-BFGS-B | 5.466331 | 4.74e-03 |
| multi-start, lattice seed | **5.189268** | 4.73e-03 |

On a simple trapped problem the same minimiser takes the residual from
5.4e-02 to 7.5e-07 in 93 iterations, so the optimiser is not the limit --
the crystal's landscape is genuinely rugged. Different seeds converge
tightly into *different* basins (one reached residual 5.2e-05 at a higher
mu), so a low residual does not imply the global minimum.

Also fixed: `2*pi/k_dominant` is the density **modulation wavelength**, not
the triangular lattice constant. The first Bragg vector satisfies
`|k| = 4*pi/(a*sqrt3)`, so `a = 2*lambda/sqrt(3)` -- a factor 1.1547.
Published 164Dy numbers quote the lattice constant, so conflating them
would have corrupted the V4 comparison. Both are now reported.

## v1.4: Drive persistence is now enforced

Earlier versions fell back to Colab's ephemeral disk without complaint if
the Drive mount failed, so a whole session's output could be lost silently.
Mount failure inside Colab now raises, the notebook asserts persistence
before running anything, and the archive prints its real path rather than
an assumed one.

## v1.3: quasi-2D kernel and the droplet crystal

The bare periodic dipolar symbol has no roton, which is why earlier
versions produced only a box-scale stripe. v1.3 adds the **quasi-2D
projected kernel**, derived by integrating the 3D symbol against the
Gaussian axial density:

```
D(k) = 2*sqrt(2) - 3*sqrt(2*pi) * u * erfcx(u),    u = k * l_z / 2
```

It runs from `+2*sqrt(2)` (repulsive) at `k=0` to `-sqrt(2)` (attractive)
at large `k`. That sign change is the roton. Verified against direct
numerical quadrature to **5e-15**, with both limits exact.

With `eps_dd = 1.8`, `l_z = 6 xi` the uniform state is roton-unstable and
relaxation produces a genuine **triangular droplet crystal**, contrast
~19.7, whose period agrees with the linear roton prediction to **1.8%**
(6.995 vs 7.124 xi) in a commensurate box.

**Open problem, diagnosed in v1.5.** The crystal is not a converged
stationary state, and measurement now says exactly why. Over `t = 6`:
energy is conserved to `1e-10` and norm to `1e-13`, so the solver is
sound; `n_peak` holds at 19.6 and contrast drifts only `3.6e-3`, so the
droplets keep their shape and there is **no collapse** (LHY is adequate);
the net crystal slide is `0.058 xi`, under 1% of the lattice constant, so
it is **not** a rigid Goldstone translation. What does change is the
Bragg *amplitude* spectrum, `||dA||/||A|| -> 0.46`: individual droplets
rearrange their positions. Seeding a perfect triangular lattice reaches
`mu = 5.233` against `5.509` from a noise seed, proving the noise-seeded
state was defected and neither is the minimum.

So this is an **optimisation problem, not a physics or solver bug**: the
landscape has many nearby minima and plain imaginary-time gradient flow
from one seed finds the wrong one. The fix is a real minimiser
(conjugate-gradient / L-BFGS on the energy functional) or a multi-seed
ensemble keeping the lowest `mu` — not more gradient-flow steps. Until
that lands, the lattice constant cannot be compared to published 164Dy
measurements, so **validation rung V4 remains started, not finished.** Three-dimensional vortex line tracking and the GPU backend are also outstanding. See `FLAGS.md`.

---

## License

MIT.
