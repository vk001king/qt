# QT — Quantum Turbulence in a Dipolar Supersolid

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vk001king/qt/blob/main/notebooks/qt_colab.ipynb)

A validated pseudo-spectral solver for the **extended Gross–Pitaevskii equation** (eGPE) with dipolar interactions and Lee–Huang–Yang corrections, built to study vortex-tangle quantum turbulence in dipolar supersolids.

**Status:** 23/23 validation checks passing. All 6 notebook cells verified by execution at **full length** (no shortened loops).

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
python tests/run_validation.py          # expect: 23/23 checks passed
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

Run `python tests/run_validation.py`. Twenty-three checks, all passing:

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

## Known limitations

The 2D dipolar kernel here is the bare periodic symbol, adequate for demonstration but not the production quasi-2D projected kernel; the demo ground state is therefore uniform rather than crystalline at the parameters shown. Reproducing the published supersolid lattice constant is validation rung V4. Three-dimensional vortex line tracking and the GPU backend are also outstanding. See `FLAGS.md`.

---

## License

MIT.
