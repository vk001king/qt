# QT — Quantum Turbulence in a Dipolar Supersolid

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vk001king/qt/blob/main/notebooks/qt_colab.ipynb)

A validated pseudo-spectral solver for the **extended Gross–Pitaevskii equation** (eGPE) with dipolar interactions and Lee–Huang–Yang corrections, built to study vortex-tangle quantum turbulence in dipolar supersolids.

**Status:** 18/18 validation checks passing. All 6 notebook cells verified by execution.

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
python tests/run_validation.py          # expect: 18/18 checks passed
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
solver.step_real(0.02, 5000)                         # evolve

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

Run `python tests/run_validation.py`. Eighteen checks, all passing:

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

Higher validation rungs (comparison against published ¹⁶⁴Dy results, glitch dynamics, turbulence regression) are listed in `ROADMAP.md` as the next work item.

---

## Known limitations

The 2D dipolar kernel here is the bare periodic symbol, adequate for demonstration but not the production quasi-2D projected kernel; the demo ground state is therefore uniform rather than crystalline at the parameters shown. Reproducing the published supersolid lattice constant is validation rung V4. Three-dimensional vortex line tracking and the GPU backend are also outstanding. See `FLAGS.md`.

---

## License

MIT.
