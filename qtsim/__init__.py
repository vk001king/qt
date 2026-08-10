"""qtsim -- eGPE solver for quantum turbulence in dipolar supersolids.

Package layout:
    kernels.py      Grid, dipolar Fourier symbols (bare + Ronen-truncated)
    lhy.py          Lee-Huang-Yang Q5 integral, dimensionless gamma
    solver.py       EGPESolver: split-step real/imaginary time, energy
    diagnostics.py  vortex detection, circulation, Helmholtz decomposition
    drive_io.py     Archive: structured timestamped output folders

Quick start:
    from qtsim import Grid, EGPESolver, Archive
    archive = Archive()                     # builds Research/Quantum_Turbulence
    run = archive.new_run("my first run")
"""

from .kernels import (
    Grid,
    bare_dipolar_symbol,
    truncated_dipolar_symbol,
    truncation_bracket,
    quasi2d_dipolar_symbol,
    quasi2d_dipolar_profile,
    bogoliubov_omega,
    roton_wavevector,
)
from .lhy import Q5, gamma_tilde
from .solver import EGPESolver
from .diagnostics import (
    plaquette_charges_2d,
    circulation_loop_2d,
    helmholtz_split_2d,
)
from .drive_io import (Archive, Run, ensure_dir, mount_drive,
                       rescue_ephemeral)
from . import diagnostics

__version__ = "1.4.0"

__all__ = [
    "Grid",
    "bare_dipolar_symbol",
    "truncated_dipolar_symbol",
    "truncation_bracket",
    "quasi2d_dipolar_symbol",
    "quasi2d_dipolar_profile",
    "bogoliubov_omega",
    "roton_wavevector",
    "Q5",
    "gamma_tilde",
    "EGPESolver",
    "plaquette_charges_2d",
    "circulation_loop_2d",
    "helmholtz_split_2d",
    "Archive",
    "Run",
    "ensure_dir",
    "mount_drive",
    "rescue_ephemeral",
    "diagnostics",
    "selfcheck",
]


def selfcheck(verbose: bool = True) -> bool:
    """Verify every public name imported correctly.

    Returns True when the package is fully functional.  Called by the
    Colab notebook immediately after import so that a broken install is
    reported clearly instead of failing later with a cryptic error.
    """
    missing = [name for name in __all__ if name not in globals()]
    if verbose:
        if missing:
            print(f"  [FAIL] missing from qtsim: {missing}")
        else:
            print(f"  [ OK ] qtsim v{__version__}: "
                  f"{len(__all__) - 1} public names available")
    return not missing
