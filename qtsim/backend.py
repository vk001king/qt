"""Array-backend selection: numpy (CPU) or cupy (GPU), chosen explicitly.

WHY THIS EXISTS.  Every module in this package used a hardcoded `import
numpy as np`.  Turning on a Colab GPU runtime does nothing by itself --
NumPy never touches the GPU, so the T4 sits idle while the CPU does all the
work.  This module lets `Grid` and `EGPESolver` opt into CuPy, which mirrors
NumPy's array API closely enough that most of the package needed no other
changes.

DESIGN CHOICE: explicit, not automatic.  Silent auto-detection would repeat
the exact mistake documented in `drive_io.py` (silent fallback to
ephemeral storage caused real data loss).  `backend="gpu"` must be
requested; if CuPy is unavailable it raises with a clear message rather
than quietly running on CPU, so a person cannot mistake a CPU run for a GPU
one.  `backend="cpu"` (the default) never imports CuPy at all, so nothing
here changes behaviour for existing CPU code or the sandbox this package
was developed and tested in.

WHAT IS NOT PORTED.  `scipy.optimize.minimize` (used by `minimize.py` for
the L-BFGS-B ground-state search) is CPU-only.  The gradient's expensive
part -- FFTs and elementwise nonlinear terms -- still runs on the selected
backend; only the packed vector handed to scipy is transferred to the host
once per iteration.  This is now the throughput floor for the minimiser on
GPU and is stated as such rather than hidden.
"""
from __future__ import annotations

import numpy as _np

_VALID = ("cpu", "gpu")


def get_backend(backend: str = "cpu"):
    """Return (xp, fft_module, name) for the requested backend.

    Parameters
    ----------
    backend : "cpu" (NumPy, default) or "gpu" (CuPy).

    Raises
    ------
    ValueError if `backend` is not one of "cpu"/"gpu".
    RuntimeError if "gpu" is requested but CuPy or a GPU is unavailable --
        deliberately loud, so a GPU run can never silently become a CPU run.
    """
    if backend not in _VALID:
        raise ValueError(f"backend must be one of {_VALID}, got {backend!r}")
    if backend == "cpu":
        return _np, _np.fft, "cpu"

    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "backend='gpu' requested but CuPy is not installed. Install "
            "the CUDA build matching Colab's runtime, e.g.\n"
            "    !pip install -q cupy-cuda12x\n"
            "then restart and re-import qtsim. (Never falls back to CPU "
            "silently -- see qtsim/backend.py.)") from exc
    try:
        n_gpus = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        raise RuntimeError(
            "backend='gpu' requested and CuPy imported, but no CUDA device "
            "is visible. In Colab: Runtime -> Change runtime type -> "
            "Hardware accelerator -> GPU, then restart the runtime.") from exc
    if n_gpus == 0:
        raise RuntimeError("backend='gpu' requested but 0 CUDA devices "
                          "were found.")
    return cp, cp.fft, "gpu"


def asnumpy(arr):
    """Move an array to the host as a NumPy array, whatever backend it is on.

    A NumPy array passes through unchanged (no copy assumptions made by
    callers). A CuPy array is copied to host via its own `.get()`.
    """
    if isinstance(arr, _np.ndarray):
        return arr
    get = getattr(arr, "get", None)
    if get is not None:
        return get()
    return _np.asarray(arr)


def device_report(xp, name: str) -> str:
    """One-line human-readable description of the active backend."""
    if name == "cpu":
        return "backend=cpu (NumPy)"
    try:
        dev = xp.cuda.Device()
        props = xp.cuda.runtime.getDeviceProperties(dev.id)
        gpu_name = props["name"].decode() if isinstance(props["name"], bytes) \
            else props["name"]
        free, total = xp.cuda.runtime.memGetInfo()
        return (f"backend=gpu (CuPy) device={dev.id} {gpu_name} "
                f"mem_free={free / 1e9:.1f}GB/{total / 1e9:.1f}GB")
    except Exception:
        return "backend=gpu (CuPy), device details unavailable"
