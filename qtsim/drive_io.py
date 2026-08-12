"""Structured output archive for Google Drive (or any filesystem).

Directory layout created and maintained automatically:

    Research/
      Quantum_Turbulence/
        _ARCHIVE_INDEX.json           <- master index of every run ever made
        2026-07-23_1432_validation/
            RUN_INFO.json             <- title, timestamp, parameters, status
            validation_results.json
        2026-07-23_1455_supersolid-groundstate-eps1.40/
            RUN_INFO.json
            checkpoints/
                ckpt_step00000500.npz
                ckpt_step00001000.npz
            diagnostics/
                diagnostics.json
            figures/
                density.png
        2026-07-24_0903_stirred-tangle-eps1.30-Ma0.50-s0/
            ...

Design rules (per user specification):
  * Every folder is checked for existence before use; created only if absent.
  * Each run gets its OWN folder named  <date>_<time>_<title-slug>.
  * Every output is written inside that run folder, never at top level.
  * A machine-readable index is updated after every run so nothing is lost.
"""
from __future__ import annotations

import json
import os
import platform
import re
import time
from datetime import datetime

# ---------------------------------------------------------------- constants
DEFAULT_BASE = "Research"
DEFAULT_PROJECT = "Quantum_Turbulence"
INDEX_NAME = "_ARCHIVE_INDEX.json"
RUN_INFO_NAME = "RUN_INFO.json"
SUBDIRS = ("checkpoints", "diagnostics", "figures", "logs")


# ---------------------------------------------------------------- utilities
def _slug(text: str, maxlen: int = 60) -> str:
    """Filesystem-safe slug: lowercase, dashes, no exotic characters."""
    s = re.sub(r"[^\w\s.-]", "", str(text)).strip().lower()
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-.")
    return (s[:maxlen].rstrip("-.") or "run")


def ensure_dir(path: str, verbose: bool = True) -> str:
    """Check whether a directory exists; create it only if it does not.

    Returns the path.  Prints which of the two happened (the user asked
    explicitly for the check-then-proceed behaviour to be visible).
    """
    if os.path.isdir(path):
        if verbose:
            print(f"  [exists ] {path}")
    else:
        os.makedirs(path, exist_ok=True)
        if verbose:
            print(f"  [created] {path}")
    return path


def mount_drive(verbose: bool = True, require: bool = True) -> str:
    """Mount Google Drive when running in Colab.

    Parameters
    ----------
    require : if True (default) and we ARE in Colab but the mount fails,
        raise instead of silently writing to ephemeral disk.  A silent
        fallback here caused a real data loss: a 200 s run was written to
        /content/drive_local and would have been destroyed when the
        session ended, with nothing in the output to indicate it.

    Outside Colab (local machine, CI) a local ./drive_local root is used,
    which is correct and is stated plainly.
    """
    in_colab = False
    try:
        import google.colab  # noqa: F401
        in_colab = True
    except ImportError:
        pass

    if in_colab:
        if os.path.isdir("/content/drive/MyDrive"):
            if verbose:
                print("  [exists ] Google Drive already mounted")
            return "/content/drive/MyDrive"
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception as exc:                       # mount refused/failed
            msg = ("Google Drive did NOT mount (%s).\n"
                   "    Results would go to ephemeral Colab disk and be LOST\n"
                   "    when the session ends.  Approve the Drive popup and\n"
                   "    re-run this cell.  To proceed anyway (data will not\n"
                   "    persist) use Archive(..., require_drive=False)."
                   % type(exc).__name__)
            if require:
                raise RuntimeError(msg) from exc
            print("  !! WARNING: " + msg)
            root = os.path.abspath("./drive_local")
            ensure_dir(root, verbose=verbose)
            return root
        if not os.path.isdir("/content/drive/MyDrive"):
            msg = ("Drive mount reported success but /content/drive/MyDrive\n"
                   "    is absent.  Refusing to write to ephemeral disk.")
            if require:
                raise RuntimeError(msg)
            print("  !! WARNING: " + msg)
            root = os.path.abspath("./drive_local")
            ensure_dir(root, verbose=verbose)
            return root
        if verbose:
            print("  [mounted] Google Drive")
        return "/content/drive/MyDrive"

    root = os.path.abspath("./drive_local")
    if verbose:
        print("  Not running in Colab -> local root: %s" % root)
    ensure_dir(root, verbose=verbose)
    return root


# ---------------------------------------------------------------- archive
class Archive:
    """Structured, timestamped run archive rooted in Google Drive.

    Example
    -------
    >>> arc = Archive()                       # mounts Drive, builds tree
    >>> run = arc.new_run("stirred tangle", params={"eps_dd": 1.3})
    >>> run.save_checkpoint(psi, step=500, t=10.0)
    >>> run.save_diagnostics(records)
    >>> run.finish(status="completed")
    """

    def __init__(self, base: str = DEFAULT_BASE,
                 project: str = DEFAULT_PROJECT,
                 root: str | None = None, verbose: bool = True,
                 require_drive: bool = True):
        print("Preparing archive tree...")
        self.root = (root if root is not None
                     else mount_drive(verbose, require=require_drive))
        self.persistent = self.root.startswith("/content/drive/MyDrive")
        self.base_dir = ensure_dir(os.path.join(self.root, base), verbose)
        self.project_dir = ensure_dir(os.path.join(self.base_dir, project),
                                      verbose)
        self.index_path = os.path.join(self.project_dir, INDEX_NAME)
        self._ensure_index(verbose)
        print(f"Archive ready: {self.project_dir}")

        # Sandbox/CI override.  MUST be explicitly requested by an
        # environment variable, never inferred -- same "explicit, not
        # automatic" rule as backend selection (see backend.py) and the
        # same rule that the earlier silent-fallback-to-ephemeral-disk bug
        # taught the hard way (see FLAGS.md, v1.4).  This flag exists
        # solely so automated tests (test_notebook_cells.py) that run
        # outside Colab -- where no Google Drive can ever be mounted --
        # can exercise the REST of the notebook logic without the
        # persistence assertion halting them at cell 1.  It does NOT
        # change self.root or where files are written; it only overrides
        # the `persistent` flag that callers check.  A real user must
        # never set this.
        if os.environ.get("QTSIM_SANDBOX_TEST") == "1":
            print("  !! QTSIM_SANDBOX_TEST=1: persistence check bypassed "
                  "for automated testing only -- files still land at the "
                  "path above, NOT on Google Drive. Never set this "
                  "variable in a real Colab session.")
            self.persistent = True
        elif root is None and not getattr(self, "persistent", False):
            try:
                import google.colab  # noqa: F401
                print("  !! NOT on Google Drive -- outputs are EPHEMERAL "
                      "and will be lost when this session ends.")
            except ImportError:
                pass
        print()

    # -- index -----------------------------------------------------------
    def _ensure_index(self, verbose: bool = True) -> None:
        if os.path.isfile(self.index_path):
            if verbose:
                n = len(self.load_index().get("runs", []))
                print(f"  [exists ] {INDEX_NAME}  ({n} previous run(s))")
        else:
            payload = {
                "project": "Quantum turbulence in a dipolar supersolid",
                "created": datetime.now().isoformat(timespec="seconds"),
                "runs": [],
            }
            with open(self.index_path, "w") as fh:
                json.dump(payload, fh, indent=1)
            if verbose:
                print(f"  [created] {INDEX_NAME}")

    def load_index(self) -> dict:
        try:
            with open(self.index_path) as fh:
                return json.load(fh)
        except Exception:
            return {"runs": []}

    def _append_index(self, entry: dict) -> None:
        idx = self.load_index()
        idx.setdefault("runs", []).append(entry)
        idx["last_updated"] = datetime.now().isoformat(timespec="seconds")
        with open(self.index_path, "w") as fh:
            json.dump(idx, fh, indent=1)

    def _update_index(self, run_id: str, **fields) -> None:
        idx = self.load_index()
        for r in idx.get("runs", []):
            if r.get("run_id") == run_id:
                r.update(fields)
                break
        idx["last_updated"] = datetime.now().isoformat(timespec="seconds")
        with open(self.index_path, "w") as fh:
            json.dump(idx, fh, indent=1)

    # -- runs ------------------------------------------------------------
    def new_run(self, title: str, params: dict | None = None,
                verbose: bool = True) -> "Run":
        """Create a fresh timestamped run folder titled <date>_<time>_<title>."""
        now = datetime.now()
        stamp = now.strftime("%Y-%m-%d_%H%M")
        run_id = f"{stamp}_{_slug(title)}"
        run_dir = os.path.join(self.project_dir, run_id)

        # collision guard (two runs launched in the same minute)
        if os.path.isdir(run_dir):
            run_id = f"{now.strftime('%Y-%m-%d_%H%M%S')}_{_slug(title)}"
            run_dir = os.path.join(self.project_dir, run_id)

        print(f"Creating run folder for '{title}':")
        ensure_dir(run_dir, verbose)
        for sub in SUBDIRS:
            ensure_dir(os.path.join(run_dir, sub), verbose)

        info = {
            "run_id": run_id,
            "title": title,
            "started": now.isoformat(timespec="seconds"),
            "started_human": now.strftime("%d %B %Y, %H:%M:%S"),
            "status": "running",
            "parameters": params or {},
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "files": {},
        }
        with open(os.path.join(run_dir, RUN_INFO_NAME), "w") as fh:
            json.dump(info, fh, indent=1)

        self._append_index({
            "run_id": run_id, "title": title,
            "started": info["started"], "status": "running",
            "parameters": params or {},
        })
        print(f"Run folder ready: {run_dir}\n")
        return Run(self, run_dir, info)

    def list_runs(self) -> list:
        """Print and return every run recorded in the index."""
        runs = self.load_index().get("runs", [])
        if not runs:
            print("No runs recorded yet.")
            return runs
        print(f"{len(runs)} run(s) in {self.project_dir}:\n")
        for r in runs:
            print(f"  {r.get('started','?')}  [{r.get('status','?'):9s}]  "
                  f"{r.get('title','?')}")
            print(f"      folder: {r.get('run_id','?')}")
        return runs

    def latest_run_dir(self, title_contains: str | None = None) -> str | None:
        """Path of the most recent run, optionally filtered by title."""
        runs = self.load_index().get("runs", [])
        if title_contains:
            runs = [r for r in runs
                    if title_contains.lower() in r.get("title", "").lower()]
        if not runs:
            return None
        return os.path.join(self.project_dir, runs[-1]["run_id"])


class Run:
    """A single timestamped run folder; all outputs go through this object."""

    def __init__(self, archive: Archive, run_dir: str, info: dict):
        self.archive = archive
        self.dir = run_dir
        self.info = info
        self.run_id = info["run_id"]
        self._t0 = time.time()

    # -- internal --------------------------------------------------------
    def _register(self, category: str, path: str) -> None:
        self.info.setdefault("files", {}).setdefault(category, []).append(
            os.path.basename(path))
        self._write_info()

    def _write_info(self) -> None:
        with open(os.path.join(self.dir, RUN_INFO_NAME), "w") as fh:
            json.dump(self.info, fh, indent=1)

    def path(self, *parts) -> str:
        return os.path.join(self.dir, *parts)

    # -- saving ----------------------------------------------------------
    def save_checkpoint(self, psi, step: int, t: float, extra: dict | None = None,
                        verbose: bool = True) -> str:
        """Save the wavefunction with step number and simulation time.

        BUG FIX (v2.5): `np.savez_compressed` requires a literal host
        NumPy array; passing a CuPy array (backend="gpu") directly failed.
        This bug was NOT caught by the fake-cupy verification in v2.4,
        because that fake module's arrays WERE real numpy arrays under
        the hood, so `isinstance(x, np.ndarray)`-style checks and
        numpy-only functions like `savez_compressed` never saw a foreign
        type there -- a real limitation of that verification method,
        stated plainly rather than left implied.  Converts `psi` to host
        explicitly before saving, regardless of backend.
        """
        import numpy as np
        from .backend import asnumpy
        stamp = datetime.now().strftime("%H%M%S")
        fname = self.path("checkpoints",
                          f"ckpt_step{int(step):08d}_t{t:09.2f}_{stamp}.npz")
        payload = {"psi": asnumpy(psi), "step": int(step), "t": float(t),
                   "wall_clock": datetime.now().isoformat(timespec="seconds")}
        if extra:
            payload.update({k: v for k, v in extra.items()
                            if isinstance(v, (int, float, str))})
        np.savez_compressed(fname, **payload)
        self._register("checkpoints", fname)
        if verbose:
            print(f"    saved checkpoint -> {os.path.basename(fname)}")
        return fname

    def save_diagnostics(self, records, name: str = "diagnostics",
                         verbose: bool = True) -> str:
        fname = self.path("diagnostics", f"{_slug(name)}.json")
        with open(fname, "w") as fh:
            json.dump(records, fh, indent=1)
        self._register("diagnostics", fname)
        if verbose:
            n = len(records) if hasattr(records, "__len__") else "?"
            print(f"    saved diagnostics -> {os.path.basename(fname)} "
                  f"({n} record(s))")
        return fname

    def save_figure(self, fig, name: str, dpi: int = 150,
                    verbose: bool = True) -> str:
        fname = self.path("figures", f"{_slug(name)}.png")
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        self._register("figures", fname)
        if verbose:
            print(f"    saved figure -> {os.path.basename(fname)}")
        return fname

    def save_text(self, text: str, name: str, verbose: bool = True) -> str:
        fname = self.path("logs", f"{_slug(name)}.txt")
        with open(fname, "w") as fh:
            fh.write(text)
        self._register("logs", fname)
        if verbose:
            print(f"    saved log -> {os.path.basename(fname)}")
        return fname

    # -- loading ---------------------------------------------------------
    def latest_checkpoint(self) -> str | None:
        d = self.path("checkpoints")
        files = sorted(f for f in os.listdir(d) if f.endswith(".npz")) \
            if os.path.isdir(d) else []
        return os.path.join(d, files[-1]) if files else None

    @staticmethod
    def load_checkpoint(fname: str):
        """Reload (psi, step, t) from a checkpoint file."""
        import numpy as np
        data = np.load(fname)
        step, t = int(data["step"]), float(data["t"])
        print(f"  Loaded {os.path.basename(fname)}: step={step}, t={t:.2f}")
        return data["psi"], step, t

    # -- lifecycle -------------------------------------------------------
    def finish(self, status: str = "completed", notes: str = "",
               verbose: bool = True) -> None:
        ended = datetime.now()
        self.info["status"] = status
        self.info["ended"] = ended.isoformat(timespec="seconds")
        self.info["ended_human"] = ended.strftime("%d %B %Y, %H:%M:%S")
        self.info["duration_seconds"] = round(time.time() - self._t0, 1)
        if notes:
            self.info["notes"] = notes
        self._write_info()
        self.archive._update_index(
            self.run_id, status=status, ended=self.info["ended"],
            duration_seconds=self.info["duration_seconds"])
        if verbose:
            print(f"\nRun '{self.info['title']}' {status} in "
                  f"{self.info['duration_seconds']} s")
            print(f"  Folder: {self.dir}")


def rescue_ephemeral(dest_root: str = "/content/drive/MyDrive",
                     src_root: str = "/content/drive_local",
                     base: str = DEFAULT_BASE,
                     project: str = DEFAULT_PROJECT) -> int:
    """Copy runs written to ephemeral disk into Google Drive.

    Use after a session where the Drive mount silently failed (fixed in
    v1.4, but older archives may exist).  Returns the number of run
    folders copied.  Existing destinations are merged, not clobbered.
    """
    import shutil
    src = os.path.join(src_root, base, project)
    dst = os.path.join(dest_root, base, project)
    if not os.path.isdir(src):
        print("Nothing to rescue: %s does not exist" % src)
        return 0
    ensure_dir(dst, verbose=False)
    n = 0
    for name in sorted(os.listdir(src)):
        s_path, d_path = os.path.join(src, name), os.path.join(dst, name)
        if os.path.isdir(s_path):
            shutil.copytree(s_path, d_path, dirs_exist_ok=True)
            n += 1
        elif not os.path.exists(d_path):
            shutil.copy2(s_path, d_path)
    print("Rescued %d run folder(s) into %s" % (n, dst))
    return n
