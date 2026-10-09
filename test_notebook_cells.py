"""Execute every code cell of qt_colab.ipynb in order, as Colab would.

Adaptations for the sandbox (no Colab, no GPU, limited time):
  * DEST is redirected to this repo instead of cloning from GitHub
  * matplotlib runs headless (Agg)
  * QTSIM_SANDBOX_TEST=1 bypasses the Drive-persistence assertion in
    Cell 1 (see drive_io.py) -- there is no Google Drive outside Colab,
    so that assertion would otherwise halt every local/CI run at cell 1.
    This bypass changes NOTHING about where files are written; it only
    lets execution proceed so the rest of the notebook can be exercised.
  * the optional GPU verification cell (Cell 9, last cell) requires real CuPy and a
    real CUDA device, neither of which exist here; it is executed and
    expected to exit cleanly via its own SystemExit, which is reported as
    SKIPPED rather than counted as a pass or a failure.
  * the SHORTEN table below ONLY redirects the git clone; loops run at
    full length (v1.0 shortened them and so missed a t~35 blow-up).
Everything else -- imports, archive, solver calls, figure saving,
diagnostics, run bookkeeping -- runs exactly as written in the notebook.

REGRESSION NOTE.  This harness silently failed at Cell 1 for every version
from v1.4 through v2.3: the persistence assertion added in v1.4 correctly
raises when no Google Drive is mounted, which is ALWAYS the case outside
Colab, and nothing in this file told it otherwise.  During that entire
span, "full notebook" verification was actually done via ad hoc per-cell
exec scripts that pre-injected `archive` as a global, bypassing Cell 1
entirely -- not by running this file.  So the claim "all cells verified
end to end" was true for cells 2-6 individually but the sanctioned,
sequential, cell-1-through-the-end harness had not actually run since
v1.3.  Fixed in v2.4 via the sandbox override above.
"""
import json
import os
import re
import sys
import traceback

# Must be set BEFORE qtsim is imported by any executed cell (Cell 1 does
# the import), and must be the only thing that flips this bypass -- see
# the REGRESSION NOTE above and drive_io.py for why it is env-gated rather
# than automatic.
os.environ["QTSIM_SANDBOX_TEST"] = "1"
print("=" * 68)
print("QTSIM_SANDBOX_TEST=1 -- bypassing the Drive-persistence assertion")
print("for this automated run only. Files still write to the local")
print("fallback path, never to Google Drive. This variable must never")
print("be set in a real Colab session.")
print("=" * 68)
print()

import matplotlib
matplotlib.use("Agg")

REPO = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(REPO, "notebooks", "qt_colab.ipynb")

# (pattern, replacement) applied to every cell so the test finishes quickly
SHORTEN = [
    # redirect the clone: use this repo directly
    (r"REPO = 'https://github\.com/[^']*'", "REPO = 'LOCAL'"),
    (r"DEST = '/content/qt'", "DEST = %r" % REPO),
    (r"if os\.path\.isdir\(DEST \+ '/\.git'\):",
     "if True:  # sandbox: repo already local"),
    (r"subprocess\.run\(\['git', '-C', DEST, 'pull', '-q'\], check=False\)",
     "print('(sandbox: skipping git pull)')"),
    # NO loop shortening.  v1.0 shortened cell 3 to t=12 and therefore
    # never reached the t~35 blow-up that hit the user in Colab.  Every
    # loop now runs at full length, exactly as Colab will run it.
]


def prepare(src):
    for pat, rep in SHORTEN:
        src = re.sub(pat, rep, src)
    return src


def main():
    nb = json.load(open(NB))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    print("Executing %d code cells from qt_colab.ipynb\n" % len(cells))

    env = {"__name__": "__main__"}
    passed, skipped = 0, 0

    for i, cell in enumerate(cells, 1):
        src = prepare("".join(cell["source"]))
        first = src.strip().split("\n")[0][:64]
        is_optional_gpu_cell = (i == len(cells)
                               and "GPU backend" in "".join(cell["source"]))
        print("=" * 68)
        print("CELL %d/%d  %s%s" % (i, len(cells), first,
              "  [OPTIONAL: requires real GPU]" if is_optional_gpu_cell else ""))
        print("=" * 68)
        try:
            exec(compile(src, "<cell %d>" % i, "exec"), env)
            print("\n--> CELL %d OK\n" % i)
            passed += 1
        except SystemExit as exc:
            # The optional GPU cell raises SystemExit by design when CuPy
            # or a CUDA device is unavailable (true in every CI/sandbox
            # environment).  SystemExit is BaseException, not Exception,
            # so it would otherwise escape uncaught and kill this whole
            # script -- reported previously as a crash, not a clean skip.
            if is_optional_gpu_cell:
                print("\n--> CELL %d SKIPPED (%s)\n" % (i, exc))
                skipped += 1
            else:
                print("\n--> CELL %d FAILED (unexpected SystemExit: %s)\n"
                      % (i, exc))
                break
        except Exception:
            print("\n--> CELL %d FAILED\n" % i)
            traceback.print_exc()
            print()
            break

    print("=" * 68)
    print("RESULT: %d passed, %d skipped, %d total"
          % (passed, skipped, len(cells)))
    print("=" * 68)
    return 0 if passed + skipped == len(cells) else 1


if __name__ == "__main__":
    sys.exit(main())
