"""Execute every code cell of qt_colab.ipynb in order, as Colab would.

Adaptations for the sandbox (no Colab, no GPU, limited time):
  * DEST is redirected to this repo instead of cloning from GitHub
  * matplotlib runs headless (Agg)
  * long loops are shortened via the SHORTEN table below
Everything else -- imports, archive, solver calls, figure saving,
diagnostics, run bookkeeping -- runs exactly as written in the notebook.
"""
import json
import os
import re
import sys
import traceback

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
    passed = 0

    for i, cell in enumerate(cells, 1):
        src = prepare("".join(cell["source"]))
        first = src.strip().split("\n")[0][:64]
        print("=" * 68)
        print("CELL %d/%d  %s" % (i, len(cells), first))
        print("=" * 68)
        try:
            exec(compile(src, "<cell %d>" % i, "exec"), env)
            print("\n--> CELL %d OK\n" % i)
            passed += 1
        except Exception:
            print("\n--> CELL %d FAILED\n" % i)
            traceback.print_exc()
            print()
            break

    print("=" * 68)
    print("RESULT: %d/%d cells executed successfully" % (passed, len(cells)))
    print("=" * 68)
    return 0 if passed == len(cells) else 1


if __name__ == "__main__":
    sys.exit(main())
