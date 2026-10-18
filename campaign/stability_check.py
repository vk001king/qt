"""Real-time stability of the vortex-free ground-state crystal (control for melting)."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from qtsim import EGPESolver, quasi2d_dipolar_symbol
from qtsim.diagnostics import droplet_centers
import track_a_scan as ta
from decay_run import psi6
amp = float(sys.argv[1]); T = float(sys.argv[2]); tag = sys.argv[3]
meta = json.load(open("/home/claude/runs2/gs_dx05.json")); psi0 = np.load("/home/claude/runs2/gs_dx05.npy")
grid = ta.build_grid(meta["lattice"], meta["cells"], meta["dx"]); L = np.array(grid.lengths)
s = EGPESolver(grid, eps_dd=meta["eps"], gamma=meta["gamma"], Dk=quasi2d_dipolar_symbol(grid, meta["l_z"]))
rng = np.random.default_rng(7)
s.psi = psi0 * np.exp(1j * amp * rng.standard_normal(grid.shape)) * (1 + amp * rng.standard_normal(grid.shape))
s.psi *= np.sqrt(float(grid.integrate(np.abs(psi0) ** 2)) / s.norm())
out = []
E0 = s.energy()["total"]
for i in range(int(T / 2.5) + 1):
    n = np.abs(s.psi) ** 2; cen = droplet_centers(n, grid, meta["lattice"])
    out.append((2.5 * i, len(cen), *psi6(cen, L)))
    s.step_real(0.005, 500)
print(tag, "dE/E %.1e" % (abs(s.energy()["total"] - E0) / abs(E0)))
for r in out: print("  t %5.1f droplets %d psi6 %.3f global %.3f" % r)
