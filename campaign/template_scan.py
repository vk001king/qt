"""Template: quasi-2D parameter scan over eps_dd and drive amplitude.

Usage (Colab or local):
    python campaign/template_scan.py --eps_dd 1.3 --Ma 0.5 --seed 42

This template demonstrates:
  - Ground-state preparation at given eps_dd
  - Stirring protocol (Gaussian stirrers, protocol P2 from Phase 5B)
  - Periodic checkpointing to Google Drive (or local disk)
  - Diagnostics collection (L(t), spectra, energy channels)
  
Adapt for your HPC batch system by replacing the Drive paths with
your scratch filesystem and wrapping in a SLURM/PBS script.
"""
import argparse
import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qtsim import (Grid, EGPESolver, bare_dipolar_symbol,
                   quasi2d_dipolar_symbol, quasi2d_dipolar_profile)
from qtsim.lhy import gamma_tilde, Q5
from qtsim.diagnostics import (plaquette_charges_2d,
                               plaquette_charges_2d_masked,
                               helmholtz_split_2d)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eps_dd", type=float, default=1.414,
                   help="Dipolar anisotropy a_dd/a_s. Experimental supersolid\n                        window is 1.377-1.453 (a_s=90-95 a0,\n                        a_dd=130.8 a0) per Casotti Nature 2024.")
    p.add_argument("--Ma", type=float, default=1.1,
                   help="Stirring Mach number U/c0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--N", type=int, default=256, help="Grid points per side")
    p.add_argument("--L", type=float, default=128.0, help="Box size (xi)")
    p.add_argument("--n0_as3", type=float, default=1.17e-4,
                   help="Gas parameter n0*a_s^3. 1.17e-4 corresponds to\n                        a_s=92.5 a0 at n0=1e21 m^-3.")
    p.add_argument("--dt", type=float, default=0.01,
                   help="Time step (tau units)")
    p.add_argument("--T_relax", type=float, default=200.0,
                   help="Imaginary-time relaxation budget")
    p.add_argument("--T_stir", type=float, default=500.0,
                   help="Stirring duration")
    p.add_argument("--T_decay", type=float, default=2000.0,
                   help="Free decay duration")
    p.add_argument("--checkpoint_every", type=int, default=500,
                   help="Checkpoint interval (steps)")
    p.add_argument("--outdir", type=str, default="./results",
                   help="Output directory (ignored when --use_archive)")
    p.add_argument("--kernel", type=str, default="quasi2d",
                   choices=("quasi2d", "bare"),
                   help="Dipolar kernel. 'quasi2d' is the projected kernel "
                        "with a roton, required for supersolid physics; "
                        "'bare' has no roton and gives only a box-scale "
                        "mode (regression use only).")
    p.add_argument("--l_z", type=float, default=8.6,
                   help="Axial confinement length in xi. Real trap\n                        (omega_z=2pi x 103 Hz) with n0~1e21 m^-3\n                        gives l_z/xi ~ 8.6; range 6-19 over\n                        plausible densities.")
    p.add_argument("--V0_factor", type=float, default=3.0,
                   help="Obstacle height in units of mu (need >~1 to shed)")
    p.add_argument("--n_stir", type=int, default=3,
                   help="Number of rotating obstacles")
    p.add_argument("--backend", type=str, default="cpu", choices=("cpu", "gpu"),
                   help="'cpu' (NumPy, default) or 'gpu' (CuPy).  Explicit "
                        "opt-in only; raises if requested and unavailable.")
    p.add_argument("--use_archive", action="store_true",
                   help="Save into MyDrive/Research/Quantum_Turbulence/"
                        "<date>_<time>_<title>/ with full structure")
    return p.parse_args()


def make_stirrers(grid, mu, n_stir=3, sigma=3.0, radius=16.0,
                  Ma=1.1, V0_factor=3.0):
    """Rotating Gaussian obstacles that actually nucleate vortices.

    Two conditions must both hold for vortex shedding, and the original
    version of this function satisfied neither (verified: it produced
    n_vortex = 0 for the whole run while pumping only sound):

      (1) the obstacle must pierce the condensate, V0 >~ mu.  The old
          code used V0 = Ma**2 = 0.25 against mu = 1.16, i.e. 22 percent
          -- it only dented the density.
      (2) the local flow must exceed the critical velocity, roughly
          Ma >~ 0.5 with c = sqrt(2) in these units.  The old code gave
          Ma = 0.35.

    With V0 = 3*mu and Ma = 1.1 the vortex count rises 0 -> 12 -> 36 ->
    94 -> 206 and saturates: a driven steady-state tangle.

    Parameters
    ----------
    mu        : chemical potential of the relaxed ground state
    n_stir    : number of obstacles on the ring
    sigma     : obstacle Gaussian width (units of xi)
    radius    : ring radius (units of xi)
    Ma        : stirring Mach number, v / c with c = sqrt(2)
    V0_factor : obstacle height in units of mu
    """
    V0 = V0_factor * mu
    omega = Ma * np.sqrt(2.0) / radius
    angles0 = np.linspace(0, 2 * np.pi, n_stir, endpoint=False)

    def V_stir(t):
        V = np.zeros(grid.shape)
        for a0 in angles0:
            a = a0 + omega * t
            xc, yc = radius * np.cos(a), radius * np.sin(a)
            r2 = (grid.X[0] - xc) ** 2 + (grid.X[1] - yc) ** 2
            V += V0 * np.exp(-r2 / (2 * sigma ** 2))
        return V

    return V_stir


def collect_diagnostics(solver, grid, t):
    """Snapshot of all observables needed for Figs. 1–5."""
    n = np.abs(solver.psi) ** 2
    E = solver.energy()
    # Report BOTH the raw and the annulus-masked count.  In a droplet
    # crystal the inter-droplet voids are near-vacuum and the raw detector
    # finds spurious windings there; masking changes L by 20-50 percent,
    # which is a systematic that must be quoted, not hidden.
    q = plaquette_charges_2d(solver.psi)
    nv_plus = int((q == 1).sum())
    nv_minus = int((q == -1).sum())
    qm, mstat = plaquette_charges_2d_masked(solver.psi, grid,
                                           return_stats=True)
    nv_plus_m = int((qm == 1).sum())
    nv_minus_m = int((qm == -1).sum())
    Ei, Ec, Etot = helmholtz_split_2d(solver.psi, grid)
    return {
        "t": float(t),
        "N": float(solver.norm()),
        "E_total": E["total"],
        "E_kin": E["kin"],
        "E_int": E["int"],
        "E_dd": E["dd"],
        "E_lhy": E["lhy"],
        "E_incomp": Ei,
        "E_comp": Ec,
        "n_vortex_plus": nv_plus,
        "n_vortex_minus": nv_minus,
        "n_vortex_total": nv_plus + nv_minus,
        "n_vortex_plus_masked": nv_plus_m,
        "n_vortex_minus_masked": nv_minus_m,
        "n_vortex_total_masked": nv_plus_m + nv_minus_m,
        "vortex_mask_reject_fraction": mstat["reject_fraction"],
        "vortex_mask_charge_imbalance": mstat["charge_imbalance"],
        "vortex_mask_trustworthy": mstat["mask_trustworthy"],
        "n_max": float(n.max()),
        "n_min": float(n.min()),
    }


def save_checkpoint(solver, step, t, params, outdir):
    os.makedirs(outdir, exist_ok=True)
    tag = f"eps{params['eps_dd']:.2f}_Ma{params['Ma']:.2f}_s{params['seed']}"
    fname = os.path.join(outdir, f"ckpt_{tag}_step{step:08d}.npz")
    np.savez_compressed(fname, psi=solver.psi, step=step, t=t,
                        **{k: v for k, v in params.items()
                           if isinstance(v, (int, float, str))})
    return fname


def main():
    args = parse_args()
    params = vars(args)
    rng = np.random.default_rng(args.seed)

    print(f"=== SSQT campaign run: eps_dd={args.eps_dd}, Ma={args.Ma}, "
          f"seed={args.seed} ===")

    # Structured archive (checks folders, creates only what is missing)
    run = None
    if args.use_archive:
        from qtsim.drive_io import Archive
        archive = Archive()
        title = (f"stirred tangle eps{args.eps_dd:.2f} "
                 f"Ma{args.Ma:.2f} seed{args.seed}")
        run = archive.new_run(title, params=params)

    # --- Grid and solver setup ---
    grid = Grid((args.N, args.N), (args.L, args.L), backend=args.backend)
    gamma = gamma_tilde(args.eps_dd, args.n0_as3)
    print(f"gamma_tilde = {gamma:.6f}, Q5 = {Q5(args.eps_dd):.6f}")

    if args.kernel == "quasi2d":
        # Projected quasi-2D kernel:
        #   D(k) = 2*sqrt(2) - 3*sqrt(2*pi)*u*erfcx(u),   u = k*l_z/2
        # runs from +2sqrt2 (repulsive) to -sqrt2 (attractive).  That sign
        # change is the roton and it is what makes a droplet crystal
        # possible.  The bare kernel has no such structure, so turbulence
        # run on it is turbulence in a rippled superfluid, NOT a supersolid.
        Dk = quasi2d_dipolar_symbol(grid, args.l_z)
        kk = np.linspace(1e-6, 12, 20000)
        Dp = quasi2d_dipolar_profile(kk, args.l_z)
        inside = kk ** 2 + 2.0 * (1.0 + args.eps_dd * Dp) + 3.0 * gamma
        imin = int(np.argmin(inside))
        lam = 2.0 * np.pi / kk[imin]
        print(f"kernel=quasi2d  l_z={args.l_z} xi   roton lambda={lam:.3f} xi"
              f"   triangular a={2*lam/np.sqrt(3):.3f} xi")
        print(f"  uniform state roton-unstable: {inside[imin] < 0}")
        if inside[imin] >= 0:
            print("  NOTE: no roton instability here, so no crystal will "
                  "form.  Increase eps_dd or l_z for supersolid physics.")
    else:
        # Bare periodic symbol.  In 2D the dipoles are polarized IN-PLANE;
        # truncating a 3D axis (0,0,1) to 2D gives (0,0), the zero vector,
        # which silently produced NaN everywhere (caught by a live run).
        # No roton -> no crystal.  Kept for regression only.
        ehat = (0, 1) if grid.dim == 2 else (0, 0, 1)
        Dk = bare_dipolar_symbol(grid, ehat)
        print("kernel=bare  (no roton; box-scale mode only -- regression use)")

    solver = EGPESolver(grid, eps_dd=args.eps_dd, gamma=gamma, Dk=Dk)
    ph = solver.max_phase_per_step(args.dt)
    print(f"timestep check: {ph:.3f} rad/step "
          f"(safe dt <= {solver.suggested_dt():.4g})")
    if ph > solver.PHASE_HARD:
        raise SystemExit(f"dt={args.dt} unsafe on this grid; "
                         f"use --dt {solver.suggested_dt():.4g} or smaller")

    # --- Ground state ---
    print("Finding ground state (imaginary time)...")
    solver.psi = np.ones(grid.shape, dtype=complex)
    noise = 0.01 * (rng.standard_normal(grid.shape)
                     + 1j * rng.standard_normal(grid.shape))
    solver.psi += noise
    N_target = float(grid.integrate(np.ones(grid.shape)))
    solver.psi *= np.sqrt(N_target / solver.norm())

    dtau = 0.005
    n_relax = int(args.T_relax / dtau)
    solver.step_imag(dtau, n_relax, norm_target=N_target)
    mu, res = solver.mu_and_residual()
    print(f"Ground state: mu={mu:.4f}, residual={res:.2e}")

    # --- Stirring phase ---
    print(f"Stirring for T={args.T_stir} ...")
    V_stir = make_stirrers(grid, mu, n_stir=args.n_stir, Ma=args.Ma,
                           V0_factor=args.V0_factor)
    print(f"  obstacle V0 = {args.V0_factor*mu:.3f} = "
          f"{args.V0_factor:.1f} x mu   Mach = {args.Ma:.2f}")
    diagnostics_log = []
    t = 0.0
    step = 0
    n_stir_steps = int(args.T_stir / args.dt)

    for i in range(n_stir_steps):
        solver.V = V_stir(t + 0.5 * args.dt)
        solver.step_real(args.dt)
        t += args.dt
        step += 1

        if step % args.checkpoint_every == 0:
            d = collect_diagnostics(solver, grid, t)
            diagnostics_log.append(d)
            print(f"  step {step}, t={t:.1f}, "
                  f"n_vortex={d['n_vortex_total']}"
                  f"(masked {d['n_vortex_total_masked']}), "
                  f"E_i={d['E_incomp']:.3f}, E_c={d['E_comp']:.3f}")
            if run is not None:
                run.save_checkpoint(solver.psi, step, t, verbose=False)
            else:
                save_checkpoint(solver, step, t, params, args.outdir)

    # --- Free decay ---
    print(f"Free decay for T={args.T_decay} ...")
    solver.V = np.zeros(grid.shape)
    n_decay_steps = int(args.T_decay / args.dt)

    for i in range(n_decay_steps):
        solver.step_real(args.dt)
        t += args.dt
        step += 1

        if step % args.checkpoint_every == 0:
            d = collect_diagnostics(solver, grid, t)
            diagnostics_log.append(d)
            print(f"  step {step}, t={t:.1f}, "
                  f"n_vortex={d['n_vortex_total']}"
                  f"(masked {d['n_vortex_total_masked']}), "
                  f"E_i={d['E_incomp']:.3f}, E_c={d['E_comp']:.3f}")
            if run is not None:
                run.save_checkpoint(solver.psi, step, t, verbose=False)
            else:
                save_checkpoint(solver, step, t, params, args.outdir)

    # --- Save diagnostics ---
    if run is not None:
        run.save_diagnostics(diagnostics_log)
        run.finish(status="completed")
    else:
        tag = f"eps{args.eps_dd:.2f}_Ma{args.Ma:.2f}_s{args.seed}"
        dfile = os.path.join(args.outdir, f"diag_{tag}.json")
        os.makedirs(args.outdir, exist_ok=True)
        with open(dfile, "w") as f:
            json.dump(diagnostics_log, f, indent=1)
        print(f"Done. Diagnostics saved to {dfile}")


if __name__ == "__main__":
    main()
