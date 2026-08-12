"""Track A: the H4' frustration-crossover experiment.

THE QUESTION
------------
Two eGPE studies of dipolar supersolids describe incompatible physics
(contradiction C8 in PHASE4_REVISED.md):

  * Alaña et al., arXiv:2405.05099 -- in a ROTATING stationary supersolid
    with FEWER vortices than interstitial sites, vortices are NOT pinned at
    density minima.  Their positions vary smoothly with drive and are set by
    the relative phases of neighbouring droplets.  No barrier, nothing to
    unpin from.
  * Poli et al., PRL 131, 223401 (2023) -- vortex UNPINNING produces
    discrete glitch events.  So there IS a barrier.

Hypothesis H4': the two are reconciled by vortex DENSITY.  Define

    R = ell / d,      ell = L^{-1/2} (mean intervortex distance),
                      d   = 2 lambda / sqrt(3) (droplet spacing)

For R > 1 there are fewer vortices than interstitial sites and each can sit
at its phase-preferred position: smooth dynamics, no avalanches (Alaña).
For R < 1 vortices outnumber sites -- GEOMETRIC FRUSTRATION -- and we
hypothesise a crossover to barrier-dominated, avalanching dynamics (Poli).

This script measures avalanche statistics as a function of R.  The crossover
location, if it exists, is the headline number of the paper.  A null result
is equally informative: it decides C8 in Alaña's favour.

WHAT IS MEASURED PER RUN
------------------------
  L(t) raw and annulus-masked (the masked count is primary; both are kept
  because the systematic is 20-50 percent -- see FLAGS.md)
  R(t) = ell/d
  dL/dt event sizes during free decay -> avalanche size distribution
  incompressible / compressible energy split
  energy and norm drift, as a live health check

RESUMABILITY
------------
Colab sessions die.  Every parameter point writes a completed marker to the
archive, and the scan skips points already finished, so re-running the same
command continues where it stopped.  Nothing is recomputed.

USAGE
-----
    python campaign/track_a_scan.py --quick          # smoke test, ~10 min
    python campaign/track_a_scan.py                  # full scan
    python campaign/track_a_scan.py --drives 0.6 0.9 1.2   # subset
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qtsim import (Grid, EGPESolver, gamma_tilde,
                   quasi2d_dipolar_symbol, quasi2d_dipolar_profile)
from qtsim.diagnostics import (plaquette_charges_2d,
                               plaquette_charges_2d_masked,
                               helmholtz_split_2d)
from qtsim.minimize import minimize_energy


# --------------------------------------------------------------- parameters
def parse_args():
    p = argparse.ArgumentParser()
    # Experimental defaults: Casotti et al., Nature 635, 327 (2024), Methods.
    p.add_argument("--eps_dd", type=float, default=1.414,
                   help="a_dd/a_s; experimental supersolid window 1.377-1.453")
    p.add_argument("--l_z", type=float, default=8.6,
                   help="axial length in xi; real trap at n0~1e21 m^-3")
    p.add_argument("--n0_as3", type=float, default=1.17e-4,
                   help="gas parameter; a_s=92.5 a0 at n0=1e21 m^-3")
    p.add_argument("--drives", type=float, nargs="+", default=None,
                   help="stirring Mach numbers to scan (default: 8 values)")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="random seeds per drive (default: 3)")
    p.add_argument("--n_stir", type=int, default=0,
                   help="number of rotating obstacles.  0 = scale with box "
                        "as max(1, cells//3), which keeps vortex production "
                        "per unit AREA roughly constant.  Without scaling, a "
                        "single obstacle in a 3x larger box gives ~6x fewer "
                        "vortices at the same drive (measured: nv=12 at "
                        "cells=5 versus nv=2 at cells=12, Ma=0.6).")
    p.add_argument("--V0_factor", type=float, default=1.5,
                   help="obstacle height in units of mu.  Needs >~1 to shed "
                        "vortices at all; larger values shed far more, so "
                        "this is the coarse control on vortex number while "
                        "--drives is the fine control.")
    p.add_argument("--cells", type=int, default=12,
                   help="droplet cells across the box.  MATTERS: nv(R=1) = "
                        "cells^2*sqrt(3)/2, and a masked vortex count is only "
                        "trustworthy above nv~20 (charge imbalance under 5pct). "
                        "cells=5 gives nv=5 at R=2 and the count is then 100pct "
                        "charge-imbalanced -- useless on the R>1 side that H4' "
                        "needs.  cells=12 gives nv=31 at R=2, nv=14 at R=3.")
    p.add_argument("--dx", type=float, default=0.5,
                   help="grid spacing in xi (<=0.5 resolves the core)")
    p.add_argument("--T_stir", type=float, default=60.0)
    p.add_argument("--T_decay", type=float, default=200.0)
    p.add_argument("--sample_every", type=float, default=0.5,
                   help="diagnostic sampling interval in tau")
    p.add_argument("--quick", action="store_true",
                   help="tiny smoke test: 2 drives, 1 seed, short times")
    p.add_argument("--outdir", type=str, default=None,
                   help="local output dir instead of Google Drive")
    p.add_argument("--backend", type=str, default="cpu", choices=("cpu", "gpu"),
                   help="'cpu' (NumPy, default) or 'gpu' (CuPy).  GPU must be "
                        "requested explicitly; a request with no CuPy/GPU "
                        "present raises rather than silently running on CPU. "
                        "In Colab, install first: !pip install -q cupy-cuda12x")
    return p.parse_args()


# ------------------------------------------------------------------ helpers
def roton_scales(eps_dd, l_z, gamma):
    """Return (k_rot, lambda, d) with d = 2 lambda/sqrt(3) the droplet spacing.

    2*pi/k_rot is the modulation WAVELENGTH.  For a triangular lattice the
    modulation wavevectors are the first Bragg vectors, so the droplet
    spacing is larger by 2/sqrt(3).  Conflating the two produced a spurious
    13.4 percent period error in v2.0 (ratio exactly sqrt(3)/2).
    """
    kk = np.linspace(1e-6, 12.0, 40000)
    D = quasi2d_dipolar_profile(kk, l_z)
    inside = kk ** 2 + 2.0 * (1.0 + eps_dd * D) + 3.0 * gamma
    i = int(np.argmin(inside))
    if inside[i] >= 0:
        raise SystemExit(
            "No roton instability at eps_dd=%.3f, l_z=%.2f (min_inside=%.4f). "
            "No crystal will form; H4' is not testable here."
            % (eps_dd, l_z, inside[i]))
    k_rot = float(kk[i])
    lam = 2 * np.pi / k_rot
    return k_rot, lam, 2 * lam / np.sqrt(3), float(inside[i])


def build_grid(d, cells, dx, backend="cpu"):
    """Commensurate triangular box: Lx = cells*d, Ly = cells*d*sqrt(3)/2."""
    Lx = cells * d
    Ly = cells * d * np.sqrt(3) / 2
    Nx = max(64, int(round(Lx / dx / 2) * 2))
    Ny = max(64, int(round(Ly / dx / 2) * 2))
    return Grid((Nx, Ny), (Lx, Ly), backend=backend)


def lattice_seed(grid, d, cells):
    """Ideal triangular droplet lattice, tiled with periodic images."""
    b1 = np.array([d, 0.0])
    b2 = np.array([d / 2, d * np.sqrt(3) / 2])
    Lx, Ly = grid.lengths
    dens = np.zeros(grid.shape)
    for i in range(-1, cells + 2):
        for j in range(-1, cells + 2):
            c = i * b1 + j * b2
            for sx in (-Lx, 0.0, Lx):
                for sy in (-Ly, 0.0, Ly):
                    dens += np.exp(-((grid.X[0] - c[0] - sx) ** 2
                                     + (grid.X[1] - c[1] - sy) ** 2)
                                   / (2 * 1.2 ** 2))
    return 0.05 + dens


def make_stirrers(grid, mu, Ma, n_stir=1, sigma=3.0, V0_factor=1.5):
    """Rotating obstacles strong and fast enough to shed vortices.

    Both conditions matter: V0 >~ mu to pierce the fluid, and Ma >~ 0.5 with
    c = sqrt(2) in these units.  An earlier version used V0 = Ma^2 = 0.25
    against mu = 1.16 and produced n_vortex = 0 for entire runs.
    """
    V0 = V0_factor * mu
    radius = 0.3 * min(grid.lengths)
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

    return V_stir, V0, radius


def sample(solver, grid, t, d, area):
    """One diagnostic snapshot.  Masked count is primary."""
    q = plaquette_charges_2d(solver.psi)
    qm, mst = plaquette_charges_2d_masked(solver.psi, grid, return_stats=True)
    nv_raw = int((q != 0).sum())
    nv = int((qm != 0).sum())
    net = int((qm == 1).sum() - (qm == -1).sum())
    L = nv / area                      # areal vortex density
    ell = 1.0 / np.sqrt(L) if L > 0 else float("inf")
    Ei, Ec, _ = helmholtz_split_2d(solver.psi, grid)
    n = np.abs(solver.psi) ** 2
    return dict(t=float(t), nv=nv, nv_raw=nv_raw, net_charge=net,
                L=float(L), ell=float(ell), R=float(ell / d),
                mask_reject=float(mst["reject_fraction"]),
                mask_imbalance=float(mst["charge_imbalance"]),
                mask_trustworthy=bool(mst["mask_trustworthy"]),
                E_incomp=float(Ei), E_comp=float(Ec),
                E_total=float(solver.energy()["total"]),
                norm=float(solver.norm()),
                n_max=float(n.max()), n_min=float(n.min()))


def detect_avalanches(recs, theta=3.0):
    """Events in -dL/dt exceeding theta local standard deviations.

    Returns (events, stats).  Sensitivity to theta is reported rather than
    hidden: the caller sweeps theta and stores all of them.
    """
    t = np.array([r["t"] for r in recs])
    L = np.array([r["L"] for r in recs])
    if len(t) < 8:
        return [], dict(n_events=0, note="too few samples")
    drop = -np.gradient(L, t)                # positive when L decreases
    med = np.median(drop)
    mad = np.median(np.abs(drop - med)) + 1e-30
    sigma = 1.4826 * mad                     # robust sigma
    thr = med + theta * sigma
    events, i = [], 0
    while i < len(drop):
        if drop[i] > thr:
            j = i
            while j + 1 < len(drop) and drop[j + 1] > thr:
                j += 1
            size = float(np.trapezoid(drop[i:j + 1], t[i:j + 1])) \
                if j > i else float(drop[i] * (t[1] - t[0]))
            events.append(dict(t_start=float(t[i]), t_end=float(t[j]),
                               size=abs(size), peak=float(drop[i:j + 1].max())))
            i = j + 1
        else:
            i += 1
    waits = [events[k + 1]["t_start"] - events[k]["t_end"]
             for k in range(len(events) - 1)]
    return events, dict(n_events=len(events), threshold=float(thr),
                        robust_sigma=float(sigma),
                        mean_size=float(np.mean([e["size"] for e in events]))
                        if events else 0.0,
                        mean_wait=float(np.mean(waits)) if waits else 0.0)


# --------------------------------------------------------------------- run
def run_point(archive, args, scales, Ma, seed):
    k_rot, lam, d, min_inside = scales
    tag = "trackA Ma%.2f seed%d eps%.3f" % (Ma, seed, args.eps_dd)

    grid = build_grid(d, args.cells, args.dx, backend=args.backend)
    area = float(np.prod(grid.lengths))
    gamma = gamma_tilde(args.eps_dd, args.n0_as3)
    Dk = quasi2d_dipolar_symbol(grid, args.l_z)

    solver = EGPESolver(grid, eps_dd=args.eps_dd, gamma=gamma, Dk=Dk)
    dt = min(0.005, solver.suggested_dt() * 0.6)

    run = archive.new_run(tag, params=dict(
        track="A", hypothesis="H4prime", eps_dd=args.eps_dd, l_z=args.l_z,
        n0_as3=args.n0_as3, Ma=Ma, seed=seed, cells=args.cells,
        grid=list(grid.shape), box=list(grid.lengths), dt=dt,
        k_rot=k_rot, lambda_roton=lam, d_droplet=d, min_inside=min_inside,
        T_stir=args.T_stir, T_decay=args.T_decay,
        n_stir=(args.n_stir if args.n_stir > 0 else max(1, args.cells // 3)),
        V0_factor=args.V0_factor))

    # ---- ground state: lattice seed, then L-BFGS polish -------------------
    Nt = float(grid.integrate(np.ones(grid.shape)))
    solver.psi = lattice_seed(grid, d, args.cells).astype(complex)
    solver.psi *= np.sqrt(Nt / solver.norm())
    solver.step_imag(0.002, 1500, norm_target=Nt)
    info = minimize_energy(solver, Nt, maxiter=1500, verbose=False)
    n = np.abs(solver.psi) ** 2
    contrast = float((n.max() - n.min()) / n.mean())
    print("    ground state: mu=%.5f residual=%.2e contrast=%.3f"
          % (info["mu"], info["residual"], contrast), flush=True)

    # phase noise so different seeds explore different tangles
    rng = np.random.default_rng(seed)
    solver.psi *= np.exp(1j * 0.02 * rng.standard_normal(grid.shape))
    solver.psi *= np.sqrt(Nt / solver.norm())

    n_stir = args.n_stir if args.n_stir > 0 else max(1, args.cells // 3)
    V_stir, V0, radius = make_stirrers(grid, info["mu"], Ma, n_stir=n_stir,
                                       V0_factor=args.V0_factor)
    recs = []
    every = max(1, int(round(args.sample_every / dt)))

    # ---- stirring --------------------------------------------------------
    t = 0.0
    for k in range(int(args.T_stir / dt)):
        solver.V = V_stir(t + 0.5 * dt)
        solver.step_real(dt)
        t += dt
        if k % every == 0:
            r = sample(solver, grid, t, d, area)
            r["phase"] = "stir"
            recs.append(r)

    solver.V = np.zeros(grid.shape)
    E0, N0 = solver.energy()["total"], solver.norm()
    r = sample(solver, grid, t, d, area)
    r["phase"] = "decay_start"
    recs.append(r)
    print("    end of stirring: nv=%d (raw %d, %.0f%% rejected) R=%.3f"
          "  mask imbalance %.1f%%%s"
          % (r["nv"], r["nv_raw"], 100 * r["mask_reject"], r["R"],
             100 * r["mask_imbalance"],
             "" if r["mask_trustworthy"] else "  [MASK SUSPECT]"),
          flush=True)

    # ---- free decay ------------------------------------------------------
    for k in range(int(args.T_decay / dt)):
        solver.step_real(dt)
        t += dt
        if k % every == 0:
            rr = sample(solver, grid, t, d, area)
            rr["phase"] = "decay"
            recs.append(rr)

    dE = abs(solver.energy()["total"] - E0) / abs(E0)
    dN = abs(solver.norm() - N0) / N0
    conservative = bool(dE < 1e-4 and dN < 1e-9)

    decay = [r for r in recs if r["phase"] == "decay"]
    av = {}
    for theta in (2.0, 3.0, 4.0):
        ev, st = detect_avalanches(decay, theta)
        av["theta_%.1f" % theta] = dict(stats=st, events=ev)

    Rs = [r["R"] for r in decay if np.isfinite(r["R"])]
    summary = dict(
        Ma=Ma, seed=seed, contrast=contrast, mu=info["mu"],
        residual=info["residual"], dt=dt,
        R_at_decay_start=r["R"], nv_at_decay_start=r["nv"],
        nv_raw_at_decay_start=r["nv_raw"],
        mask_reject_at_decay_start=float(r["mask_reject"]),
        mask_imbalance_at_decay_start=float(r["mask_imbalance"]),
        mask_trustworthy=bool(r["mask_trustworthy"]),
        R_decay_min=float(min(Rs)) if Rs else None,
        R_decay_max=float(max(Rs)) if Rs else None,
        energy_drift=float(dE), norm_drift=float(dN),
        conservative=conservative,
        avalanches={k: v["stats"] for k, v in av.items()},
        d_droplet=d, lambda_roton=lam)

    run.save_diagnostics(recs, name="timeseries", verbose=False)
    run.save_diagnostics(av, name="avalanches", verbose=False)
    run.save_diagnostics([summary], name="summary", verbose=False)
    run.save_checkpoint(solver.psi, step=int(t / dt), t=t, verbose=False)
    run.finish("completed" if conservative else "completed_unconservative",
               verbose=False)

    flag = "" if conservative else "   [NOT CONSERVATIVE -- distrust]"
    print("    decay: R %.3f -> %.3f | events(3s)=%d | dE/E=%.1e%s"
          % (summary["R_at_decay_start"],
             summary["R_decay_min"] or float("nan"),
             av["theta_3.0"]["stats"]["n_events"], dE, flag), flush=True)
    return summary


def main():
    args = parse_args()
    if args.quick:
        args.drives = args.drives or [0.6, 2.4]
        args.seeds = args.seeds or [0]
        args.cells, args.T_stir, args.T_decay = 8, 15.0, 40.0
    # R = ell/d must BRACKET 1.  Measured on cells=5: Ma=0.8 already gives
    # nv=183 (R=0.34) while nv(R=1) is only ~22, so the useful range is far
    # below the earlier guess.  These low drives, with V0_factor=1.5 and a
    # single obstacle, are chosen to straddle the crossover.
    # Calibrated for the DEFAULT box (cells=12, 4 obstacles, V0=1.5 mu).
    # Measured there: Ma=0.6 -> nv=28 (R=2.11), Ma=2.0 -> nv=91 (R=1.17),
    # both with mask imbalance under 4 percent.  These drives straddle R=1
    # and keep nv above the ~20 trustworthiness floor throughout.
    # A drive list calibrated at one box size does NOT transfer to another.
    args.drives = args.drives or [0.4, 0.6, 0.9, 1.3, 1.8, 2.4, 3.2, 4.2]
    args.seeds = args.seeds or [0, 1, 2]

    gamma = gamma_tilde(args.eps_dd, args.n0_as3)
    scales = roton_scales(args.eps_dd, args.l_z, gamma)
    k_rot, lam, d, min_inside = scales

    if args.outdir:
        from qtsim.drive_io import Archive
        archive = Archive(root=os.path.abspath(args.outdir), verbose=False)
    else:
        from qtsim.drive_io import Archive
        archive = Archive()

    print("=" * 66)
    print("TRACK A -- H4' frustration crossover (contradiction C8)")
    print("=" * 66)
    print("eps_dd=%.3f  l_z=%.2f xi  gamma=%.5f  backend=%s"
          % (args.eps_dd, args.l_z, gamma, args.backend))
    print("roton: k=%.4f  lambda=%.3f xi  droplet spacing d=%.3f xi"
          % (k_rot, lam, d))
    print("       min_inside=%.4f (unstable, crystal will form)" % min_inside)
    print("box: %d x %d cells" % (args.cells, args.cells))
    print("drives: %s" % args.drives)
    print("seeds : %s" % args.seeds)
    grid0 = build_grid(d, args.cells, args.dx, backend=args.backend)
    area0 = float(np.prod(grid0.lengths))
    nv1 = area0 / d ** 2
    print("grid: %d x %d, box %.1f x %.1f xi"
          % (grid0.shape[0], grid0.shape[1], *grid0.lengths))
    print("R = ell/d.  H4' predicts avalanches appear as R crosses below 1.")
    print("  R=1 corresponds to nv = %.0f vortices in this box." % nv1)
    print("  R=2 -> nv=%.0f,  R=1.5 -> nv=%.0f,  R=0.7 -> nv=%.0f,"
          "  R=0.5 -> nv=%.0f"
          % (nv1 / 4, nv1 / 2.25, nv1 / 0.49, nv1 / 0.25))
    print("  obstacles: %d (scaled from cells)"
          % (args.n_stir if args.n_stir > 0 else max(1, args.cells // 3)))
    print("  The scan MUST bracket R=1.  Vortex yield depends on --drives,")
    print("  --V0_factor, --n_stir AND --cells together, so a drive list")
    print("  calibrated for one box size will NOT transfer to another.")
    print("  Run a 2-point probe first (e.g. --drives 0.5 2.0 --seeds 0")
    print("  with short times) and check that R straddles 1 before")
    print("  committing to the full scan.")
    print()

    done = {r.get("title") for r in archive.load_index().get("runs", [])
            if r.get("status", "").startswith("completed")}
    results, t0 = [], time.time()
    for Ma in args.drives:
        for seed in args.seeds:
            tag = "trackA Ma%.2f seed%d eps%.3f" % (Ma, seed, args.eps_dd)
            if tag in done:
                print("  [skip, already completed] %s" % tag, flush=True)
                continue
            print("  Ma=%.2f seed=%d" % (Ma, seed), flush=True)
            try:
                results.append(run_point(archive, args, scales, Ma, seed))
            except Exception as exc:            # keep the scan alive
                print("    FAILED: %s: %s" % (type(exc).__name__, exc),
                      flush=True)

    print()
    print("=" * 66)
    print("SCAN SUMMARY  (%.1f min)" % ((time.time() - t0) / 60))
    print("=" * 66)
    print("   Ma  seed  nv(raw)   rej%  imb%  R_start   R_min  ev(3s)  cons  mask")
    for s in results:
        print("  %4.2f  %4d  %3d(%4d)  %4.0f  %4.1f  %7.3f  %6.3f  %5d  %s  %s"
              % (s["Ma"], s["seed"], s["nv_at_decay_start"],
                 s["nv_raw_at_decay_start"],
                 100 * s["mask_reject_at_decay_start"],
                 100 * s["mask_imbalance_at_decay_start"],
                 s["R_at_decay_start"], s["R_decay_min"] or float("nan"),
                 s["avalanches"]["theta_3.0"]["n_events"],
                 "T" if s["conservative"] else "F",
                 "ok" if s["mask_trustworthy"] else "SUSPECT"))
    thin = [s for s in results if s["nv_at_decay_start"] < 20]
    if thin:
        print()
        print("  !! %d row(s) have nv < 20 at decay start, where the masked"
              % len(thin))
        print("     count is not trustworthy.  Re-run those at larger --cells")
        print("     before drawing any conclusion about the R>1 side.")
    Rs = [s["R_at_decay_start"] for s in results if s["R_at_decay_start"]]
    if Rs and (min(Rs) > 1.0 or max(Rs) < 1.0):
        print()
        print("  !! WARNING: R ranged %.3f to %.3f -- the scan did NOT"
              % (min(Rs), max(Rs)))
        print("     bracket R=1, so the crossover cannot be seen.  Adjust")
        print("     --drives and/or --V0_factor and re-run.")
    print()
    print("H4' READING GUIDE:")
    print("  Group the rows by R.  If event counts are ~0 for R>1 and rise")
    print("  sharply for R<1, that is the frustration crossover and C8 is")
    print("  resolved in favour of density-dependent pinning.  If events are")
    print("  absent at all R, C8 resolves toward Alana (no barrier) and the")
    print("  paper reframes around smooth phase-set dynamics.  If events")
    print("  appear at all R, Alana's smoothness is an artefact of imposed")
    print("  lattice symmetry.  All three outcomes are publishable; only the")
    print("  framing changes.")
    print()
    print("  The 'mask' column matters most at LOW vortex number -- which is")
    print("  exactly the R>1 side of the crossover H4' needs.  Rejection")
    print("  reaches 90-95% there and the masked count becomes strongly")
    print("  charge-imbalanced (100% at nv=1).  Since R scales as nv^-1/2, an")
    print("  ambiguity of 1 vs 3 vortices is a factor 1.7 in R.  Any SUSPECT")
    print("  row on the R>1 side should be re-run at larger --cells rather")
    print("  than interpreted: nv(R=1) = cells^2*sqrt(3)/2, so cells=12 keeps")
    print("  nv=31 at R=2 and nv=14 at R=3.")
    print()
    print("  NOTE, measured: vortex yield is NON-MONOTONIC in drive at the")
    print("  top of the range (Ma=3.2 gave nv=391 while Ma=4.2 gave nv=174).")
    print("  So do not assume larger Ma means smaller R -- read R off the")
    print("  table rather than inferring it from the drive.")
    print()
    print("  Check the 'conservative' column first.  Any False row is a")
    print("  numerical failure, not physics, and must be discarded.")
    print("  Compare theta=2,3,4 in the saved avalanches.json before")
    print("  quoting any event count -- threshold sensitivity is a known")
    print("  systematic (see FLAGS.md).")


if __name__ == "__main__":
    main()
