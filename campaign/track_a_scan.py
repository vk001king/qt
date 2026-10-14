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
import glob
import hashlib
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
                               helmholtz_split_2d, persistent_charges,
                               droplet_centers, interstitial_charges,
                               link_coherence, measured_spacing)
from collections import deque
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
    p.add_argument("--report", action="store_true",
                   help="do NOT simulate: read every finished Track A run in "
                        "the archive, keep only those whose STORED params "
                        "match the current --cells/--T_decay/... settings, "
                        "print the table and write trackA_results.csv")
    p.add_argument("--surrogates", type=int, default=0,
                   help="with --report: number of surrogates per run for the "
                        "formal null test (0 = skip; 199 recommended)")
    p.add_argument("--persist", type=int, default=3,
                   help="v2.8: frames a vortex must persist in to enter the "
                        "filtered count nv_p (current + persist-1 previous "
                        "samples).  1 = no filter.")
    p.add_argument("--persist_radius", type=float, default=1.0,
                   help="v2.8: how far (xi) a vortex may move between samples "
                        "and still count as the same one")
    p.add_argument("--snapshots", type=int, default=0,
                   help="v2.8: save this many CONSECUTIVE psi samples "
                        "(complex64) at decay start, mid-decay and end, for "
                        "offline checks of vortex detection.  0 = off. "
                        "About 0.4 MB per frame at cells=12.")
    p.add_argument("--count_version", type=int, default=None,
                   help="v2.8: vortex-count version a point is keyed on "
                        "(1 = masked count only, <= v2.7; 2 = adds nv_p and "
                        "nv_lat, v2.8; 3 = adds link coherence, measured "
                        "spacing and windowed values, v2.9).  Scans default "
                        "to 3. "
                        "--report ignores it unless given, and then shows "
                        "which count each row used.")
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
    """Ideal triangular droplet lattice, tiled with periodic images.

    BUG FIX (v2.5): this function was written before the GPU backend
    existed and built `dens` as a plain host NumPy array, then accumulated
    Gaussian bumps computed from `grid.X` -- which is a CuPy array on
    `backend="gpu"`.  Mixing a host `np.ndarray` into an in-place `+=` with
    a CuPy operand raises `TypeError: Unsupported type <class
    'numpy.ndarray'>`: CuPy deliberately refuses silent host/device
    mixing rather than doing something slow or wrong.  Fixed by building
    entirely on the HOST with plain NumPy (cheap, one-time, and identical
    across backends) and moving to the grid's backend in a single transfer
    at the end -- the same pattern already used in kernels.py for kernel
    construction.
    """
    b1 = np.array([d, 0.0])
    b2 = np.array([d / 2, d * np.sqrt(3) / 2])
    Lx, Ly = grid.lengths
    X_host = [grid.to_host(Xi) for Xi in grid.X]
    dens = np.zeros(grid.shape)
    for i in range(-1, cells + 2):
        for j in range(-1, cells + 2):
            c = i * b1 + j * b2
            for sx in (-Lx, 0.0, Lx):
                for sy in (-Ly, 0.0, Ly):
                    dens += np.exp(-((X_host[0] - c[0] - sx) ** 2
                                     + (X_host[1] - c[1] - sy) ** 2)
                                   / (2 * 1.2 ** 2))
    return grid.asarray(0.05 + dens)


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
    xp = grid.xp

    def V_stir(t):
        """Called every timestep -- this IS the hot path GPU is meant to
        accelerate, so unlike lattice_seed (a one-time cost) this stays
        entirely on `grid.xp`/device with NO host round trip per call."""
        V = xp.zeros(grid.shape)
        for a0 in angles0:
            a = a0 + omega * t
            xc, yc = radius * np.cos(a), radius * np.sin(a)  # host scalars
            r2 = (grid.X[0] - xc) ** 2 + (grid.X[1] - yc) ** 2
            V = V + V0 * xp.exp(-r2 / (2 * sigma ** 2))
        return V

    return V_stir, V0, radius


def make_tracker(args, grid):
    """History of earlier charge maps for the persistence filter (v2.8)."""
    k = max(1, int(args.persist))
    r = max(1, int(round(args.persist_radius / float(grid.dx[0]))))
    return dict(hm=deque(maxlen=k - 1), hr=deque(maxlen=k - 1), k=k, r=r)


def sample(solver, grid, t, d, area, tracker=None):
    """One diagnostic snapshot.  Masked count is primary.

    v2.8: with a tracker, also the persistence-filtered counts nv_p
    (masked) and nv_p_raw (unmasked) -- see qtsim.diagnostics.
    persistent_charges for why.  p_ready is False until the history holds
    persist-1 frames (the first samples are filtered against fewer)."""
    q = plaquette_charges_2d(solver.psi)
    qm, mst = plaquette_charges_2d_masked(solver.psi, grid, return_stats=True)
    nv_raw = int((q != 0).sum())
    nv = int((qm != 0).sum())
    extra = {}
    if tracker is not None:
        qp = persistent_charges(qm, list(tracker["hm"]), tracker["r"])
        qpr = persistent_charges(q, list(tracker["hr"]), tracker["r"])
        nv_p = int((qp != 0).sum())
        # interstitial (lattice-level) winding: the robust count, v2.8
        nden = grid.xp.abs(solver.psi) ** 2
        cen = droplet_centers(nden, grid, d)
        wl, _ = interstitial_charges(q, cen, grid)
        nv_lat = int(np.abs(wl).sum())
        L_lat = nv_lat / area
        lc = link_coherence(solver.psi, cen, grid)          # v2.9
        d_m = measured_spacing(len(cen), area)
        L_p = nv_p / area
        extra = dict(nv_lat=nv_lat, n_droplets=int(len(cen)),
                     n_sites=int(len(wl)),
                     n_multi=int((np.abs(wl) > 1).sum()),
                     L_lat=float(L_lat),
                     R_lat=float(1.0 / np.sqrt(L_lat) / d) if L_lat > 0
                     else float("inf"),
                     d_meas=d_m,
                     R_lat_m=float(1.0 / np.sqrt(L_lat) / d_m) if L_lat > 0
                     else float("inf"),
                     coh_cos=lc["cos"], coh_frac_gt=lc["frac_gt_half_pi"],
                     coh_median=lc["median_abs"],
                     nv_p=nv_p, nv_p_raw=int((qpr != 0).sum()),
                     net_charge_p=int((qp == 1).sum() - (qp == -1).sum()),
                     L_p=float(L_p),
                     R_p=float(1.0 / np.sqrt(L_p) / d) if L_p > 0
                     else float("inf"),
                     p_ready=bool(len(tracker["hm"]) == tracker["k"] - 1))
        if tracker["k"] > 1:
            tracker["hm"].append(qm)
            tracker["hr"].append(q)
    net = int((qm == 1).sum() - (qm == -1).sum())
    L = nv / area                      # areal vortex density
    ell = 1.0 / np.sqrt(L) if L > 0 else float("inf")
    Ei, Ec, _ = helmholtz_split_2d(solver.psi, grid)
    # grid.xp.abs (not global np.abs) for consistency with the rest of the
    # backend-aware code, rather than relying on CuPy's __array_function__
    # dispatch of top-level numpy calls, which is a real but implicit path.
    n = grid.xp.abs(solver.psi) ** 2
    return dict(t=float(t), nv=nv, nv_raw=nv_raw, net_charge=net,
                L=float(L), ell=float(ell), R=float(ell / d),
                mask_reject=float(mst["reject_fraction"]),
                mask_imbalance=float(mst["charge_imbalance"]),
                mask_trustworthy=bool(mst["mask_trustworthy"]),
                E_incomp=float(Ei), E_comp=float(Ec),
                E_total=float(solver.energy()["total"]),
                norm=float(solver.norm()),
                n_max=float(n.max()), n_min=float(n.min()), **extra)


# np.trapezoid exists only in NumPy >= 2.0; np.trapz was removed in 2.x.
# requirements.txt allows numpy>=1.21, so support both.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def _events_from_series(t, L, theta):
    """Core event detector on arrays.  See detect_avalanches for the rules.

    Robust scale: sigma = 1.4826 * MAD of -dL/dt.

    BUG FIX (v2.6, finding F3): the old code used MAD + 1e-30.  When most
    -dL/dt samples are identical (small, nearly constant vortex number --
    exactly the R > 1 side) MAD is 0, sigma collapses to ~1e-30, and EVERY
    nonzero drop counted as an event.  Now:
      MAD > 0           -> sigma = 1.4826 MAD          (sigma_method 'mad')
      MAD == 0, std > 0 -> sigma = std(-dL/dt)         ('std_fallback')
      std == 0          -> no events, flat signal      ('flat')
    The method used is returned so the fallback is never silent.
    Cost: O(n log n) for the medians, n = number of samples.
    """
    drop = -np.gradient(L, t)                # positive when L decreases
    med = float(np.median(drop))
    mad = float(np.median(np.abs(drop - med)))
    scale = max(float(np.max(np.abs(drop))), 1e-300)
    if mad > 1e-12 * scale:
        sigma, method = 1.4826 * mad, "mad"
    else:
        sd = float(np.std(drop))
        if sd > 1e-12 * scale and np.any(drop != 0):
            sigma, method = sd, "std_fallback"
        else:
            return [], dict(n_events=0, threshold=float("inf"),
                            robust_sigma=0.0, sigma_method="flat",
                            mean_size=0.0, mean_wait=0.0)
    thr = med + theta * sigma
    events, i = [], 0
    while i < len(drop):
        if drop[i] > thr:
            j = i
            while j + 1 < len(drop) and drop[j + 1] > thr:
                j += 1
            size = float(_trapz(drop[i:j + 1], t[i:j + 1])) \
                if j > i else float(drop[i] * (t[1] - t[0]))
            events.append(dict(t_start=float(t[i]), t_end=float(t[j]),
                               size=abs(size), peak=float(drop[i:j + 1].max())))
            i = j + 1
        else:
            i += 1
    waits = [events[k + 1]["t_start"] - events[k]["t_end"]
             for k in range(len(events) - 1)]
    return events, dict(n_events=len(events), threshold=float(thr),
                        robust_sigma=float(sigma), sigma_method=method,
                        mean_size=float(np.mean([e["size"] for e in events]))
                        if events else 0.0,
                        mean_wait=float(np.mean(waits)) if waits else 0.0)


def detect_avalanches(recs, theta=3.0):
    """Events in -dL/dt exceeding median + theta robust standard deviations.

    An event is a contiguous run of samples above threshold.  Returns
    (events, stats).  Sensitivity to theta is reported rather than hidden:
    the caller sweeps theta and stores all of them.
    """
    t = np.array([r["t"] for r in recs], dtype=float)
    L = np.array([r["L"] for r in recs], dtype=float)
    if len(t) < 8:
        return [], dict(n_events=0, note="too few samples")
    return _events_from_series(t, L, theta)


# ------------------------------------------------------ formal null test
def _detrend_endpoints(x):
    """Remove the straight line joining the end points (Theiler et al.).

    Fourier surrogates treat the series as periodic; a decaying L(t) has a
    large end-to-start jump that would leak power into every frequency.
    Returns (residual, trend) with x = residual + trend.
    """
    n = len(x)
    trend = x[0] + (x[-1] - x[0]) * np.arange(n) / max(n - 1, 1)
    return x - trend, trend


def phase_randomised(x, rng):
    """Fourier-transform surrogate: same power spectrum, random phases.

    Null hypothesis: x is a stationary LINEAR GAUSSIAN process.  Heavy-
    tailed bursts and nonlinear structure are destroyed, so an observed
    event count far above the surrogates indicates bursts beyond what a
    Gaussian process with the same correlations produces.  O(n log n).
    """
    n = len(x)
    X = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, len(X))
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0                          # Nyquist term must stay real
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n) + x.mean()


def iaaft(x, rng, n_iter=200):
    """Iterative amplitude-adjusted Fourier surrogate (Schreiber-Schmitz).

    Keeps the exact value distribution of x and (approximately) its power
    spectrum.  Null: a linear Gaussian process seen through a static
    monotone transform.  Because it keeps the value distribution, it keeps
    the integer quantisation of the vortex count, which the plain phase-
    randomised surrogate does not.  O(n_iter * n log n).
    """
    amp = np.abs(np.fft.rfft(x))
    sorted_x = np.sort(x)
    y = rng.permutation(x)
    for _ in range(n_iter):
        Y = np.fft.rfft(y)
        y = np.fft.irfft(amp * np.exp(1j * np.angle(Y)), len(x))
        ranks = np.argsort(np.argsort(y))
        y_new = sorted_x[ranks]
        if np.array_equal(y_new, y):
            break
        y = y_new
    return y


def surrogate_test(recs, thetas=(2.0, 3.0, 4.0), n_surr=199, seed=0,
                   methods=("phase", "iaaft")):
    """Formal null test of the avalanche event count for ONE run.

    For each surrogate of L(t) (end-point detrended, surrogate made from
    the residual, trend added back) the SAME detector is applied.  One-
    sided p-value with the +1 correction (Davison & Hinkley):

        p = (1 + #{surrogates with n_events >= observed}) / (1 + n_surr)

    so the smallest attainable p is 1/(n_surr+1) = 0.005 for 199.
    A plain time-shuffle is NOT offered: it destroys autocorrelation and
    therefore only tests temporal clustering, not heavy tails.

    Returns {method: {theta_X: {observed, surr_mean, surr_std, p}}}.
    Cost: n_surr * len(methods) detector calls on ~400 samples (seconds).
    """
    t = np.array([r["t"] for r in recs], dtype=float)
    L = np.array([r["L"] for r in recs], dtype=float)
    out = {}
    if len(t) < 8:
        return {"note": "too few samples"}
    resid, trend = _detrend_endpoints(L)
    obs = {th: _events_from_series(t, L, th)[1]["n_events"] for th in thetas}
    for m in methods:
        rng = np.random.default_rng(seed)
        counts = {th: [] for th in thetas}
        for _ in range(n_surr):
            s = phase_randomised(resid, rng) if m == "phase" \
                else iaaft(resid, rng)
            Ls = s + trend
            for th in thetas:
                counts[th].append(_events_from_series(t, Ls, th)[1]["n_events"])
        out[m] = {}
        for th in thetas:
            c = np.array(counts[th])
            out[m]["theta_%.1f" % th] = dict(
                observed=int(obs[th]), surr_mean=float(c.mean()),
                surr_std=float(c.std()),
                p=float((1 + np.sum(c >= obs[th])) / (1 + n_surr)))
    out["n_surr"] = int(n_surr)
    return out


# ------------------------------------- matched null: annihilation clustering
# v2.7 (handoff step 2).  The phase / IAAFT surrogates above model L(t) as a
# continuous Gaussian signal.  The masked vortex count is not that: it is an
# integer staircase that only steps DOWN (pair annihilation) apart from mask
# flicker, and over a long decay its step rate falls steeply.  The phase
# null flagged 5/6 long-decay runs while IAAFT flagged 1/6, which points at
# a mis-specified null rather than physics.
#
# The physically matched question is: are annihilations CLUSTERED in time
# beyond what a time-varying rate alone produces?  So:
#
#   events   each pair annihilation = one event (masked count falls by 2),
#            located to the sampling interval in which it happened.
#   null H0  inhomogeneous Poisson process whose rate lambda(t) is estimated
#            from the run itself -- same total count, same slow decay of
#            the rate, no clustering beyond that.
#   rate     least concave majorant (LCM) of the cumulative event count =
#            Grenander maximum-likelihood estimate of a NON-INCREASING
#            rate.  Its only assumption is that annihilation slows down as
#            the tangle thins.  A burst after a quiet stretch is bridged by
#            a chord of the majorant, so it is NOT absorbed into the rate.
#   statistic  time-rescaling theorem: u = Lambda(t) turns the null into a
#            unit-rate Poisson process, whose waiting times are Exp(1), so
#            CV = 1 and burstiness B = (CV-1)/(CV+1) = 0.  Clustering gives
#            CV > 1, B > 0.  Shinomoto's local variation LV (= 1 for Poisson
#            at any slowly varying rate) is kept as a cross-check.
#   p-value  parametric bootstrap conditioned on the event count: surrogate
#            counts ~ Multinomial(N, Lambda-increments) on the SAME sampling
#            grid, drawn from the LCM smoothed over 8 expected events (raw
#            Grenander bootstrap is inconsistent, see _smooth_cum_rate), and
#            the rate is RE-ESTIMATED on every surrogate, so the estimation
#            step is inside the null distribution.  One-sided upper p with
#            the +1 correction, as in surrogate_test.
def annihilation_events(t, nv, med_win=3):
    """Pair-annihilation counts per sampling interval from a masked count.

    Mask flicker (a vortex dropping out of the mask for one sample, then
    back) would otherwise create a fake drop-and-recover.  Two steps:
      1. running median over `med_win` samples (odd; 1 = off) removes
         isolated one-sample spikes in either direction;
      2. running minimum: in free decay the vortex number cannot rise, so
         any remaining upward step is mask noise and is not undone later.
    Cumulative pairs C_k = floor((env_0 - env_k)/2); odd masked drops
    (a vortex leaving the mask) accumulate until a full pair is removed.

    Returns (counts, info): counts[k] = pairs annihilated in (t_k, t_k+1].
    info['flicker'] is the fraction of RAW sample-to-sample steps that go
    UP -- a direct measure of how noisy the masked count is.
    """
    nv = np.asarray(nv, dtype=float)
    n = len(nv)
    if med_win > 1 and n >= med_win:
        h = med_win // 2
        pad = np.r_[np.full(h, nv[0]), nv, np.full(h, nv[-1])]
        win = np.lib.stride_tricks.sliding_window_view(pad, med_win)
        filt = np.median(win, axis=1)
    else:
        filt = nv.copy()
    env = np.minimum.accumulate(filt)
    C = np.floor((env[0] - env) / 2.0 + 1e-9)
    counts = np.diff(C).astype(int)
    steps = np.diff(nv)
    nz = steps[steps != 0]
    info = dict(n_pairs=int(counts.sum()),
                flicker=float(np.mean(nz > 0)) if len(nz) else 0.0,
                nv_start=float(nv[0]), nv_end=float(nv[-1]),
                med_win=int(med_win))
    return counts, info


def _lcm(x, y):
    """Least concave majorant of points (x, y), x increasing; values at x.

    Upper convex hull by Andrew's monotone chain, O(n).
    """
    hx, hy = [x[0]], [y[0]]
    for xi, yi in zip(x[1:], y[1:]):
        while len(hx) >= 2 and ((hy[-1] - hy[-2]) * (xi - hx[-2])
                                <= (yi - hy[-2]) * (hx[-1] - hx[-2])):
            hx.pop()
            hy.pop()
        hx.append(xi)
        hy.append(yi)
    return np.interp(x, hx, hy)


def _cum_rate(t, counts, rate="lcm"):
    """Estimated cumulative intensity Lambda at the sample times t.

    rate='lcm'    Grenander (non-increasing rate) -- the matched null.
    rate='const'  homogeneous Poisson -- the UNMATCHED null, kept only to
                  show how much a decaying rate alone inflates the CV.
    (A Gaussian-kernel rate with a fixed bandwidth was tried and dropped:
    when the early decay is faster than the bandwidth it flattens the
    initial peak and gave 78 percent false positives on synthetic data.)
    Lambda(t_0) = 0 and Lambda(t_end) = total count in both cases.
    """
    C = np.r_[0.0, np.cumsum(counts)]
    N = C[-1]
    if N == 0:
        return C
    if rate == "lcm":
        return _lcm(t, C)
    if rate == "const":
        return N * (t - t[0]) / (t[-1] - t[0])
    raise ValueError("rate must be 'lcm' or 'const'")


def _smooth_cum_rate(t, Lam, h_events):
    """Smooth a concave Lambda(t) with a bandwidth of `h_events` EXPECTED
    events, i.e. in rescaled time, so the smoothing is scale-free: fine
    where annihilations are frequent (early), coarse where rare (late).

    Done on the inverse t(u), which is convex for concave Lambda; Gaussian
    convolution keeps it convex, so the smoothed rate stays non-increasing.
    The ends are extended linearly with the exact one-sided slopes 1/lambda
    (this preserves convexity).  The flat tail after the last annihilation
    keeps rate 0, as in the LCM.

    WHY: bootstrapping from the raw Grenander estimate is inconsistent
    (Kosorok 2008; Sen, Banerjee & Woodroofe, Ann. Stat. 38, 1953 (2010))
    -- its rate jumps put extra variance into every surrogate.  Measured on
    synthetic H0 series: mean p 0.60-0.64 unsmoothed versus 0.49-0.58 with
    h = 8 events, while power against avalanches stays 20/20.
    """
    N = float(Lam[-1])
    h = min(float(h_events), N / 8.0)
    if h <= 0 or N <= 0:
        return Lam
    j = int(np.searchsorted(Lam, N - 1e-9 * N))     # first index at N
    tt, LL = t[:j + 1], Lam[:j + 1]
    keep = np.r_[True, np.diff(LL) > 0]
    tt, LL = tt[keep], LL[keep]
    if len(tt) < 3:
        return Lam
    du = min(0.05, h / 20.0)
    u = np.arange(0.0, N + 0.5 * du, du)
    tu = np.interp(u, LL, tt)
    s0 = (tt[1] - tt[0]) / (LL[1] - LL[0])          # dt/du at u = 0
    s1 = (tt[-1] - tt[-2]) / (LL[-1] - LL[-2])      # dt/du at u = N
    npad = int(np.ceil(4 * h / du))
    k = np.arange(1, npad + 1) * du
    ext = np.r_[tu[0] - s0 * k[::-1], tu, tu[-1] + s1 * k]
    ker = np.exp(-0.5 * (np.arange(-npad, npad + 1) * du / h) ** 2)
    ts = np.convolve(ext, ker / ker.sum(), mode="valid")
    Ls = np.interp(t, ts, u, left=0.0, right=N)
    Ls = Ls - Ls[0]
    return Ls * (N / Ls[-1]) if Ls[-1] > 0 else Lam


def _clustering_stats(counts, Lam, rng, n_jitter=4):
    """CV, burstiness B and local variation LV of rescaled waiting times.

    Events in interval (t_k, t_k+1] are placed uniformly at random in
    rescaled time between Lambda(t_k) and Lambda(t_k+1).  Given the count
    in a bin, the event times of an inhomogeneous Poisson process are iid
    with density proportional to lambda, i.e. UNIFORM in rescaled time, so
    this reconstructs the continuous process exactly under H0 and removes
    the zero waiting times that 0.5-tau sampling would otherwise create.
    (Placing all events of a bin at the bin edge made the test strongly
    conservative: 1 percent rejections at nominal 5.)  Statistics are
    averaged over `n_jitter` placements; the same is done for surrogates.

      CV = std(w)/mean(w)                     Poisson: 1, bursty: > 1
      B  = (CV-1)/(CV+1)  (Goh & Barabasi 2008)   Poisson: 0
      LV = 3/(n-1) sum ((w_i-w_i+1)/(w_i+w_i+1))^2  (Shinomoto 2003)
           Poisson: 1 for ANY slowly varying rate, so it is a cross-check
           that does not lean on the rate estimate.
    """
    if int(counts.sum()) < 6:
        return dict(cv=float("nan"), B=float("nan"), lv=float("nan"))
    lo = np.repeat(Lam[:-1], counts)
    hi = np.repeat(Lam[1:], counts)
    cv, lv = 0.0, 0.0
    for _ in range(n_jitter):
        u = np.sort(lo + (hi - lo) * rng.random(len(lo)))
        w = np.diff(u)
        cv += w.std() / w.mean()
        s = w[1:] + w[:-1]
        lv += 3.0 * np.mean(((w[1:] - w[:-1]) / np.where(s > 0, s, 1)) ** 2)
    cv, lv = cv / n_jitter, lv / n_jitter
    return dict(cv=float(cv), B=float((cv - 1) / (cv + 1)), lv=float(lv))


def waiting_time_test(recs, n_surr=199, seed=0, rate="lcm", med_win=3,
                      n_jitter=4, min_pairs=6, smooth_events=8.0):
    """Annihilation-clustering test of ONE run against a matched Poisson null.

    `recs` are decay-phase timeseries records with keys 't' and 'nv'.
    Returns the observed CV, B and LV, their bootstrap null means, and
    one-sided p-values: p_cv (clustering, CV high; the headline),
    p_cv_low (regularity, CV low) and p_lv (LV high).  Runs with fewer
    than `min_pairs` annihilations return p = nan, note='too few events'.
    Cost: n_surr x O(n): about 1 s for 199 surrogates of 400 samples.
    """
    t = np.array([r["t"] for r in recs], dtype=float)
    nv = np.array([r["nv"] for r in recs], dtype=float)
    nan = float("nan")
    base = dict(rate=rate, med_win=int(med_win), n_surr=int(n_surr),
                n_jitter=int(n_jitter), smooth_events=float(smooth_events))
    empty = dict(cv=nan, B=nan, lv=nan, p_cv=nan, p_cv_low=nan, p_lv=nan)
    if len(t) < 8:
        return dict(base, n_pairs=0, note="too few samples", **empty)
    counts, info = annihilation_events(t, nv, med_win)
    base.update(info)
    N = info["n_pairs"]
    if N < min_pairs:
        return dict(base, note="too few events", **empty)
    rng = np.random.default_rng(seed)
    Lam = _cum_rate(t, counts, rate)
    obs = _clustering_stats(counts, Lam, rng, n_jitter)
    Lgen = (_smooth_cum_rate(t, Lam, smooth_events) if rate == "lcm"
            else Lam)
    prob = np.clip(np.diff(Lgen), 0.0, None)
    prob = prob / prob.sum()
    sc, sl = np.empty(n_surr), np.empty(n_surr)
    for i in range(n_surr):
        cs = rng.multinomial(N, prob)
        st = _clustering_stats(cs, _cum_rate(t, cs, rate), rng, n_jitter)
        sc[i], sl[i] = st["cv"], st["lv"]
    P = lambda k: float((1 + k) / (1 + n_surr))
    return dict(base, note="", cv=obs["cv"], B=obs["B"], lv=obs["lv"],
                cv_null_mean=float(sc.mean()), cv_null_std=float(sc.std()),
                lv_null_mean=float(sl.mean()),
                p_cv=P(np.sum(sc >= obs["cv"])),
                p_cv_low=P(np.sum(sc <= obs["cv"])),
                p_lv=P(np.sum(sl >= obs["lv"])))


def binom_tail(k, n, p=0.05):
    """P(X >= k) for X ~ Binomial(n, p): chance of k or more 'hits'."""
    from math import comb
    return float(sum(comb(n, j) * p ** j * (1 - p) ** (n - j)
                     for j in range(k, n + 1)))


def fisher_combined(ps):
    """Fisher's combined p over independent runs (nan entries ignored)."""
    from scipy.stats import chi2
    ps = np.array([p for p in ps if p is not None and np.isfinite(p)])
    if len(ps) == 0:
        return float("nan"), 0
    return float(chi2.sf(-2 * np.sum(np.log(ps)), 2 * len(ps))), len(ps)


# ------------------------------------------------- resume key (fix F2)
# Every input that changes the PHYSICS or the MEASUREMENT of a point.
# BUG FIX (v2.6, finding F2): completed points used to be matched by the
# title "trackA Ma%.2f seed%d eps%.3f" alone, which ignores cells, T_stir,
# T_decay, dx, l_z, n0_as3, V0_factor and n_stir.  An 8-cell run therefore
# counted as "done" for the 12-cell scan (two rows of the 24-point table
# came from an 8-cell box), and a --T_decay 600 re-run would have skipped
# every point.  Points are now matched on all of these.
KEY_FIELDS = ("eps_dd", "l_z", "n0_as3", "Ma", "seed", "cells", "dx",
              "T_stir", "T_decay", "sample_every", "n_stir", "V0_factor",
              "count_version", "persist", "persist_radius")
# v2.8: the last three change the MEASUREMENT (which vortex counts are
# recorded), not the physics.  Runs from <= v2.7 lack them and are read
# with LEGACY values (count_version 1 = masked count only, no filter).
LEGACY = {"sample_every": 0.5, "count_version": 1, "persist": 1,
          "persist_radius": 0.0}
COUNT_FIELDS = ("count_version", "persist", "persist_radius")


def resolved_n_stir(args):
    return args.n_stir if args.n_stir > 0 else max(1, args.cells // 3)


def point_params(args, Ma, seed):
    """The physical-parameter set that identifies one scan point."""
    return dict(eps_dd=float(args.eps_dd), l_z=float(args.l_z),
                n0_as3=float(args.n0_as3), Ma=float(Ma), seed=int(seed),
                cells=int(args.cells), dx=float(args.dx),
                T_stir=float(args.T_stir), T_decay=float(args.T_decay),
                sample_every=float(args.sample_every),
                n_stir=int(resolved_n_stir(args)),
                V0_factor=float(args.V0_factor),
                count_version=int(args.count_version
                                  if args.count_version is not None else 3),
                persist=int(args.persist),
                persist_radius=float(args.persist_radius))


def param_key(pp):
    """Stable 10-hex-digit hash of a point_params dict (floats rounded)."""
    norm = {k: (round(float(pp[k]), 9) if isinstance(pp[k], float)
                else pp[k]) for k in KEY_FIELDS}
    blob = json.dumps(norm, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def run_title(pp):
    """Human-readable run title.  Includes the box and decay time so the
    folder name alone no longer hides an 8-cell or long-decay run."""
    return ("trackA Ma%.2f seed%d eps%.3f c%d Td%g k%s"
            % (pp["Ma"], pp["seed"], pp["eps_dd"], pp["cells"],
               pp["T_decay"], param_key(pp)))


def stored_matches(stored, want, legacy_defaults=None, ignore=()):
    """True if a run's STORED params equal the wanted point params.

    Runs made before v2.6 did not store dx or sample_every.  dx is then
    recovered from the stored box/grid (box/N, which differs from the
    requested dx by rounding, so a 5 percent tolerance is used); a missing
    sample_every is taken from `legacy_defaults` (the v2.5 default 0.5).
    Any other missing field means "cannot verify" -> no match, so the point
    is re-run rather than wrongly skipped.  Fields in `ignore` are not
    compared (--report uses this to show runs of every count version).
    """
    legacy_defaults = legacy_defaults or LEGACY
    if not stored or stored.get("track") != "A":
        return False
    for k in KEY_FIELDS:
        if k in ignore:
            continue
        v = stored.get(k)
        if v is None and k == "dx" and stored.get("box") and stored.get("grid"):
            v = float(stored["box"][0]) / float(stored["grid"][0])
            if abs(v - want["dx"]) > 0.05 * want["dx"]:
                return False
            continue
        if v is None:
            v = legacy_defaults.get(k)
        if v is None:
            return False
        if isinstance(want[k], float):
            if abs(float(v) - want[k]) > 1e-9 * max(1.0, abs(want[k])):
                return False
        elif int(v) != int(want[k]):
            return False
    return True


# --------------------------------------------------------------------- run
def run_point(archive, args, scales, Ma, seed):
    k_rot, lam, d, min_inside = scales
    pp = point_params(args, Ma, seed)
    tag = run_title(pp)

    grid = build_grid(d, args.cells, args.dx, backend=args.backend)
    area = float(np.prod(grid.lengths))
    gamma = gamma_tilde(args.eps_dd, args.n0_as3)
    Dk = quasi2d_dipolar_symbol(grid, args.l_z)

    solver = EGPESolver(grid, eps_dd=args.eps_dd, gamma=gamma, Dk=Dk)
    dt = min(0.005, solver.suggested_dt() * 0.6)

    params = dict(track="A", hypothesis="H4prime", **pp)
    params.update(param_key=param_key(pp), grid=list(grid.shape),
                  box=[float(v) for v in grid.lengths], dt=dt,
                  k_rot=k_rot, lambda_roton=lam, d_droplet=d,
                  min_inside=min_inside, backend=args.backend)
    run = archive.new_run(tag, params=params)

    # ---- ground state: lattice seed, then L-BFGS polish -------------------
    Nt = float(grid.integrate(np.ones(grid.shape)))
    solver.psi = lattice_seed(grid, d, args.cells).astype(complex)
    solver.psi *= np.sqrt(Nt / solver.norm())
    solver.step_imag(0.002, 1500, norm_target=Nt)
    info = minimize_energy(solver, Nt, maxiter=1500, verbose=False)
    n = grid.xp.abs(solver.psi) ** 2
    contrast = float((n.max() - n.min()) / n.mean())
    print("    ground state: mu=%.5f residual=%.2e contrast=%.3f"
          % (info["mu"], info["residual"], contrast), flush=True)

    # Phase noise so different seeds explore different tangles.
    #
    # BUG FIX (v2.5): rng.standard_normal() always returns a plain host
    # NumPy array regardless of backend (NumPy's RNG has no CuPy
    # equivalent used here), and the old code multiplied `solver.psi`
    # (a full CuPy array on backend="gpu") in place by that host array --
    # the same host/device mixing error as lattice_seed and make_stirrers
    # above.  Generate the noise on the host (cheap, one-time, RNG
    # semantics must stay host-side for reproducibility anyway) then move
    # it to the grid's backend in one transfer before multiplying.
    rng = np.random.default_rng(seed)
    noise_host = np.exp(1j * 0.02 * rng.standard_normal(grid.shape))
    solver.psi = solver.psi * grid.asarray(noise_host)
    solver.psi = solver.psi * np.sqrt(Nt / solver.norm())

    n_stir = resolved_n_stir(args)
    V_stir, V0, radius = make_stirrers(grid, info["mu"], Ma, n_stir=n_stir,
                                       V0_factor=args.V0_factor)
    recs = []
    every = max(1, int(round(args.sample_every / dt)))
    tracker = make_tracker(args, grid)
    n_dec = len(range(0, int(args.T_decay / dt), every))   # decay samples
    S = max(0, int(args.snapshots))
    snap_idx = set()
    if S:
        for start in (0, n_dec // 2, n_dec - S):
            snap_idx.update(range(max(0, start), min(n_dec, start + S)))

    # ---- stirring --------------------------------------------------------
    t = 0.0
    for k in range(int(args.T_stir / dt)):
        solver.V = V_stir(t + 0.5 * dt)
        solver.step_real(dt)
        t += dt
        if k % every == 0:
            r = sample(solver, grid, t, d, area, tracker)
            r["phase"] = "stir"
            recs.append(r)

    # BUG FIX (v2.5): same host/device mixing error -- solver.V feeds
    # directly into _W(n) = self.V + n + ... where n is device-resident,
    # so a plain host np.zeros() here raised the same TypeError on gpu.
    solver.V = grid.xp.zeros(grid.shape)
    E0, N0 = solver.energy()["total"], solver.norm()
    r = sample(solver, grid, t, d, area, tracker)
    r["phase"] = "decay_start"
    recs.append(r)
    print("    end of stirring: nv=%d (raw %d, %.0f%% rejected) R=%.3f"
          "  mask imbalance %.1f%%%s"
          % (r["nv"], r["nv_raw"], 100 * r["mask_reject"], r["R"],
             100 * r["mask_imbalance"],
             "" if r["mask_trustworthy"] else "  [MASK SUSPECT]"),
          flush=True)

    # ---- free decay ------------------------------------------------------
    i_dec = 0
    for k in range(int(args.T_decay / dt)):
        solver.step_real(dt)
        t += dt
        if k % every == 0:
            rr = sample(solver, grid, t, d, area, tracker)
            rr["phase"] = "decay"
            recs.append(rr)
            if i_dec in snap_idx:
                run.save_checkpoint(solver.psi.astype(np.complex64),
                                    step=int(round(t / dt)), t=t,
                                    extra=dict(tag="snapshot", i_decay=i_dec),
                                    verbose=False)
            i_dec += 1

    dE = abs(solver.energy()["total"] - E0) / abs(E0)
    dN = abs(solver.norm() - N0) / N0
    conservative = bool(dE < 1e-4 and dN < 1e-9)

    decay = [r for r in recs if r["phase"] == "decay"]
    av = {}
    for theta in (2.0, 3.0, 4.0):
        ev, st = detect_avalanches(decay, theta)
        av["theta_%.1f" % theta] = dict(stats=st, events=ev)
    # v2.8: the same detector on the persistence-filtered L_p
    decay_p = [dict(t=x["t"], L=x["L_lat"]) for x in decay if "L_lat" in x]
    av_p = {}
    for theta in (2.0, 3.0, 4.0):
        ev, st = detect_avalanches(decay_p, theta)
        av_p["theta_%.1f" % theta] = dict(stats=st, events=ev)

    def window(key, first, span=5.0):
        """Mean of `key` over the first / last `span` tau of the decay."""
        xs = [(q["t"], q[key]) for q in decay if key in q
              and q[key] is not None and np.isfinite(q[key])]
        if not xs:
            return None
        t0, t1 = xs[0][0], xs[-1][0]
        sel = [v for tt, v in xs
               if (tt <= t0 + span if first else tt >= t1 - span)]
        return float(np.mean(sel)) if sel else None

    def slip_rate():
        """Mean |change| of nv_lat per tau between samples -- the rate of
        winding changes (phase slips), not of net decay."""
        x = np.array([float(q["nv_lat"]) for q in decay if "nv_lat" in q])
        if len(x) < 2:
            return None
        return float(np.mean(np.abs(np.diff(x))) / args.sample_every)

    def upfrac(key):
        x = np.diff([float(q[key]) for q in decay if key in q])
        nz = x[x != 0]
        return float(np.mean(nz > 0)) if len(nz) else 0.0

    Rs = [r["R"] for r in decay if np.isfinite(r["R"])]
    summary = dict(
        Ma=Ma, seed=seed, cells=args.cells, T_stir=args.T_stir,
        T_decay=args.T_decay, param_key=param_key(pp), contrast=contrast, mu=info["mu"],
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
        avalanches_p={k: v["stats"] for k, v in av_p.items()},
        nv_p_at_decay_start=r.get("nv_p"), R_p_at_decay_start=r.get("R_p"),
        nv_lat_at_decay_start=r.get("nv_lat"),
        R_lat_at_decay_start=r.get("R_lat"),
        n_droplets_at_decay_start=r.get("n_droplets"),
        upstep_frac_nv=upfrac("nv"), upstep_frac_nv_p=upfrac("nv_p"),
        upstep_frac_nv_lat=upfrac("nv_lat"),
        nv_lat_first5=window("nv_lat", True), nv_lat_last5=window("nv_lat", False),
        R_lat_m_first5=window("R_lat_m", True),
        R_lat_m_last5=window("R_lat_m", False),
        d_meas_first5=window("d_meas", True),
        coh_cos_first5=window("coh_cos", True),
        coh_cos_last5=window("coh_cos", False),
        slip_rate=slip_rate(),
        count_version=pp["count_version"], persist=pp["persist"],
        persist_radius=pp["persist_radius"], snapshots=S,
        d_droplet=d, lambda_roton=lam)

    run.save_diagnostics(recs, name="timeseries", verbose=False)
    run.save_diagnostics(av, name="avalanches", verbose=False)
    run.save_diagnostics(av_p, name="avalanches_p", verbose=False)
    run.save_diagnostics([summary], name="summary", verbose=False)
    run.save_checkpoint(solver.psi, step=int(t / dt), t=t, verbose=False)
    run.finish("completed" if conservative else "completed_unconservative",
               verbose=False)

    flag = "" if conservative else "   [NOT CONSERVATIVE -- distrust]"
    print("    decay: R %.3f -> %.3f | events(3s)=%d | dE/E=%.1e%s"
          % (summary["R_at_decay_start"],
             summary["R_decay_min"] or float("nan"),
             av["theta_3.0"]["stats"]["n_events"], dE, flag), flush=True)
    print("    counts: nv %s -> %s (up-steps %.0f%%) | nv_lat %s -> %s "
          "(up-steps %.0f%%, droplets %s) | events_lat(3s)=%d%s"
          % (decay[0]["nv"], decay[-1]["nv"], 100 * summary["upstep_frac_nv"],
             decay[0].get("nv_lat"), decay[-1].get("nv_lat"),
             100 * summary["upstep_frac_nv_lat"], decay[0].get("n_droplets"),
             av_p["theta_3.0"]["stats"].get("n_events", 0),
             "  | %d snapshots saved" % len(snap_idx) if S else ""),
          flush=True)
    f = lambda v, fmt: (fmt % v) if v is not None else "n/a"
    print("    lattice: nv_lat(5tau avg) %s -> %s | R_lat_m %s -> %s "
          "(d_meas %s xi vs roton d %.2f) | coherence <cos dphi> %s -> %s "
          "| slip rate %s /tau"
          % (f(summary["nv_lat_first5"], "%.1f"), f(summary["nv_lat_last5"], "%.1f"),
             f(summary["R_lat_m_first5"], "%.3f"), f(summary["R_lat_m_last5"], "%.3f"),
             f(summary["d_meas_first5"], "%.2f"), d,
             f(summary["coh_cos_first5"], "%.3f"), f(summary["coh_cos_last5"], "%.3f"),
             f(summary["slip_rate"], "%.1f")), flush=True)
    return summary


# ------------------------------------------------------------ report
def load_trackA_runs(project_dir):
    """Every Track A run folder with a finished summary.

    Matches folder names case-INSENSITIVELY: Archive slugs are lowercase
    ('tracka-...'), and a '*trackA*' glob found nothing on Drive (Linux
    paths are case-sensitive).  Identity comes from RUN_INFO.json params,
    never from the folder name.
    """
    out = []
    for d in sorted(glob.glob(os.path.join(project_dir, "*"))):
        if "tracka" not in os.path.basename(d).lower():
            continue
        f_sum = os.path.join(d, "diagnostics", "summary.json")
        f_info = os.path.join(d, "RUN_INFO.json")
        if not (os.path.isfile(f_sum) and os.path.isfile(f_info)):
            continue
        try:
            info = json.load(open(f_info))
            summ = json.load(open(f_sum))[0]
        except Exception as exc:
            print("  !! unreadable run %s: %s" % (os.path.basename(d), exc))
            continue
        out.append(dict(dir=d, info=info, params=info.get("parameters", {}),
                        summary=summ))
    return out


def implied_cells(nv, R):
    """cells implied by R^2 nv = cells^2 sqrt(3)/2 (exact for this box)."""
    if not nv or not R or not np.isfinite(R):
        return float("nan")
    return float(np.sqrt(R * R * nv / (np.sqrt(3) / 2)))


def summarise_waiting_time(good):
    """Ensemble summary of the matched-null test (v2.7).

    For each p column: how many runs have p < 0.05, the binomial chance of
    at least that many under H0, and Fisher's combined p, for all trusted
    runs and split at R = 1.  Runs with too few annihilations are left out
    and counted.
    """
    print()
    print("  MATCHED NULL (annihilation waiting times vs inhomogeneous "
          "Poisson, own rate):")
    for col, name in (("p_wt_cv", "CV high (clustered)"),
                      ("p_wt_lv", "LV high (cross-check)"),
                      ("p_wt_cv_low", "CV low (regular)")):
        for grp, sel in (("all   ", good),
                         ("R > 1 ", [x for x in good
                                     if x["R_at_decay_start"] > 1]),
                         ("R <= 1", [x for x in good
                                     if x["R_at_decay_start"] <= 1])):
            ps = [x.get(col) for x in sel]
            ps = [p for p in ps if p is not None and np.isfinite(p)]
            if not ps:
                continue
            k = sum(p < 0.05 for p in ps)
            fp, _ = fisher_combined(ps)
            print("    %-22s %s: p<0.05 in %2d of %2d (chance P(>=%d) = %.2f)"
                  "  Fisher p = %.3g"
                  % (name, grp, k, len(ps), k, binom_tail(k, len(ps)), fp))
    few = [x for x in good if x.get("wt_note")]
    if few:
        print("    %d run(s) excluded from the matched test: %s"
              % (len(few), ", ".join(sorted({x["wt_note"] for x in few}))))
    cvs = [x["wt_cv"] for x in good
           if x.get("wt_cv") is not None and np.isfinite(x["wt_cv"])]
    if cvs:
        print("    mean CV = %.3f  (Poisson 1.0; per-run null mean is in the CSV)"
              % np.mean(cvs))


def report(archive, args):
    """Table of finished runs whose STORED params match the current settings.

    Filters on every KEY_FIELDS entry except Ma and seed.  Rows whose
    R^2*nv disagrees with the stored cell count are flagged, which catches
    any future box mix-up (finding F1).  With --surrogates N, runs the
    formal null test on each run's saved timeseries.json.
    """
    runs = load_trackA_runs(archive.project_dir)
    print("Track A run folders with a summary: %d" % len(runs))
    rows, rejected = {}, {}
    for r in runs:
        p = r["params"]
        Ma, seed = p.get("Ma"), p.get("seed")
        if Ma is None or seed is None:
            continue
        want = point_params(args, Ma, seed)
        ign = COUNT_FIELDS if args.count_version is None else ()
        if not stored_matches(p, want, ignore=ign):
            why = ", ".join("%s=%s" % (k, p.get(k)) for k in
                            ("cells", "T_stir", "T_decay", "V0_factor")
                            if p.get(k) is not None and k in want
                            and float(p.get(k)) != float(want[k]))
            rejected.setdefault(why or "other params differ", []).append(
                os.path.basename(r["dir"]))
            continue
        k = (round(float(Ma), 4), int(seed))
        if k in rows:                       # duplicate: keep the newest
            print("  note: duplicate point Ma=%.2f seed=%d -> keeping %s"
                  % (k[0], k[1], os.path.basename(r["dir"])))
        rows[k] = r
    for why, ds in rejected.items():
        print("  excluded %d run(s) with different params (%s)"
              % (len(ds), why))
    print("Matching cells=%d T_stir=%g T_decay=%g: %d point(s)\n"
          % (args.cells, args.T_stir, args.T_decay, len(rows)))

    hdr = ("%5s %4s %6s %7s %4s %4s %4s %4s %6s"
           % ("Ma", "seed", "nv", "R", "ev2", "ev3", "ev4", "cons", "cellsR"))
    if args.surrogates:
        hdr += "  p_phase(3)  p_iaaft(3)  pairs    CV  p_wt   p_LV count"
    print(hdr)
    table = []
    for (Ma, seed), r in sorted(rows.items()):
        s = r["summary"]
        a = s["avalanches"]
        ic = implied_cells(s["nv_at_decay_start"], s["R_at_decay_start"])
        cells_ok = abs(ic - args.cells) < 0.05 * args.cells
        row = dict(Ma=Ma, seed=seed, nv_at_decay_start=s["nv_at_decay_start"],
                   R_at_decay_start=round(s["R_at_decay_start"], 4),
                   R_decay_min=s.get("R_decay_min"),
                   events_theta2=a["theta_2.0"]["n_events"],
                   events_theta3=a["theta_3.0"]["n_events"],
                   events_theta4=a["theta_4.0"]["n_events"],
                   conservative=s["conservative"],
                   mask_trustworthy=s.get("mask_trustworthy"),
                   implied_cells=round(ic, 2), cells_consistent=cells_ok,
                   run_id=os.path.basename(r["dir"]))
        line = ("%5.2f %4d %6d %7.3f %4d %4d %4d %4s %6.2f%s"
                % (Ma, seed, row["nv_at_decay_start"], row["R_at_decay_start"],
                   row["events_theta2"], row["events_theta3"],
                   row["events_theta4"], "T" if row["conservative"] else "F",
                   ic, "" if cells_ok else " !!BOX"))
        if args.surrogates:
            f_ts = os.path.join(r["dir"], "diagnostics", "timeseries.json")
            try:
                ts = [x for x in json.load(open(f_ts))
                      if x.get("phase") == "decay"]
                # v2.8: use the persistence-filtered count when recorded
                if ts and "nv_lat" in ts[0]:
                    ts = [dict(t=x["t"], L=x["L_lat"], nv=x["nv_lat"])
                          for x in ts]
                    row["count"] = "nv_lat"
                else:
                    row["count"] = "nv"
                st = surrogate_test(ts, n_surr=args.surrogates, seed=seed)
                for m in ("phase", "iaaft"):
                    for th in ("2.0", "3.0", "4.0"):
                        row["p_%s_theta%s" % (m, th[0])] = \
                            st[m]["theta_" + th]["p"]
                line += "  %10.3f  %10.3f" % (row["p_phase_theta3"],
                                              row["p_iaaft_theta3"])
                # v2.7: matched null -- annihilation waiting times against
                # an inhomogeneous Poisson process with the run's own rate
                wt = waiting_time_test(ts, n_surr=args.surrogates, seed=seed)
                row.update(n_pairs=wt["n_pairs"], flicker=wt.get("flicker"),
                           wt_cv=wt["cv"], wt_B=wt["B"], wt_lv=wt["lv"],
                           wt_cv_null=wt.get("cv_null_mean"),
                           p_wt_cv=wt["p_cv"], p_wt_cv_low=wt["p_cv_low"],
                           p_wt_lv=wt["p_lv"], wt_note=wt["note"])
                line += "  %5d %5.2f %5.3f %6.3f %s%s" % (
                    wt["n_pairs"], wt["cv"], wt["p_cv"], wt["p_lv"],
                    row["count"],
                    "" if not wt["note"] else "  (" + wt["note"] + ")")
            except Exception as exc:
                line += "  (surrogate test failed: %s)" % exc
        print(line)
        table.append(row)

    if table:
        import csv
        out = os.path.join(archive.project_dir,
                           "trackA_results_c%d_Td%g.csv"
                           % (args.cells, args.T_decay))
        keys = list(dict.fromkeys(k for row in table for k in row))
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(table)
        print("\nwrote %s" % out)

        # grouped means, R > 1 versus R <= 1, trusted rows only
        good = [x for x in table if x["conservative"] and x["cells_consistent"]]
        for name, sel in (("R > 1 ", [x for x in good
                                      if x["R_at_decay_start"] > 1]),
                          ("R <= 1", [x for x in good
                                      if x["R_at_decay_start"] <= 1])):
            if sel:
                print("  %s: %2d runs  mean events th2=%.2f th3=%.2f th4=%.2f"
                      % (name, len(sel),
                         np.mean([x["events_theta2"] for x in sel]),
                         np.mean([x["events_theta3"] for x in sel]),
                         np.mean([x["events_theta4"] for x in sel])))
        near = [x for x in good if 0.8 <= x["R_at_decay_start"] <= 1.3]
        print("  runs with R in [0.8, 1.3] (the crossover): %d" % len(near))
        if args.surrogates:
            sig = [x for x in good if x.get("p_phase_theta3", 1) < 0.05]
            print("  runs with p_phase(theta=3) < 0.05: %d of %d "
                  "(about %.1f expected by chance)"
                  % (len(sig), len(good), 0.05 * len(good)))
            summarise_waiting_time(good)
    return table


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

    from qtsim.drive_io import Archive
    if args.outdir:
        archive = Archive(root=os.path.abspath(args.outdir), verbose=False)
    else:
        archive = Archive(verbose=False)

    if args.report:
        report(archive, args)
        return

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
    print("  obstacles: %d (scaled from cells)" % resolved_n_stir(args))
    print("  The scan MUST bracket R=1.  Vortex yield depends on --drives,")
    print("  --V0_factor, --n_stir AND --cells together, so a drive list")
    print("  calibrated for one box size will NOT transfer to another.")
    print("  Run a 2-point probe first (e.g. --drives 0.5 2.0 --seeds 0")
    print("  with short times) and check that R straddles 1 before")
    print("  committing to the full scan.")
    print()

    finished = [r for r in archive.load_index().get("runs", [])
                if r.get("status", "").startswith("completed")]
    results, t0 = [], time.time()
    for Ma in args.drives:
        for seed in args.seeds:
            pp = point_params(args, Ma, seed)
            tag = run_title(pp)
            hit = next((r for r in finished
                        if stored_matches(r.get("parameters"), pp)), None)
            if hit is not None:
                print("  [skip, already completed with identical params] "
                      "%s  (%s)" % (tag, hit.get("run_id")), flush=True)
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
