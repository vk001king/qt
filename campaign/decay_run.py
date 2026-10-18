"""v2.12 decay campaign for the corrected quasi-2D model.

Two protocols, both starting from the production ground state built by
build_ground.py:

  imprint  N_v vortices (neutral, random signs) imprinted at randomly chosen
           interstitial sites of the intact crystal, followed by a damped
           stage (i dpsi/dt = (1 - i g)(H - mu) psi, T_cool) that heals the
           cores and removes imprint sound, then free decay.  Varies vortex
           number at fixed droplet number and order.
  stir     four rotating Gaussian obstacles (as in track_a_scan.py), then the
           same damped stage and free decay.
  phase    energy-matched, vortex-free control: a smooth random phase field
           (Gaussian random field, |k| <= k_rot, zero winding) whose amplitude
           is tuned so that the energy injected equals that of the imprint of
           --match_Nv vortices with the same seed; then the same damped stage
           and free decay.  Separates the effect of quantised circulation from
           that of the injected energy.

Every sample (interval sample_every) records: N_lat, raw plaquette count,
droplet number, bond order |psi6| (local) and global, link coherence,
measured spacing and R, incompressible/compressible kinetic energy, total
energy and norm.  Snapshots of psi are saved at the start, middle and end of
the free decay.  Output: <outdir>/<tag>/{timeseries.json, summary.json, *.npy}
"""
import sys, os, json, time, argparse
import numpy as np
from scipy.spatial import Delaunay
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from qtsim import EGPESolver, quasi2d_dipolar_symbol
from qtsim.diagnostics import (plaquette_charges_2d, droplet_centers, interstitial_charges,
                               link_coherence, measured_spacing, helmholtz_split_2d)
from qtsim.imprint import imprint
import track_a_scan as ta


def psi6(cen, L):
    n0 = len(cen)
    if n0 < 3:
        return float("nan"), float("nan")
    tiles = np.concatenate([cen + L * np.array([a, b]) for a in (-1, 0, 1) for b in (-1, 0, 1)])
    s = Delaunay(tiles).simplices
    e = np.concatenate([s[:, [0, 1]], s[:, [1, 2]], s[:, [2, 0]]])
    e = e[e[:, 0] < n0]
    v = tiles[e[:, 1]] - tiles[e[:, 0]]
    z = np.exp(6j * np.arctan2(v[:, 1], v[:, 0]))
    loc = np.zeros(n0, complex); cnt = np.zeros(n0)
    np.add.at(loc, e[:, 0], z); np.add.at(cnt, e[:, 0], 1)
    return float(np.mean(np.abs(loc / np.maximum(cnt, 1)))), float(np.abs(z.mean()))


def sample(s, grid, lat, area, t, phase):
    psi = s.psi
    n = np.abs(psi) ** 2
    q = plaquette_charges_2d(psi)
    cen = droplet_centers(n, grid, lat)
    w, _ = interstitial_charges(q, cen, grid)
    nl = int(np.abs(w).sum())
    lc = link_coherence(psi, cen, grid)
    dm = measured_spacing(len(cen), area)
    p6l, p6g = psi6(cen, np.array(grid.lengths))
    Ei, Ec, _ = helmholtz_split_2d(psi, grid)
    N = s.norm()
    return dict(t=float(t), phase=phase, N_lat=nl, raw=int((q != 0).sum()),
                n_multi=int((np.abs(w) > 1).sum()), droplets=int(len(cen)),
                psi6=p6l, psi6_global=p6g, coh=lc["cos"], d_meas=dm,
                R=float(np.sqrt(area / nl) / dm) if nl > 0 else float("inf"),
                Ei=float(Ei / N), Ec=float(Ec / N), E=float(s.energy()["total"] / N), norm=float(N))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground", default="/home/claude/runs2/gs_dx05")
    ap.add_argument("--protocol", choices=("imprint", "stir", "phase"), default="imprint")
    ap.add_argument("--Nv", type=int, default=160)
    ap.add_argument("--Ma", type=float, default=1.0)
    ap.add_argument("--match_Nv", type=int, default=80,
                    help="phase protocol: inject the same energy as imprinting this many vortices")
    ap.add_argument("--T_stir", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T_cool", type=float, default=5.0)
    ap.add_argument("--gamma_cool", type=float, default=0.03)
    ap.add_argument("--T_decay", type=float, default=200.0)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--sample_every", type=float, default=0.5)
    ap.add_argument("--outdir", default="/home/claude/runs2/decay")
    ap.add_argument("--sites_ground", default=None,
                    help="take interstitial sites from this ground state (same box), "
                         "so that runs on different grids start from the same vortex positions")
    a = ap.parse_args()

    meta = json.load(open(a.ground + ".json"))
    psi0 = np.load(a.ground + ".npy")
    lat = meta["lattice"]
    grid = ta.build_grid(lat, meta["cells"], meta["dx"])
    assert list(grid.shape) == list(psi0.shape)
    area = float(np.prod(grid.lengths))
    L = np.array(grid.lengths)
    s = EGPESolver(grid, eps_dd=meta["eps"], gamma=meta["gamma"],
                   Dk=quasi2d_dipolar_symbol(grid, meta["l_z"]))
    s.psi = psi0.astype(complex).copy()
    Nt = s.norm()
    rng = np.random.default_rng(a.seed)
    tag = {"imprint": "imprint_N%d" % a.Nv, "stir": "stir_Ma%.2f" % a.Ma,
           "phase": "phase_mN%d" % a.match_Nv}[a.protocol] \
        + "_s%d_Td%g_dt%g_dx%g" % (a.seed, a.T_decay, a.dt, meta["dx"])
    out = os.path.join(a.outdir, tag)
    os.makedirs(out, exist_ok=True)
    every = max(1, int(round(a.sample_every / a.dt)))
    recs = []
    t0 = time.time()
    t = 0.0
    if a.protocol in ("imprint", "phase"):
        Nimp = a.Nv if a.protocol == "imprint" else a.match_Nv
        if a.sites_ground:
            m2 = json.load(open(a.sites_ground + ".json"))
            g2 = ta.build_grid(m2["lattice"], m2["cells"], m2["dx"])
            cen = droplet_centers(np.abs(np.load(a.sites_ground + ".npy")) ** 2, g2, lat)
        else:
            cen = droplet_centers(np.abs(psi0) ** 2, grid, lat)
        tiles = np.concatenate([cen + L * np.array([u, v]) for u in (-1, 0, 1) for v in (-1, 0, 1)])
        tc = tiles[Delaunay(tiles).simplices].mean(axis=1)
        sites = tc[np.all((tc >= 0) & (tc < L), axis=1)]
        idx = rng.choice(len(sites), Nimp, replace=False)
        xs, ys = (sites[idx] + rng.normal(0, 0.3, (Nimp, 2))).T
        qs = np.array([1] * (Nimp // 2) + [-1] * (Nimp // 2)); rng.shuffle(qs)
        psi_v = imprint(s.psi, grid, xs, ys, qs)
        psi_v *= np.sqrt(Nt / np.sum(np.abs(psi_v) ** 2 * grid.dV))
        if a.protocol == "imprint":
            s.psi = psi_v
        else:
            E_gs = s.energy()["total"]
            s.psi = psi_v
            dE_target = s.energy()["total"] - E_gs
            KX, KY = grid.K
            kr = meta["k_rot"]
            fk = (rng.standard_normal(grid.shape) + 1j * rng.standard_normal(grid.shape)) \
                * (np.hypot(KX, KY) <= kr) * (np.hypot(KX, KY) > 0)
            phi = np.real(np.fft.ifft2(fk)); phi /= phi.std()

            def dE_of(A):
                s.psi = psi0.astype(complex) * np.exp(1j * A * phi)
                return s.energy()["total"] - E_gs
            lo_, hi_ = 0.0, 0.1
            while dE_of(hi_) < dE_target:
                hi_ *= 2
            for _ in range(50):
                mid = 0.5 * (lo_ + hi_)
                lo_, hi_ = (mid, hi_) if dE_of(mid) < dE_target else (lo_, mid)
            A = 0.5 * (lo_ + hi_)
            dE_got = dE_of(A)
            print("phase control: dE_target %.4e  dE %.4e  A %.4f rad" % (dE_target / Nt, dE_got / Nt, A), flush=True)
        recs.append(sample(s, grid, lat, area, t, "imprinted"))
    else:
        V_stir, V0, radius = ta.make_stirrers(grid, meta["states"][0]["mu"], a.Ma, n_stir=max(1, meta["cells"] // 3),
                                              V0_factor=1.5)
        s.psi = s.psi * np.exp(1j * 0.02 * rng.standard_normal(grid.shape))
        for k in range(int(a.T_stir / a.dt)):
            s.V = V_stir(t + 0.5 * a.dt)
            s.step_real(a.dt)
            t += a.dt
            if k % every == 0:
                recs.append(sample(s, grid, lat, area, t, "stir"))
        s.V = np.zeros(grid.shape)
    # damped stage
    for k in range(int(a.T_cool / a.dt)):
        ta.step_damped(s, a.dt, a.gamma_cool, Nt)
        t += a.dt
        if k % every == 0:
            recs.append(sample(s, grid, lat, area, t, "cool"))
    E0 = s.energy()["total"]
    recs.append(sample(s, grid, lat, area, t, "decay_start"))
    nd = int(a.T_decay / a.dt)
    snaps = {0: "start", nd // 2: "mid", nd - 1: "end"}
    np.save(os.path.join(out, "psi_start.npy"), s.psi.astype(np.complex64))
    for k in range(nd):
        s.step_real(a.dt)
        t += a.dt
        if k % every == 0:
            recs.append(sample(s, grid, lat, area, t, "decay"))
        if k in snaps and k > 0:
            np.save(os.path.join(out, "psi_%s.npy" % snaps[k]), s.psi.astype(np.complex64))
    dE = abs(s.energy()["total"] - E0) / abs(E0)
    dN = abs(s.norm() - Nt) / Nt
    json.dump(recs, open(os.path.join(out, "timeseries.json"), "w"))
    extra = dict(phase_amplitude=A, dE_injected=dE_got / Nt, dE_target=dE_target / Nt) \
        if a.protocol == "phase" else {}
    summ = dict(vars(a), **extra, tag=tag, energy_drift=dE, norm_drift=dN, wall_s=time.time() - t0,
                model=dict((k, meta[k]) for k in ("eps", "l_z", "gamma", "lattice", "cells", "dx", "grid", "box")))
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
    dec = [r for r in recs if r["phase"] in ("decay_start", "decay")]
    print("%s | N_lat %d -> %d | psi6 %.2f -> %.2f | coh %.2f -> %.2f | Ec %.3f -> %.3f | dE/E %.1e | %.0f s"
          % (tag, dec[0]["N_lat"], dec[-1]["N_lat"], dec[0]["psi6"], dec[-1]["psi6"], dec[0]["coh"],
             dec[-1]["coh"], dec[0]["Ec"], dec[-1]["Ec"], dE, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
