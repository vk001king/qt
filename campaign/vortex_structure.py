"""v2.12: structure of the vortex population in saved snapshots.

For each run folder and each snapshot (psi_start / psi_mid / psi_end):
  * all phase singularities from the plaquette detector (raw count)
  * nearest-neighbour distance r_nn (periodic) and whether the nearest
    neighbour has the opposite sign
  * fraction of vortices in close opposite-sign pairs, r_nn < a/2
    (a = lattice constant), compared with the same statistic for the same
    number of charges at uniformly random positions with random signs
  * density modulation std(n)/mean(n) (crystal: ~1; uniform fluid: 0)
  * density at the core location, relative to the mean density, smoothed
    over 1.5 xi (where the vortex sits: droplet or interstitial region)

Usage: vortex_structure.py run_folder [...] --out file.json
"""
import os, sys, json, argparse
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from qtsim.diagnostics import plaquette_charges_2d
import track_a_scan as ta


def stats(xy, q, L, rpair, rng=None):
    if len(q) < 2:
        return dict(n=int(len(q)), frac_pair=float("nan"), frac_opp_nn=float("nan"), rnn_median=float("nan"),
                    n_free_2xi=int(len(q)), n_free_half_a=int(len(q)), ropp_median=float("nan"))
    tr = cKDTree(np.mod(xy, L), boxsize=L)
    d, i = tr.query(np.mod(xy, L), k=2)
    rnn, j = d[:, 1], i[:, 1]
    opp = q[j] != q
    # distance to the nearest opposite-sign charge
    ropp = np.full(len(q), np.inf)
    for sgn in (1, -1):
        a_, b_ = q == sgn, q == -sgn
        if a_.any() and b_.any():
            ropp[a_] = cKDTree(np.mod(xy[b_], L), boxsize=L).query(np.mod(xy[a_], L))[0]
    return dict(n=int(len(q)), frac_pair=float(np.mean(opp & (rnn < rpair))),
                n_free_2xi=int(np.sum(ropp >= 2.0)), n_free_half_a=int(np.sum(ropp >= rpair)),
                ropp_median=float(np.median(ropp)),
                frac_opp_nn=float(np.mean(opp)), rnn_median=float(np.median(rnn)))


def analyse(folder, rng):
    sm = json.load(open(os.path.join(folder, "summary.json")))
    meta = json.load(open(sm["ground"] + ".json"))
    grid = ta.build_grid(meta["lattice"], meta["cells"], meta["dx"])
    L = np.array(grid.lengths)
    a = meta["lattice"]
    out = dict(tag=sm["tag"], lattice=a, rpair=0.5 * a)
    ts = [r for r in json.load(open(os.path.join(folder, "timeseries.json"))) if r["phase"] in ("decay_start", "decay")]
    tdec = np.array([r["t"] for r in ts]) - ts[0]["t"]
    nlat = np.array([r["N_lat"] for r in ts]); raw = np.array([r["raw"] for r in ts])
    T = sm["T_decay"]; tsnap = {"start": 0.0, "mid": T / 2, "end": T}
    for snap in ("start", "mid", "end"):
        f = os.path.join(folder, "psi_%s.npy" % snap)
        if not os.path.exists(f):
            continue
        psi = np.load(f).astype(complex)
        q = plaquette_charges_2d(psi)
        ix, iy = np.nonzero(q)
        qq = q[ix, iy]
        xy = np.stack([(ix + 0.5) * grid.dx[0], (iy + 0.5) * grid.dx[1]], axis=1)
        n = gaussian_filter(np.abs(psi) ** 2, 1.5 / grid.dx[0], mode="wrap")
        ncore = n[ix, iy] / n.mean() if len(qq) else np.array([])
        st = stats(xy, qq, L, 0.5 * a)
        nn = np.abs(psi) ** 2
        st.update(modulation=float(nn.std() / nn.mean()))
        # random reference with the same number of charges and signs
        ref = [stats(rng.uniform(0, 1, (len(qq), 2)) * L, rng.permutation(qq), L, 0.5 * a)["frac_pair"]
               for _ in range(200)] if len(qq) >= 2 else [float("nan")]
        st.update(net_charge=int(qq.sum()), frac_pair_random=float(np.nanmean(ref)),
                  frac_pair_random_sd=float(np.nanstd(ref)),
                  ncore_median=float(np.median(ncore)) if len(qq) else float("nan"),
                  frac_in_droplet=float(np.mean(ncore > 1.0)) if len(qq) else float("nan"))
        k = int(np.argmin(np.abs(tdec - tsnap[snap])))
        st.update(N_lat_ts=int(nlat[k]), raw_ts=int(raw[k]), t_snap=float(tsnap[snap]))
        out[snap] = st
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    rows = []
    for f in args.folders:
        rows.append(analyse(f, rng))
        r = rows[-1]
        print(r["tag"], " | ".join("%s: n=%d free(2xi)=%d free(a/2)=%d Nlat=%d pair=%.2f (rand %.2f) ropp=%.2f" %
                                    (s, r[s]["n"], r[s]["n_free_2xi"], r[s]["n_free_half_a"], r[s]["N_lat_ts"],
                                     r[s]["frac_pair"], r[s]["frac_pair_random"], r[s]["ropp_median"])
                                    for s in ("start", "mid", "end") if s in r), flush=True)
    json.dump(rows, open(args.out, "w"), indent=1)
