"""v2.12 analysis of decay_run.py output (melting, decay laws, statistics).

Per run (one folder with timeseries.json + summary.json):
  * melting time t_m: first time the 5-tau running mean of the local bond
    order |psi6| falls below psi6* = (psi6_crystal + psi6_random)/2
  * surviving fraction f(t) = N_lat(t)/N_lat(0) at fixed times
  * decay-law fits to ln N_lat(t) (exponential, Vinen, mixed), N0 free,
    least squares; AIC with the nominal n and with the effective
    n_eff = n / (2 tau_int) from the residual autocorrelation
  * moving-block residual bootstrap (B resamples, block length 2 tau_int):
    95% intervals for all parameters and for Delta AIC, and the fraction of
    resamples in which each model is best
  * sensitivity: fit window (drop first 5 tau; stop when N_lat < 10) and
    sampling (every 2 tau)
  * independent diagnostic: incompressible energy per vortex E_i/N_lat(t)
"""
import os, sys, json, glob, argparse
import numpy as np
from scipy.optimize import least_squares

PSI6_CRYSTAL, PSI6_RANDOM = 0.983, 0.371
PSI6_STAR = 0.5 * (PSI6_CRYSTAL + PSI6_RANDOM)


def load(folder):
    ts = json.load(open(os.path.join(folder, "timeseries.json")))
    sm = json.load(open(os.path.join(folder, "summary.json")))
    dec = [r for r in ts if r["phase"] in ("decay_start", "decay")]
    t0 = dec[0]["t"]
    g = lambda k: np.array([r[k] for r in dec], float)
    return dict(sm=sm, all=ts, t=g("t") - t0, N=g("N_lat"), Nraw=g("raw"), nmulti=g("n_multi"), psi6=g("psi6"), psi6g=g("psi6_global"),
                coh=g("coh"), Ec=g("Ec"), Ei=g("Ei"), R=g("R"), droplets=g("droplets"),
                tabs=g("t"))


def running(x, w=10):
    k = np.ones(w) / w
    y = np.convolve(x, k, mode="valid")
    return np.concatenate([np.full(w // 2, np.nan), y, np.full(len(x) - len(y) - w // 2, np.nan)])


def melting_time(ts):
    """Melting time measured from the moment of imprint/stirring end (all phases)."""
    t = np.array([r["t"] for r in ts]); p = np.array([r["psi6"] for r in ts])
    pm = running(p, 10)
    below = np.where(pm < PSI6_STAR)[0]
    return float(t[below[0]]) if len(below) else float("inf")


# ------------------------------------------------------------------ models
def m_exp(p, t):  # p = (lnN0, a)
    return p[0] - p[1] * t


def m_vin(p, t):  # p = (lnN0, b)
    return p[0] - np.log1p(p[1] * np.exp(p[0]) * t)


def m_mix(p, t):  # p = (lnN0, a, b)
    N0, a, b = np.exp(p[0]), p[1], p[2]
    if a < 1e-12:
        return p[0] - np.log1p(b * N0 * t)
    return np.log(a * N0) - a * t - np.log(a + b * N0 * (1 - np.exp(-a * t)))


MODELS = {"exp": (m_exp, 2), "vinen": (m_vin, 2), "mixed": (m_mix, 3)}


def fit(name, t, y):
    f, k = MODELS[name]
    lo = [-np.inf] + [0.0] * (k - 1)
    best = None
    a0 = max(1e-5, -np.polyfit(t, y, 1)[0])
    starts = {"exp": [[y[0], a0]], "vinen": [[y[0], a0 / np.exp(y[0])], [y[0], 1e-6]],
              "mixed": [[y[0], a0, 1e-7], [y[0], a0 / 2, a0 / 2 / np.exp(y[0])], [y[0], 1e-6, a0 / np.exp(y[0])]]}[name]
    for s0 in starts:
        r = least_squares(lambda p: f(p, t) - y, s0, bounds=(lo, np.inf), method="trf",
                          x_scale="jac", max_nfev=5000)
        if best is None or r.cost < best.cost:
            best = r
    res = y - f(best.x, t)
    return best.x, res


def tau_int(res):
    """Integrated autocorrelation time (in samples) with Sokal windowing."""
    x = res - res.mean()
    n = len(x)
    ac = np.correlate(x, x, "full")[n - 1:] / (x.var() * n)
    tau = 0.5
    for M in range(1, n // 2):
        tau = 0.5 + ac[1:M + 1].sum()
        if M >= 5 * tau:
            break
    return max(0.5, float(tau))


def aic(res, k, n_eff=None):
    n = len(res)
    ne = n if n_eff is None else n_eff
    return ne * np.log(np.mean(res ** 2)) + 2 * k


def select(t, y, B=300, seed=0):
    fits = {m: fit(m, t, y) for m in MODELS}
    resid_best = fits["mixed"][1]
    ti = tau_int(resid_best)
    n_eff = len(y) / (2 * ti)
    A = {m: aic(fits[m][1], MODELS[m][1]) for m in MODELS}
    Ae = {m: aic(fits[m][1], MODELS[m][1], n_eff) for m in MODELS}
    out = dict(n=len(y), tau_int_samples=ti, n_eff=n_eff,
               params={m: list(map(float, fits[m][0])) for m in MODELS},
               rms={m: float(np.sqrt(np.mean(fits[m][1] ** 2))) for m in MODELS},
               dAIC={m: float(A[m] - min(A.values())) for m in MODELS},
               dAIC_eff={m: float(Ae[m] - min(Ae.values())) for m in MODELS})
    if B <= 0:
        return out
    # moving-block residual bootstrap around the mixed-model fit
    rng = np.random.default_rng(seed)
    f_mix = MODELS["mixed"][0](fits["mixed"][0], t)
    blk = max(1, int(round(2 * ti)))
    nb = int(np.ceil(len(y) / blk))
    wins = {m: 0 for m in MODELS}
    pars = {m: [] for m in MODELS}
    dA = {m: [] for m in MODELS}
    for _ in range(B):
        starts = rng.integers(0, len(y) - blk + 1, nb)
        rb = np.concatenate([resid_best[s:s + blk] for s in starts])[:len(y)]
        yb = f_mix + rb
        fb = {m: fit(m, t, yb) for m in MODELS}
        Ab = {m: aic(fb[m][1], MODELS[m][1], n_eff) for m in MODELS}
        mn = min(Ab.values())
        wins[min(Ab, key=Ab.get)] += 1
        for m in MODELS:
            pars[m].append(fb[m][0]); dA[m].append(Ab[m] - mn)
    out["bootstrap"] = dict(B=B, block=blk,
                            best_fraction={m: wins[m] / B for m in MODELS},
                            params_95={m: np.percentile(np.array(pars[m]), [2.5, 97.5], axis=0).tolist() for m in MODELS},
                            dAIC_eff_95={m: np.percentile(dA[m], [2.5, 97.5]).tolist() for m in MODELS})
    return out


def analyse_run(folder, B=300, do_stats=True):
    d = load(folder)
    t, N = d["t"], d["N"]
    ok = N > 0
    r = dict(tag=d["sm"]["tag"], protocol=d["sm"]["protocol"], Nv=d["sm"].get("Nv"),
             Ma=d["sm"].get("Ma"), seed=d["sm"]["seed"], T_decay=d["sm"]["T_decay"],
             energy_drift=d["sm"]["energy_drift"], t_melt=melting_time(d["all"]),
             N0=float(N[:10].mean()), psi6_start=float(d["psi6"][:10].mean()),
             psi6_end=float(d["psi6"][-10:].mean()), coh_start=float(d["coh"][:10].mean()),
             coh_end=float(d["coh"][-10:].mean()), Ec_start=float(d["Ec"][:10].mean()),
             Ec_end=float(d["Ec"][-10:].mean()), R0=float(np.nanmean(d["R"][:10][np.isfinite(d["R"][:10])])),
             droplets0=float(d["droplets"][:10].mean()))
    for T in (25, 50, 100, 200, 400):
        sel = (t >= T - 5) & (t <= T)
        if T <= t[-1] + 1e-9 and sel.any():
            r["f_%d" % T] = float(N[sel].mean() / r["N0"])
    # independent check: E_i per vortex, start vs end windows
    r["Ei_per_vortex_start"] = float(d["Ei"][:10].mean() / max(r["N0"], 1))
    r["Ei_per_vortex_end"] = float(d["Ei"][-10:].mean() / max(N[-10:].mean(), 1))
    r["corr_lnEi_lnN"] = float(np.corrcoef(np.log(d["Ei"][ok]), np.log(N[ok]))[0, 1]) if ok.sum() > 10 else None
    if do_stats and ok.sum() > 50:
        y = np.log(N[ok]); tt = t[ok]
        r["laws"] = select(tt, y, B=B)
        # sensitivity: drop first 5 tau and stop when N < 10
        m = (tt >= 5) & (np.exp(y) >= 10)
        if m.sum() > 50:
            r["laws_window"] = select(tt[m], y[m], B=0)
        r["laws_sub2tau"] = select(tt[::4], y[::4], B=0)
    # the same analysis for the raw plaquette count (every phase singularity,
    # including +- pairs inside one interstitial cell and cores on droplets)
    Nr = d["Nraw"]
    r["Nraw0"] = float(Nr[:10].mean()); r["Nraw_end"] = float(Nr[-10:].mean())
    r["Nlat_end"] = float(N[-10:].mean())
    r["nmulti_mean"] = float(d["nmulti"].mean()); r["nmulti_max"] = int(d["nmulti"].max())
    okr = Nr > 0
    if do_stats and okr.sum() > 50 and Nr[okr].min() < Nr[okr].max():
        r["laws_raw"] = select(t[okr], np.log(Nr[okr]), B=B)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--B", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for f in a.folders:
        try:
            rows.append(analyse_run(f, B=a.B))
            print(rows[-1]["tag"], "t_melt %.1f" % rows[-1]["t_melt"], flush=True)
        except Exception as e:
            print("FAILED", f, e, flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)
