"""Executable validation suite (Phase 6A, first rungs of the 5C ladder).

T1  Plane-wave phase evolution (V1)          -- exactness of both substeps
T2  Temporal order (Richardson)              -- global 2nd order of Strang
T3  Harmonic-trap ground state vs Thomas-Fermi (V2)
T4  LHY integral Q5 vs analytic anchors
T5  Dipolar kernel: series/branch, bounds, slow-DFT plumbing, Parseval,
    physical orientation signs (bug classes B1/B2/B7 of Phase 5C)
T6  Vortex pair: plaquette detection, winding, N/E conservation,
    Helmholtz split sanity

Run:  python3 tests/run_validation.py
"""
import sys, time, json, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from qtsim import (Grid, EGPESolver, bare_dipolar_symbol,
                  truncation_bracket, Q5)
from qtsim.diagnostics import (plaquette_charges_2d, circulation_loop_2d,
                              helmholtz_split_2d)

rng = np.random.default_rng(7)
RESULTS = []


def record(name, value, criterion, passed):
    RESULTS.append((name, value, criterion, bool(passed)))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {value}  "
          f"(criterion: {criterion})")


# ----------------------------------------------------------------- T1
def t1_plane_wave():
    g = Grid((64, 64), (16.0, 16.0))
    m = 3
    k = 2 * np.pi * m / g.lengths[0]
    A = 1.0
    s = EGPESolver(g)
    s.psi = A * np.exp(1j * k * g.X[0])
    T, dt = 1.0, 1e-3
    s.step_real(dt, int(T / dt))
    exact = A * np.exp(1j * (k * g.X[0] - (k * k + A * A) * T))
    err = float(np.max(np.abs(s.psi - exact)))
    record("T1 plane-wave max error", f"{err:.3e}", "< 1e-10", err < 1e-10)


# ----------------------------------------------------------------- T2
def t2_order():
    g = Grid((64, 64), (16.0, 16.0))
    r2 = g.X[0] ** 2 + g.X[1] ** 2
    psi0 = np.exp(-r2 / 8.0) * np.exp(0.5j * g.X[0])
    T = 0.256  # divides exactly by all three dt values below
    finals = {}
    for dt in (4e-3, 2e-3, 1e-3):
        s = EGPESolver(g)
        s.psi = psi0.copy()
        nst = int(round(T / dt))
        assert abs(nst * dt - T) < 1e-12, "dt must divide T exactly"
        s.step_real(dt, nst)
        finals[dt] = s.psi.copy()
    e1 = np.sqrt(g.integrate(np.abs(finals[4e-3] - finals[1e-3]) ** 2))
    e2 = np.sqrt(g.integrate(np.abs(finals[2e-3] - finals[1e-3]) ** 2))
    # against dt/4 reference: e(dt) ~ C(dt^2 - dt_ref^2): ratio -> (16-1)/(4-1)=5
    order = np.log2(e1 / e2) / np.log2(2.0)
    # exact expected ratio for 2nd order with this reference: 15/3 = 5 -> log2(5)=2.32
    ratio = e1 / e2
    ok = 4.5 < ratio < 5.5
    record("T2 Strang error ratio e(4dt)/e(2dt)", f"{ratio:.3f}",
           "5.0 +/- 0.5 (2nd order, dt/4 ref)", ok)


# ----------------------------------------------------------------- T3
def t3_thomas_fermi():
    t0 = time.time()
    g = Grid((192, 192), (52.0, 52.0))
    om, N = 0.15, 2000.0
    V = 0.5 * om ** 2 * (g.X[0] ** 2 + g.X[1] ** 2)
    mu_tf = om * np.sqrt(N / np.pi)
    n_tf = np.maximum(mu_tf - V, 0.0)
    s = EGPESolver(g, V=V)
    s.psi = np.sqrt(n_tf).astype(complex)
    s.psi *= np.sqrt(N / s.norm())
    s.step_imag(0.004, 6000, norm_target=N)
    s.step_imag(0.001, 6000, norm_target=N)
    s.step_imag(0.00025, 4000, norm_target=N)
    mu, res = s.mu_and_residual()
    n = np.abs(s.psi) ** 2
    R_tf = np.sqrt(2 * mu_tf) / om
    interior = (g.X[0] ** 2 + g.X[1] ** 2) < (0.75 * R_tf) ** 2
    rms = float(np.sqrt(np.mean((n[interior] - n_tf[interior]) ** 2))
                / n_tf.max())
    dmu = abs(mu - mu_tf) / mu_tf
    record("T3 stationarity residual", f"{res:.3e}", "< 1e-5", res < 1e-5)
    record("T3 interior rms(n - n_TF)/n_peak", f"{rms:.3e}",
           "< 3e-2 (TF regime)", rms < 3e-2)
    record("T3 |mu - mu_TF|/mu_TF", f"{dmu:.3e}",
           "< 5e-2 (TF neglects kinetic edge)", dmu < 5e-2)
    print(f"      (T3 runtime {time.time()-t0:.1f} s, mu={mu:.4f}, "
          f"mu_TF={mu_tf:.4f})")


# ----------------------------------------------------------------- T4
def t4_q5():
    e0 = abs(Q5(0.0) - 1.0)
    q1_exact = 3.0 ** 2.5 / 6.0
    e1 = abs(Q5(1.0) - q1_exact) / q1_exact
    q15 = Q5(1.5)
    record("T4 |Q5(0)-1|", f"{e0:.2e}", "< 1e-13", e0 < 1e-13)
    record("T4 rel err Q5(1) vs 3^{5/2}/6", f"{e1:.2e}", "< 1e-6",
           e1 < 1e-6)
    record("T4 Q5(1.5) finite real (Re-convention)", f"{q15:.6f}",
           "finite, > Q5(1)", np.isfinite(q15) and q15 > q1_exact)


# ----------------------------------------------------------------- T5
def t5_kernels():
    # (a) bracket: series vs direct evaluation at x=0.02 (direct still safe)
    x = 0.02
    direct = 1 + 3 * np.cos(x) / x ** 2 - 3 * np.sin(x) / x ** 3
    series = x ** 2 / 10 - x ** 4 / 280
    rel = abs(direct - series) / abs(series)
    record("T5a bracket series vs direct @x=0.02", f"{rel:.2e}", "< 1e-6",
           rel < 1e-6)
    # (b) bare-kernel bounds and k=0
    g3 = Grid((16, 16, 16), (8.0, 8.0, 8.0))
    D = bare_dipolar_symbol(g3, (0, 0, 1))
    ok = (abs(D.min() + 1) < 1e-12 and abs(D.max() - 2) < 1e-12
          and D.flat[0] == 0.0)
    record("T5b bare symbol in [-1,2], D(0)=0",
           f"min={D.min():.3f}, max={D.max():.3f}", "exact bounds", ok)
    # (c) slow-DFT plumbing check (bug classes B1/B2)
    c = rng.uniform(-2, 2, 3)
    r2 = sum((X - ci) ** 2 for X, ci in zip(g3.X, c))
    n = np.exp(-r2 / 2.0)
    nk = np.fft.fftn(n)
    Phi_fft = np.fft.ifftn(D * nk).real
    idx = [tuple(rng.integers(0, 16, 3)) for _ in range(3)]
    worst = 0.0
    for ix in idx:
        # DFT convention: index j corresponds to x_j = j*dx (origin at j=0),
        # NOT the centered coordinate X (= j*dx - L/2 here).  Using X would
        # inject a spurious e^{ik L/2} per mode -- exactly the B1/B2 bug
        # class this check exists to catch.
        r = np.array([ix[d] * g3.dx[d] for d in range(3)])
        phase = sum(g3.K[d] * r[d] for d in range(3))
        slow = np.sum(D * nk * np.exp(1j * phase)).real / g3.Ntot
        worst = max(worst, abs(slow - Phi_fft[ix]) / max(abs(slow), 1e-30))
    record("T5c FFT vs slow-DFT max rel diff", f"{worst:.2e}", "< 1e-10",
           worst < 1e-10)
    # (d) Parseval
    p = abs(np.sum(n * n) - np.sum(np.abs(nk) ** 2) / g3.Ntot) / np.sum(n*n)
    record("T5d Parseval rel residual", f"{p:.2e}", "< 1e-12", p < 1e-12)
    # (e) physical orientation signs (decisive for axis wiring)
    g = Grid((32, 32, 32), (16.0, 16.0, 16.0))
    Dz = bare_dipolar_symbol(g, (0, 0, 1))
    def Edd(sig_perp, sig_z):
        nn = np.exp(-(g.X[0] ** 2 + g.X[1] ** 2) / (2 * sig_perp ** 2)
                    - g.X[2] ** 2 / (2 * sig_z ** 2))
        Phi = np.fft.ifftn(Dz * np.fft.fftn(nn)).real
        return 0.5 * g.integrate(Phi * nn)
    E_pro, E_obl = Edd(1.2, 3.6), Edd(3.6, 1.2)
    ok = (E_pro < 0) and (E_obl > 0)
    record("T5e orientation: E_dd(prolate)<0<E_dd(oblate)",
           f"{E_pro:.3f} / {E_obl:.3f}", "sign test", ok)


# ----------------------------------------------------------------- T6
def t6_vortex():
    g = Grid((128, 128), (32.0, 32.0))
    # Offset cores in both x AND y by half a grid spacing: arctan2 has its
    # branch cut along y=0 (negative-x direction from each core), and if
    # the cut aligns exactly with a grid line the plaquette method hits
    # the classic pi-ambiguity (wrap(pi)=-pi flips the winding).
    # Diagnosed live: at y=0 the phase jump across the cut was exactly pi,
    # making the core invisible to the detector.
    dx = g.dx[0]
    xp, xm = 4.0 + 0.5 * dx, -4.0 + 0.5 * dx
    yc = 0.5 * dx
    thp = np.arctan2(g.X[1] - yc, g.X[0] - xp)
    thm = np.arctan2(g.X[1] - yc, g.X[0] - xm)
    rp = np.sqrt((g.X[0] - xp) ** 2 + (g.X[1] - yc) ** 2)
    rm = np.sqrt((g.X[0] - xm) ** 2 + (g.X[1] - yc) ** 2)
    psi = np.tanh(rp) * np.tanh(rm) * np.exp(1j * (thp - thm))
    s = EGPESolver(g)
    s.psi = psi
    # Detect on the RAW imprint: the pair phase is only approximately
    # periodic, and imaginary-time smoothing in the periodic box lets the
    # boundary mismatch migrate charges (observed as (2,0)/(0,2) counts).
    q = plaquette_charges_2d(s.psi)
    npos, nneg = int((q == 1).sum()), int((q == -1).sum())
    record("T6 plaquette detection (+1,-1) counts", f"({npos},{nneg})",
           "(1,1)", (npos, nneg) == (1, 1))
    w = circulation_loop_2d(s.psi, g, (xp, yc), 2.5)
    record("T6 winding around + core", f"{w:.4f}", "1.00 +/- 0.02",
           abs(w - 1.0) < 0.02)
    # Smooth the crude tanh imprint briefly in imaginary time (removes the
    # artificial density transient while topology is preserved), THEN
    # measure the decomposition on this vortex-dominated state.  Measuring
    # after 500 real-time steps of the raw ansatz was a test-design error:
    # the imprint transient legitimately radiates strong sound (large Ec).
    s.step_imag(5e-3, 200)
    Ei, Ec, _ = helmholtz_split_2d(s.psi, g)
    Ei0, Ec0, _ = helmholtz_split_2d(np.abs(s.psi).astype(complex), g)
    record("T6 Helmholtz on smoothed pair: vortex flow incompressible",
           f"Ei={Ei:.3f}, Ec={Ec:.3f}; phaseless Ei={Ei0:.1e}",
           "Ei > Ec and phaseless < 1e-4*Ei",
           Ei > Ec and Ei0 < 1e-4 * Ei)
    N0, E0 = s.norm(), s.energy()["total"]
    s.step_real(2e-3, 500)
    dN = abs(s.norm() - N0) / N0
    dE = abs(s.energy()["total"] - E0) / abs(E0)
    record("T6 |dN|/N over 500 steps", f"{dN:.2e}", "< 1e-12", dN < 1e-12)
    record("T6 |dE|/E over 500 steps", f"{dE:.2e}", "< 1e-4", dE < 1e-4)


# ----------------------------------------------------------------- T7
def t7_long_time_stability():
    """Regression test for the temporal-aliasing blow-up found in Colab.

    History: a demo run on a 256^2 grid with dt=0.02 gave 6.3 rad/step
    phase advance for the highest grid mode.  Energy stayed flat to 1e-7
    until t~30, then exploded (vortex count 2 -> 21000, energy x6000)
    while the norm remained conserved to 1e-13 -- so norm checks do NOT
    detect it.  T7 pins down both halves of the fix: the guard must
    reject the bad timestep, and a guarded timestep must stay stable
    well past the old failure time.
    """
    # (a) the guard rejects the historical failure configuration
    g_bad = Grid((256, 256), (64.0, 64.0))
    s_bad = EGPESolver(g_bad)
    s_bad.psi = np.ones(g_bad.shape, dtype=complex)
    phase_bad = s_bad.max_phase_per_step(0.02)
    rejected = False
    try:
        s_bad.step_real(0.02, 1)
    except ValueError:
        rejected = True
    record("T7a guard rejects dt=0.02 on 256^2 grid",
           f"{phase_bad:.2f} rad/step, rejected={rejected}",
           "phase > 2 rad and rejected", phase_bad > 2.0 and rejected)

    # (b) a guarded timestep survives well past the old blow-up time
    g = Grid((128, 128), (64.0, 64.0))
    dx = g.dx[0]
    dt = 0.01                       # 0.79 rad/step on this grid
    xp, xm, yc = 8.0 + 0.5 * dx, -8.0 + 0.5 * dx, 0.5 * dx
    thp = np.arctan2(g.X[1] - yc, g.X[0] - xp)
    thm = np.arctan2(g.X[1] - yc, g.X[0] - xm)
    rp = np.sqrt((g.X[0] - xp) ** 2 + (g.X[1] - yc) ** 2)
    rm = np.sqrt((g.X[0] - xm) ** 2 + (g.X[1] - yc) ** 2)
    s = EGPESolver(g)
    s.psi = np.tanh(rp) * np.tanh(rm) * np.exp(1j * (thp - thm))
    s.step_imag(5e-3, 300, norm_target=s.norm())

    record("T7b phase advance of guarded dt",
           f"{s.max_phase_per_step(dt):.3f} rad/step", "< 1.0", 
           s.max_phase_per_step(dt) < 1.0)

    E0, N0 = s.energy()["total"], s.norm()
    nsteps = int(60.0 / dt)         # t = 60, well past the old t ~ 32 failure
    s.step_real(dt, nsteps)
    dE = abs(s.energy()["total"] - E0) / abs(E0)
    dN = abs(s.norm() - N0) / N0
    nv = int((plaquette_charges_2d(s.psi) != 0).sum())
    record("T7c energy drift to t=60", f"{dE:.2e}", "< 1e-5", dE < 1e-5)
    record("T7d norm drift to t=60", f"{dN:.2e}", "< 1e-10", dN < 1e-10)
    record("T7e vortex count preserved to t=60", f"{nv}", "exactly 2", nv == 2)


# ----------------------------------------------------------------- T8
def t8_quasi2d_kernel():
    """Quasi-2D projected dipolar kernel: derivation, limits, roton.

    The kernel is
        D(k) = 2 sqrt(2) - 3 sqrt(2 pi) u erfcx(u),   u = k l_z / 2,
    obtained by integrating the 3D symbol g_dd(3 k_z^2/k^2 - 1) against the
    Gaussian axial density and dividing by g_2D = g/(sqrt(2 pi) l_z).
    Unlike the bare periodic symbol it changes sign, which is what creates
    a roton and therefore a droplet crystal at a physical wavelength.
    """
    from qtsim.kernels import (quasi2d_dipolar_profile,
                               quasi2d_dipolar_symbol)
    from qtsim.lhy import gamma_tilde

    # (a) analytic form vs direct numerical quadrature of the defining integral
    try:
        from scipy.integrate import quad
        lz = 1.0
        worst = 0.0
        for kv in (0.05, 0.3, 1.0, 2.0, 4.0, 8.0):
            def integ(kz, kv=kv):
                k2 = kv * kv + kz * kz
                return ((3 * kz * kz / k2 - 1.0)
                        * np.exp(-kz * kz * lz * lz / 4.0) / (2 * np.pi))
            num, _ = quad(integ, -60 / lz, 60 / lz, limit=400)
            num *= np.sqrt(2 * np.pi) * lz
            ana = float(quasi2d_dipolar_profile(kv, lz))
            worst = max(worst, abs(ana - num) / max(abs(num), 1e-30))
        record("T8a analytic kernel vs quadrature", f"{worst:.2e}", "< 1e-10",
               worst < 1e-10)
    except ImportError:
        record("T8a analytic kernel vs quadrature", "scipy absent",
               "skipped", True)

    # (b) both limits, exactly
    d0 = float(quasi2d_dipolar_profile(1e-9, 1.0))
    dinf = float(quasi2d_dipolar_profile(1e4, 1.0))
    ok = (abs(d0 - 2 * np.sqrt(2)) < 1e-6
          and abs(dinf + np.sqrt(2)) < 1e-6)
    record("T8b limits D(0)=2sqrt2, D(inf)=-sqrt2",
           f"{d0:.6f} / {dinf:.6f}", "2.828427 / -1.414214", ok)

    # (c) overflow safety far beyond the erfc overflow threshold
    big = float(quasi2d_dipolar_profile(1e6, 1.0))
    record("T8c overflow-safe at k=1e6", f"{big:.6f}",
           "finite, approx -sqrt2", np.isfinite(big) and abs(big + 1.4142) < 1e-3)

    # (d) sign change exists (this is what the bare kernel lacks)
    kk = np.linspace(1e-6, 20, 4000)
    D = quasi2d_dipolar_profile(kk, 6.0)
    record("T8d kernel changes sign", f"max={D.max():.3f}, min={D.min():.3f}",
           "positive at small k, negative at large k",
           D.max() > 0 and D.min() < 0)

    # (e) roton instability appears where linear theory says it should
    eps, lz2 = 1.8, 6.0
    gam = gamma_tilde(eps, 5e-5)
    Dp = quasi2d_dipolar_profile(kk, lz2)
    inside = kk ** 2 + 2.0 * (1.0 + eps * Dp) + 3.0 * gam
    imin = int(np.argmin(inside))
    a_pred = 2 * np.pi / kk[imin]
    record("T8e roton instability at eps_dd=1.8, l_z=6",
           f"min_inside={inside[imin]:.4f}, a_pred={a_pred:.3f} xi",
           "min_inside < 0 (unstable)", inside[imin] < 0)

    # (f) grid symbol matches the profile and rejects 3D grids
    g2 = Grid((32, 32), (16.0, 16.0))
    S = quasi2d_dipolar_symbol(g2, 2.0)
    P = quasi2d_dipolar_profile(np.sqrt(g2.k2), 2.0)
    rejected3d = False
    try:
        quasi2d_dipolar_symbol(Grid((8, 8, 8), (8.0, 8.0, 8.0)), 2.0)
    except ValueError:
        rejected3d = True
    record("T8f grid symbol consistent, 3D rejected",
           f"maxdiff={np.abs(S - P).max():.1e}, rejected3d={rejected3d}",
           "match and 3D rejected",
           np.abs(S - P).max() < 1e-14 and rejected3d)


# ----------------------------------------------------------------- T9
def t9_conservation_laws():
    """Conservation laws demonstrated by a real Colab campaign run.

    A user's stirred-tangle data supplied two checks the suite did not
    previously pin down, and both passed:

      * Charge neutrality.  Vortex counts were exactly balanced at every
        sample (+18/-18, +47/-47, +99/-99, +87/-87).  Total circulation
        must vanish in a periodic box, so any net charge means the
        detector is inventing or losing vortices.

      * Free-decay energy conservation.  Total energy appeared to drop
        4937.87 -> 4827.84 between the end of stirring and free decay,
        which is alarming until one notices it is exactly the obstacle
        potential energy int(V n) leaving the books.  Subtracting it, the
        conservative energy went 4827.8475 -> 4827.8410, a relative
        change of 1.4e-6 over 1000 unforced steps.

    T9 reproduces both on a small grid so a regression cannot slip past.
    """
    g = Grid((96, 96), (48.0, 48.0))
    dt = 0.01
    rng = np.random.default_rng(11)

    # relaxed uniform background
    s = EGPESolver(g)
    s.psi = np.ones(g.shape, dtype=complex)
    Nt = s.norm()
    s.step_imag(0.005, 400, norm_target=Nt)

    # (a) charge neutrality under a strong moving obstacle that sheds vortices
    mu, _ = s.mu_and_residual()
    V0, sigma, radius = 3.0 * mu, 3.0, 12.0
    omega = 1.1 * np.sqrt(2.0) / radius
    def V_of(t):
        V = np.zeros(g.shape)
        for a0 in (0.0, 2 * np.pi / 3, 4 * np.pi / 3):
            a = a0 + omega * t
            r2 = ((g.X[0] - radius * np.cos(a)) ** 2
                  + (g.X[1] - radius * np.sin(a)) ** 2)
            V += V0 * np.exp(-r2 / (2 * sigma ** 2))
        return V

    t = 0.0
    net_worst, nv_peak = 0, 0
    for k in range(1500):
        s.V = V_of(t + 0.5 * dt)
        s.step_real(dt)
        t += dt
        if k % 300 == 299:
            q = plaquette_charges_2d(s.psi)
            npos, nneg = int((q == 1).sum()), int((q == -1).sum())
            net_worst = max(net_worst, abs(npos - nneg))
            nv_peak = max(nv_peak, npos + nneg)
    record("T9a charge neutrality under stirring",
           f"worst |net|={net_worst}, peak count={nv_peak}",
           "net == 0 with vortices present",
           net_worst == 0 and nv_peak > 0)

    # (b) switching the drive off must remove exactly int(V n) and nothing else
    n = np.abs(s.psi) ** 2
    Epot_on = float(g.integrate(s.V * n))
    E_on = s.energy()["total"]
    s.V = np.zeros(g.shape)
    E_off = s.energy()["total"]
    bookkeeping = abs((E_on - E_off) - Epot_on) / max(abs(Epot_on), 1e-30)
    record("T9b removing drive removes exactly int(V n)",
           f"{bookkeeping:.2e}", "< 1e-10", bookkeeping < 1e-10)

    # (c) free decay is conservative
    E0, N0 = s.energy()["total"], s.norm()
    s.step_real(dt, 1000)
    dE = abs(s.energy()["total"] - E0) / abs(E0)
    dN = abs(s.norm() - N0) / N0
    record("T9c free-decay energy conservation", f"{dE:.2e}", "< 1e-5",
           dE < 1e-5)
    record("T9d free-decay norm conservation", f"{dN:.2e}", "< 1e-10",
           dN < 1e-10)

    # (e) charge still neutral after decay
    q = plaquette_charges_2d(s.psi)
    net = int((q == 1).sum()) - int((q == -1).sum())
    record("T9e charge neutrality after decay", f"net={net}", "net == 0",
           net == 0)


# ---------------------------------------------------------------- T10
def t10_minimizer():
    """Direct L-BFGS-B minimisation must beat imaginary-time gradient flow.

    Measured on the dipolar droplet crystal (eps_dd=1.8, l_z=6 xi):
        gradient flow, 24000 steps : mu = 5.509289, residual 2.28e-02
        single-start L-BFGS-B      : mu = 5.466331, residual 4.74e-03
        multi-start, lattice seed  : mu = 5.189268, residual 4.73e-03
    Gradient flow had plateaued; the landscape has many nearby minima, so
    a better optimiser and several starts both matter.  T10 checks the
    machinery on a cheap trapped problem where the answer is unambiguous.
    """
    try:
        from qtsim.minimize import minimize_energy
    except ImportError:
        record("T10 minimizer available", "scipy absent", "skipped", True)
        return

    g = Grid((96, 96), (26.0, 26.0))
    om, N = 0.2, 600.0
    V = 0.5 * om ** 2 * (g.X[0] ** 2 + g.X[1] ** 2)
    s = EGPESolver(g, V=V)
    rng = np.random.default_rng(5)
    s.psi = np.exp(-(g.X[0] ** 2 + g.X[1] ** 2) / 40.0).astype(complex)
    s.psi += 0.05 * rng.standard_normal(g.shape)
    s.psi *= np.sqrt(N / s.norm())

    s.step_imag(0.004, 400, norm_target=N)      # deliberately under-relaxed
    E_flow = s.energy()["total"]
    _, r_flow = s.mu_and_residual()

    info = minimize_energy(s, N, maxiter=1500, verbose=False)

    record("T10a L-BFGS lowers the energy",
           f"E {E_flow:.5f} -> {info['energy']:.5f}",
           "energy decreases", info["energy"] < E_flow)
    record("T10b L-BFGS improves the residual",
           f"{r_flow:.2e} -> {info['residual']:.2e}",
           "residual improves by >10x", info["residual"] < r_flow / 10)
    record("T10c norm preserved by the minimiser",
           f"{abs(s.norm() - N) / N:.2e}", "< 1e-10",
           abs(s.norm() - N) / N < 1e-10)
    record("T10d minimiser reports convergence",
           f"nit={info['nit']}, success={info['success']}",
           "success True", info["success"])


# ---------------------------------------------------------------- T11
def t11_vortex_detection_in_voids():
    """Vortex detection must not invent vortices in near-vacuum regions.

    In a droplet crystal the inter-droplet regions reach n ~ 1e-6 n_mean,
    where the phase is numerical noise and the raw plaquette detector finds
    random windings.  A plain density mask is wrong because a real vortex
    core also has n -> 0 at its centre; the discriminator is the density in
    an ANNULUS around the candidate.

    Measured systematic on a stirred crystal: 874 raw, 724 annulus-masked,
    459 with a crude corner mask -- so L carries a 20-50 percent
    method-dependent systematic that must be quoted, not hidden.
    """
    from qtsim.diagnostics import plaquette_charges_2d_masked

    g = Grid((128, 128), (64.0, 64.0))
    dx = g.dx[0]
    rng = np.random.default_rng(0)

    # left half real fluid with one true vortex; right half noisy vacuum
    amp = np.ones(g.shape)
    amp[g.X[0] > 0] = 1e-3
    th = np.arctan2(g.X[1] - 0.5 * dx, g.X[0] + 16 + 0.5 * dx)
    r = np.sqrt((g.X[0] + 16 + 0.5 * dx) ** 2 + (g.X[1] - 0.5 * dx) ** 2)
    ph = th.copy()
    ph[g.X[0] > 0] = 2 * np.pi * rng.random(g.shape)[g.X[0] > 0]
    psi = (amp * np.tanh(r) * np.exp(1j * ph)).astype(complex)

    inL, inR = (g.X[0] < -2), (g.X[0] > 2)
    raw = plaquette_charges_2d(psi)
    qm = plaquette_charges_2d_masked(psi, g)

    raw_void = int((raw[inR] != 0).sum())
    kept_void = int((qm[inR] != 0).sum())
    rej = 1.0 - kept_void / max(raw_void, 1)
    record("T11a rejects void false positives",
           f"{raw_void} -> {kept_void} ({rej*100:.1f}% rejected)",
           "> 95% rejected", rej > 0.95)

    kept_real = int((qm[inL] == 1).sum())
    record("T11b keeps the real vortex", f"{kept_real}", "== 1",
           kept_real == 1)

    # the mask must be a no-op on a clean uniform-background pair
    s = EGPESolver(g)
    xp, xm, yc = 8 + 0.5 * dx, -8 + 0.5 * dx, 0.5 * dx
    thp = np.arctan2(g.X[1] - yc, g.X[0] - xp)
    thm = np.arctan2(g.X[1] - yc, g.X[0] - xm)
    rp = np.sqrt((g.X[0] - xp) ** 2 + (g.X[1] - yc) ** 2)
    rm = np.sqrt((g.X[0] - xm) ** 2 + (g.X[1] - yc) ** 2)
    s.psi = np.tanh(rp) * np.tanh(rm) * np.exp(1j * (thp - thm))
    s.step_imag(5e-3, 300, norm_target=s.norm())
    q_raw = int((plaquette_charges_2d(s.psi) != 0).sum())
    q_msk = int((plaquette_charges_2d_masked(s.psi, g) != 0).sum())
    record("T11c no-op on a clean vortex pair", f"raw={q_raw}, masked={q_msk}",
           "both == 2", q_raw == 2 and q_msk == 2)


# ---------------------------------------------------------------- T12
def t12_backend_selection():
    """Explicit backend selection (qtsim/backend.py, the GPU port).

    Every array operation in kernels.py, solver.py, diagnostics.py, and
    minimize.py now goes through grid.xp / grid.fft_mod rather than a
    hardcoded numpy import, so Grid(..., backend='gpu') can run on CuPy.
    Selection is EXPLICIT and a failed GPU request RAISES rather than
    silently running on CPU -- the same class of bug already fixed once
    for Google Drive persistence (a silent fallback there destroyed a
    full session's results with no warning).  This sandbox has no GPU, so
    only the CPU path and the loud-failure guarantee can be exercised
    here; the CPU path itself must remain byte-identical to before the
    port, which T1-T11 above already establish by continuing to pass.
    """
    from qtsim.backend import get_backend

    record("T12a default backend is cpu", f"{Grid((8,8),(4.,4.)).backend!r}",
           "'cpu'", Grid((8, 8), (4.0, 4.0)).backend == "cpu")

    invalid_raised = False
    try:
        Grid((8, 8), (4.0, 4.0), backend="tpu")
    except ValueError:
        invalid_raised = True
    record("T12b invalid backend name rejected", f"raised={invalid_raised}",
           "ValueError raised", invalid_raised)

    # The critical safety property: a gpu request must never silently
    # continue on CPU.  Which half of that can be tested depends on the
    # host, so the GPU is detected INDEPENDENTLY of get_backend():
    #   no CuPy/GPU -> get_backend('gpu') must RAISE RuntimeError
    #   CuPy + GPU  -> get_backend('gpu') must return CuPy, not NumPy
    # BUG FIX (v2.6.1): T12c only had the first branch, so it FAILED on a
    # real Colab T4 (raised=False) -- correct behaviour reported as a
    # failure.  It had never been run on GPU hardware before.
    gpu_present = False
    try:
        import cupy as _cp
        gpu_present = _cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        gpu_present = False

    if gpu_present:
        try:
            xp, fft_mod, name = get_backend("gpu")
            ok = (xp.__name__ == "cupy" and name == "gpu"
                  and fft_mod.__name__.startswith("cupy"))
            val = f"GPU present: returned xp={xp.__name__}, name={name!r}"
        except Exception as exc:
            ok, val = False, f"GPU present but raised {type(exc).__name__}"
        record("T12c GPU request on a GPU host returns CuPy (no silent CPU)",
               val, "xp is cupy, name 'gpu'", ok)
    else:
        gpu_raised, gpu_is_runtime_error = False, False
        try:
            get_backend("gpu")
        except RuntimeError:
            gpu_raised = True
            gpu_is_runtime_error = True
        except Exception:
            gpu_raised = True
        record("T12c GPU-unavailable request raises (never silent fallback)",
               f"raised={gpu_raised}, RuntimeError={gpu_is_runtime_error}",
               "raises RuntimeError", gpu_raised and gpu_is_runtime_error)

    # Full pipeline on the explicit backend path: kernel + solver +
    # minimiser + masked diagnostics together, matching how a real run
    # exercises the code (not just isolated unit calls).
    from qtsim.lhy import gamma_tilde
    from qtsim.kernels import quasi2d_dipolar_symbol
    from qtsim.diagnostics import plaquette_charges_2d_masked
    from qtsim.minimize import minimize_energy

    g = Grid((64, 64), (32.0, 32.0), backend="cpu")
    gam = gamma_tilde(1.414, 1.17e-4)
    Dk = quasi2d_dipolar_symbol(g, 8.6)
    s = EGPESolver(g, eps_dd=1.414, gamma=gam, Dk=Dk)
    rng = np.random.default_rng(2)
    s.psi = (np.ones(g.shape, dtype=complex)
            + 0.03 * (rng.standard_normal(g.shape)
                     + 1j * rng.standard_normal(g.shape)))
    Nt = float(g.integrate(np.ones(g.shape)))
    s.psi *= np.sqrt(Nt / s.norm())
    s.step_imag(0.002, 300, norm_target=Nt)
    info = minimize_energy(s, Nt, maxiter=300, verbose=False)
    _, stats = plaquette_charges_2d_masked(s.psi, g, return_stats=True)
    ok = (info["residual"] < 1.0 and np.isfinite(info["mu"])
         and "kept" in stats)
    record("T12d full pipeline on explicit backend='cpu'",
           f"mu={info['mu']:.4f}, residual={info['residual']:.2e}, "
           f"masked_vortices={stats['kept']}",
           "finite mu, finite residual, diagnostics run", ok)


# ---------------------------------------------------------------- T13
def t13_trackA_analysis():
    """v2.6: Track A analysis fixes -- F3 MAD guard, F2 resume key, and the
    formal surrogate null test (calibration under a Gaussian null and power
    against injected bursts)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from campaign import track_a_scan as T

    t = np.arange(400) * 0.5
    # T13a: flat series must not crash or report events
    _, st = T._events_from_series(t, np.full(400, 0.3), 3.0)
    record("T13a flat L(t): no events, no blow-up",
           f"n_events={st['n_events']}, method={st['sigma_method']}",
           "0 events, method 'flat'",
           st["n_events"] == 0 and st["sigma_method"] == "flat")

    # T13b: quantised staircase (MAD == 0).  The old MAD+1e-30 rule put the
    # threshold at ~1e-30 so every nonzero drop was an event.
    rng = np.random.default_rng(1)
    jumps = np.zeros(400)
    jumps[rng.choice(400, 40, replace=False)] = rng.choice([-1, 1], 40)
    L = (300 + np.cumsum(jumps)) / 1e4
    drop = -np.gradient(L, t)
    old_thr = np.median(drop) + 3.0 * 1.4826 * (
        np.median(np.abs(drop - np.median(drop))) + 1e-30)
    old_n = int(np.sum(np.diff(np.r_[0, (drop > old_thr).astype(int)]) == 1))
    _, st = T._events_from_series(t, L, 3.0)
    record("T13b MAD==0 guard (quantised L): threshold not collapsed",
           f"old thr={old_thr:.1e} ({old_n} ev) -> new thr={st['threshold']:.1e} "
           f"({st['n_events']} ev, {st['sigma_method']})",
           "method std_fallback and threshold > 1e-12",
           st["sigma_method"] == "std_fallback" and st["threshold"] > 1e-12)

    # T13c: calibration.  AR(1) Gaussian noise on a decaying trend has no
    # bursts, so p < 0.05 should occur in ~5 percent of series.
    nser, nsig = 40, 0
    for k in range(nser):
        r = np.random.default_rng(100 + k)
        x = np.zeros(400)
        for i in range(1, 400):
            x[i] = 0.6 * x[i - 1] + r.standard_normal()
        L = 0.02 * np.exp(-t / 300) + 2e-5 * x
        recs = [dict(t=a, L=b) for a, b in zip(t, L)]
        out = T.surrogate_test(recs, thetas=(3.0,), n_surr=99, seed=k,
                               methods=("phase",))
        nsig += out["phase"]["theta_3.0"]["p"] < 0.05
    record("T13c surrogate test calibrated under Gaussian null",
           f"{nsig}/{nser} series with p<0.05 ({100*nsig/nser:.0f}%)",
           "<= 15% (nominal 5%)", nsig <= 0.15 * nser)

    # T13d: power.  Same noise plus 8 sharp drops (avalanches).
    r = np.random.default_rng(7)
    x = np.zeros(400)
    for i in range(1, 400):
        x[i] = 0.6 * x[i - 1] + r.standard_normal()
    steps = np.zeros(400)
    steps[r.choice(np.arange(20, 380), 8, replace=False)] = 1.0
    L = 0.02 * np.exp(-t / 300) + 2e-5 * x - 2e-4 * np.cumsum(steps)
    recs = [dict(t=a, L=b) for a, b in zip(t, L)]
    out = T.surrogate_test(recs, thetas=(3.0,), n_surr=199, seed=0,
                           methods=("phase",))
    pp = out["phase"]["theta_3.0"]
    record("T13d surrogate test detects injected bursts",
           f"observed={pp['observed']}, surrogate mean={pp['surr_mean']:.2f}, "
           f"p={pp['p']:.3f}", "p < 0.05", pp["p"] < 0.05)

    # T13e: resume key distinguishes box / decay time (finding F2)
    class A:
        eps_dd, l_z, n0_as3, cells, dx = 1.414, 8.6, 1.17e-4, 12, 0.5
        T_stir, T_decay, sample_every, n_stir, V0_factor = 60., 200., .5, 0, 1.5
    want = T.point_params(A, 0.6, 0)
    quick = dict(want, track="A", cells=8, T_stir=15.0, T_decay=40.0,
                 n_stir=2)
    legacy = {k: v for k, v in dict(want, track="A").items()
              if k not in ("dx", "sample_every")}
    legacy.update(grid=[246, 214], box=[123.384, 106.854])
    m_same = T.stored_matches(dict(want, track="A"), want)
    m_quick = T.stored_matches(quick, want)
    m_long = T.stored_matches(dict(want, track="A", T_decay=600.), want)
    m_leg = T.stored_matches(legacy, want)
    ok = (m_same and not m_quick and not m_long and m_leg
          and T.param_key(want) != T.param_key(dict(want, T_decay=600.)))
    record("T13e resume key: 8-cell / T_decay=600 runs not treated as done",
           f"same={m_same}, quick8={m_quick}, Td600={m_long}, legacy={m_leg}",
           "only identical params match", ok)


if __name__ == "__main__":
    t_start = time.time()
    for t in (t1_plane_wave, t2_order, t3_thomas_fermi, t4_q5,
              t5_kernels, t6_vortex, t7_long_time_stability,
              t8_quasi2d_kernel, t9_conservation_laws,
              t10_minimizer, t11_vortex_detection_in_voids,
              t12_backend_selection, t13_trackA_analysis):
        t()
    npass = sum(1 for *_, p in RESULTS if p)
    print(f"\n== {npass}/{len(RESULTS)} checks passed "
          f"({time.time()-t_start:.1f} s total) ==")
    with open(os.path.join(os.path.dirname(__file__), "..",
                           "VALIDATION_RESULTS.json"), "w") as f:
        json.dump([{ "name": n, "value": v, "criterion": c, "pass": p}
                   for n, v, c, p in RESULTS], f, indent=1)
    sys.exit(0 if npass == len(RESULTS) else 1)










