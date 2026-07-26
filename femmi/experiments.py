"""
femmi/experiments.py
Reusable pieces for the paper's core experiments -- the ones that decide whether
to choose FEMMI over Kaiser-Squires in a real weak-lensing pipeline. The example
scripts in examples/paper/ are thin wrappers that call these and plot the result.

The thesis is structural, not RMSE. Two claims were tested; they did NOT fare
equally, and the difference is the whole point of this module:

  (i) FEMMI imposes the correct far-field boundary condition, reducing edge bias.
      HOLDS on independent truth -- ~1.7x lower DC-removed error than KS in the
      corner region, and the margin grows outward. See independent_truth_recovery.

 (ii) FEMMI breaks the DC / mass-sheet degeneracy, recovering the *absolute*
      normalisation. DOES NOT HOLD in practice. The DC mode is genuinely removed
      from the forward operator's null space (constant_mode_response > 0 while
      ks_constant_mode_response == 0 exactly), but ~99.9% of that response sits in
      the square domain's corners and grows under refinement instead of
      converging. On truth FEMMI did not generate, its mean-kappa error is 0.047
      against KS's 0.049. See MATH.md 6.3a.

Experimental-design note. mass_sheet_recovery and boundary_error_profile generate
their test shear with FEMMI's OWN forward (femmi_forward_shear). That is the
regime FEMMI's far-field assumption exactly describes, and it shows the mechanism
cleanly -- but it is an inverse crime, because the DC component of the truth is
then by construction in the range of F. independent_truth_recovery repeats both
measurements against femmi.truth (analytic GalSim NFW, or a MassiveNuS map with
aperiodic shear), which neither method's forward produced. Quote THAT one.
"""

from __future__ import annotations
import numpy as np

from .operators import build_operators
from .forward import DifferentiableForward
from .inverse import MAPReconstructor, kaiser_squires
from .catalog import analytic_gaussian_shear


# --------------------------------------------------------------------------- #
# forward / reconstruction helpers
# --------------------------------------------------------------------------- #
def square_ops(nx, half_width=2.5, coupling="steinbach"):
    return build_operators(nx, nx, -half_width, half_width, -half_width, half_width,
                           verbose=False, coupling=coupling)


def femmi_forward_shear(ops, kappa):
    """Shear predicted by FEMMI's own forward gamma = F(kappa) (float64), i.e. the
    isolated-field / far-field-zero regime FEMMI assumes."""
    import jax.numpy as jnp
    fwd = DifferentiableForward(ops, lam_reg=1e-2)
    g1, g2 = fwd.gamma_from_kappa(jnp.asarray(kappa))
    return np.asarray(g1), np.asarray(g2)


def femmi_map(ops, g1, g2, noise_std, wiener_length=0.5, lam=None, weight=None,
              prior=None, prior_kw=None):
    """FEMMI MAP reconstruction (Morozov-selected lambda when lam is None).

    prior: None for the default Wiener/Matern prior, or a kind string accepted by
    priors.make_prior ('tv', 'sparse', 'maxent', ...), or a Prior instance. Note
    that Morozov lambda-selection applies to the Wiener prior only; custom priors
    run at the fixed lam_reg, matching catalog.reconstruct_catalog.
    """
    fwd = DifferentiableForward(ops, lam_reg=(1e-2 if lam is None else lam))
    prior_obj = prior
    if isinstance(prior, str):
        from .priors import make_prior
        prior_obj = make_prior(prior, ops, **(prior_kw or {}))
    rec = MAPReconstructor(fwd, wiener_length=wiener_length,
                           noise_std=(noise_std if lam is None else None),
                           data_weight=weight, prior=prior_obj, callback_every=0)
    k, _ = rec.reconstruct(g1, g2, verbose=False)
    return np.asarray(k)


def ks_map(g1, g2, nodes, grid_size=64):
    """Kaiser-Squires reconstruction sampled at the mesh nodes."""
    return np.asarray(kaiser_squires(g1, g2, nodes, grid_size=grid_size))


def radial_profile(kappa, nodes, center=(0.0, 0.0), nbins=24, rmax=None):
    """Azimuthally-averaged kappa(r). Returns (r_centres, profile)."""
    r = np.hypot(nodes[:, 0] - center[0], nodes[:, 1] - center[1])
    rmax = float(rmax if rmax is not None else r.max())
    edges = np.linspace(0.0, rmax, nbins + 1)
    which = np.clip(np.digitize(r, edges) - 1, 0, nbins - 1)
    k = np.asarray(kappa)
    prof = np.array([np.nanmean(k[which == b]) if np.any(which == b) else np.nan
                     for b in range(nbins)])
    return 0.5 * (edges[:-1] + edges[1:]), prof


# --------------------------------------------------------------------------- #
# P0.1 -- DC mode / absolute normalisation (mass-sheet degeneracy)
# --------------------------------------------------------------------------- #
def mass_sheet_recovery(nx=16, half_width=2.5, sigma=0.6, amp=1.0, noise_std=0.02,
                        wiener_length=1.0, ks_grid=48, seed=0, self_consistent=True):
    """Reconstruct a compact, known-nonzero-mean convergence field with FEMMI and
    KS and compare recovery of the ABSOLUTE level (the DC / mass-sheet mode).

    MECHANISM DEMO, NOT THE HEADLINE NUMBER. self_consistent=True generates the
    shear from FEMMI's own forward, so the truth's DC component is by construction
    in the range of F -- an inverse crime, and the ~40x advantage it reports does
    not transfer to real data. self_consistent=False uses infinite-domain analytic
    shear instead, and FEMMI's DC recovery does not merely degrade, it vanishes
    (mean_femmi ~ 0.000 against a truth mean of 0.086).

    Use independent_truth_recovery for the number to quote; keep this for showing
    the mechanism in the regime FEMMI's far-field assumption exactly describes.

    Returns a dict with the maps, radial profiles, and mean-kappa errors.
    """
    ops = square_ops(nx, half_width)
    nodes = np.array(ops.mesh.nodes)
    kt, g1a, g2a = analytic_gaussian_shear(nodes, sigma=sigma, amp=amp)
    if self_consistent:
        g1t, g2t = femmi_forward_shear(ops, kt)
    else:
        g1t, g2t = np.asarray(g1a), np.asarray(g2a)
    rng = np.random.default_rng(seed)
    g1n = g1t + rng.normal(0, noise_std, len(g1t))
    g2n = g2t + rng.normal(0, noise_std, len(g2t))

    k_f = femmi_map(ops, g1n, g2n, noise_std, wiener_length=wiener_length,
                    weight=np.ones(len(nodes)))
    k_ks = ks_map(g1n, g2n, nodes, grid_size=ks_grid)

    r, p_t = radial_profile(kt, nodes)
    _, p_f = radial_profile(k_f, nodes)
    _, p_ks = radial_profile(k_ks, nodes)
    mt = float(np.nanmean(kt))
    return dict(nodes=nodes, truth=kt, femmi=k_f, ks=k_ks, r=r,
                prof_truth=p_t, prof_femmi=p_f, prof_ks=p_ks,
                mean_truth=mt, mean_femmi=float(np.nanmean(k_f)), mean_ks=float(np.nanmean(k_ks)),
                err_femmi=abs(float(np.nanmean(k_f)) - mt),
                err_ks=abs(float(np.nanmean(k_ks)) - mt))


# --------------------------------------------------------------------------- #
# P0.1 + P1.4 -- the SAME test on INDEPENDENT truth (no inverse crime)
# --------------------------------------------------------------------------- #
def independent_truth_recovery(nx=18, half_width=2.5, source="nfw", noise_std=0.02,
                               wiener_length=1.0, ks_grid=48, nbins=14, seed=0,
                               kappa_max=1.0, truth_kw=None):
    """The headline comparison, run on truth that NEITHER method generated.

    `femmi.truth` supplies kappa and gamma from an analytic GalSim NFW halo field
    (or a MassiveNuS simulation map with aperiodic shear). The shear therefore
    comes from neither FEMMI's FEM-BEM forward nor the periodic KS FFT, which
    closes the inverse-crime objection to mass_sheet_recovery / boundary_error_
    profile: those generate the test shear with FEMMI's own operator.

    Measures both paper claims at once, since they share all the setup:
      * absolute normalisation (P0.1): error in the recovered MEAN kappa;
      * boundary bias (P1.4): DC-removed error vs radius, FEMMI vs KS.

    WHAT THIS TEST ACTUALLY SHOWS (measured, nx=14, shape noise 0.02):

      * Boundary bias -- FEMMI WINS, and by more the closer to the edge you look.
        For one compact halo: corner-region error 0.018 vs KS 0.031 (1.7x), and
        the margin grows monotonically outward. Holds with the halo placed near
        the edge too (0.014 vs 0.030, 2.1x). This claim survives.

      * Absolute normalisation -- FEMMI DOES NOT WIN. mean-kappa error 0.0485 vs
        KS 0.0503; FEMMI recovers ~4% of the true mean. The dramatic result in
        mass_sheet_recovery (0.002 vs 0.085) is an artefact of generating the test
        shear with FEMMI's own forward. The DC mode IS formally observable to
        FEMMI (constant_mode_response > 0, KS exactly 0), but ~99.9% of that
        response sits in the square's corners and GROWS under refinement rather
        than converging, so it does not become practical recovery.

    KNOWN SENSITIVITY (documented, not yet explained): with several offset halos
    instead of one, FEMMI's DC-removed error roughly doubles and KS wins at all
    radii (interior 0.027 vs 0.013). It is not prior over-smoothing -- sweeping
    wiener_length over 0.2..1.0 moves it by <2%. Pass truth_kw={'halos': (...)} to
    reproduce. This is an open item, not a settled result.

    Returns a dict with maps, radial profiles, mean-kappa errors, and the error-
    vs-radius curves.
    """
    from .truth import independent_truth

    ops = square_ops(nx, half_width)
    nodes = np.array(ops.mesh.nodes)
    tkw = dict(truth_kw or {})
    if source == "nfw" and "halos" not in tkw:
        # One compact, centred halo: the isolated / far-field-zero regime FEMMI's
        # BEM explicitly assumes. See the multi-halo caveat in the docstring --
        # this default is the stated scope of the claim, not the best-looking case.
        tkw["halos"] = ((2.0e14, 4.0, (0.0, 0.0)),)
    kt, g1t, g2t = independent_truth(nodes, source=source, half_width=half_width,
                                     seed=seed, **tkw)

    rng = np.random.default_rng(seed)
    g1n = g1t + rng.normal(0, noise_std, len(g1t))
    g2n = g2t + rng.normal(0, noise_std, len(g2t))

    k_f = femmi_map(ops, g1n, g2n, noise_std, wiener_length=wiener_length,
                    weight=np.ones(len(nodes)))
    k_ks = ks_map(g1n, g2n, nodes, grid_size=ks_grid)

    # An NFW cusp reaches kappa > 1 at the node nearest the centre -- the strong-
    # lensing core, where the weak-shear approximation BOTH methods assume is
    # invalid and neither is being tested fairly. Drop those nodes from the
    # comparison, the same cut catalog.field_to_catalog makes with kappa_max.
    weak = np.ones(len(nodes), bool) if kappa_max is None else (kt < kappa_max)
    sel = lambda a: np.where(weak, a, np.nan)

    r, p_t = radial_profile(sel(kt), nodes)
    _, p_f = radial_profile(sel(k_f), nodes)
    _, p_ks = radial_profile(sel(k_ks), nodes)

    dm = lambda a: a - np.nanmean(a[weak])    # DC-removed -> isolates *shape* error
    rb, e_f = radial_profile(sel(np.abs(dm(k_f) - dm(kt))), nodes, nbins=nbins)
    _, e_ks = radial_profile(sel(np.abs(dm(k_ks) - dm(kt))), nodes, nbins=nbins)

    mt = float(np.nanmean(kt[weak]))
    mf = float(np.nanmean(k_f[weak])); mk = float(np.nanmean(k_ks[weak]))
    return dict(nodes=nodes, truth=kt, femmi=k_f, ks=k_ks, source=source,
                weak=weak, n_strong=int((~weak).sum()),
                r=r, prof_truth=p_t, prof_femmi=p_f, prof_ks=p_ks,
                r_err=rb, err_femmi_r=e_f, err_ks_r=e_ks, half_width=half_width,
                mean_truth=mt, mean_femmi=mf, mean_ks=mk,
                err_femmi=abs(mf - mt), err_ks=abs(mk - mt))


# --------------------------------------------------------------------------- #
# P0.2 -- boundary bias (correct far-field BC vs KS)
# --------------------------------------------------------------------------- #
def boundary_error_profile(nx=20, half_width=2.5, sigma=0.7, noise_std=0.02,
                           wiener_length=0.5, ks_grid=64, nbins=14, seed=0):
    """Reconstruction error |kappa_rec - kappa_true| vs distance-from-centre, FEMMI
    vs KS. FEMMI's exact exterior BC should keep the error flatter toward the edge
    where KS's periodic/Dirichlet truncation biases it."""
    ops = square_ops(nx, half_width)
    nodes = np.array(ops.mesh.nodes)
    kt, _, _ = analytic_gaussian_shear(nodes, sigma=sigma, amp=1.0)
    g1t, g2t = femmi_forward_shear(ops, kt)
    rng = np.random.default_rng(seed)
    g1n = g1t + rng.normal(0, noise_std, len(g1t))
    g2n = g2t + rng.normal(0, noise_std, len(g2t))
    k_f = femmi_map(ops, g1n, g2n, noise_std, wiener_length=wiener_length,
                    weight=np.ones(len(nodes)))
    k_ks = ks_map(g1n, g2n, nodes, grid_size=ks_grid)
    dm = lambda a: a - np.nanmean(a)           # DC-removed so this isolates *shape* error
    r, e_f = radial_profile(np.abs(dm(k_f) - dm(kt)), nodes, nbins=nbins)
    _, e_ks = radial_profile(np.abs(dm(k_ks) - dm(kt)), nodes, nbins=nbins)
    return dict(r=r, err_femmi=e_f, err_ks=e_ks, half_width=half_width)


# --------------------------------------------------------------------------- #
# P1.5 -- forward-operator convergence
# --------------------------------------------------------------------------- #
def manufactured_potential(nodes, R=1.5, c=1.0, p=6):
    """The compactly-supported manufactured lensing potential psi = c(1-(r/R)^2)^p
    for r<R, else 0. p=6 makes psi in C^5 (H^6), smooth enough that the O(h^4) P3
    rate is not regularity-limited; and psi=0 for r>=R, so if R<half_width it
    vanishes at the FEMMI boundary (no finite-vs-infinite-domain floor)."""
    r = np.hypot(nodes[:, 0], nodes[:, 1])
    u = 1.0 - (r / R) ** 2
    return np.where(r < R, c * u ** p, 0.0)


def manufactured_bump(nodes, R=1.5, c=1.0, p=6):
    """(kappa, g1, g2) consistent with manufactured_potential: kappa = 1/2 lap(psi),
    the shear is the traceless Hessian (FEMMI's sign convention). psi = c u^p,
    u = 1-(r/R)^2, gives
        lap(psi) = 4 c p [ (p-1) r^2/R^4 u^{p-2} - u^{p-1}/R^2 ]
        gamma_t  = 2 c p (p-1) r^2/R^4 u^{p-2}."""
    x, y = nodes[:, 0], nodes[:, 1]
    r = np.hypot(x, y); phi = np.arctan2(y, x)
    u = 1.0 - (r / R) ** 2
    inside = r < R
    lap = 4 * c * p * ((p - 1) * r**2 / R**4 * u**(p - 2) - u**(p - 1) / R**2)
    gt = 2 * c * p * (p - 1) * r**2 / R**4 * u**(p - 2)
    lap = np.where(inside, lap, 0.0); gt = np.where(inside, gt, 0.0)
    return 0.5 * lap, gt * np.cos(2 * phi), gt * np.sin(2 * phi)


def forward_convergence(nxs=(8, 12, 16, 24, 32, 40), half_width=2.5, R=1.5, p=6):
    """Convergence of FEMMI's recovered lensing POTENTIAL psi = F-solve(kappa) toward
    a COMPACTLY-SUPPORTED manufactured psi (manufactured_potential), at increasing
    resolution -- the validation of the forward operator F itself.

    psi is zero at the boundary (compact support), so there is no finite-vs-
    infinite-domain floor; the additive gauge (FEMMI pins one node) is removed
    before comparing. For P3 elements the theory rate is O(h^4) in L2, and that is
    what this measures (the shear is a well-defined post-hoc derivative of psi, not
    a property of F, so it is not the operator-validation quantity). Returns
    (h, err, fitted_order)."""
    import jax.numpy as jnp
    from .forward import DifferentiableForward

    hs, errs = [], []
    for nx in nxs:
        ops = square_ops(nx, half_width)
        nodes = np.array(ops.mesh.nodes)
        kt, _, _ = manufactured_bump(nodes, R=R, p=p)
        psi = np.asarray(DifferentiableForward(ops, lam_reg=1e-2).psi_from_kappa(jnp.asarray(kt)))
        pt = manufactured_potential(nodes, R=R, p=p)
        dm = lambda a: a - a.mean()            # remove the gauge (additive constant)
        errs.append(np.linalg.norm(dm(psi) - dm(pt)) / (np.linalg.norm(dm(pt)) + 1e-30))
        hs.append(2.0 * half_width / nx)
    hs, errs = np.array(hs), np.array(errs)
    from .convergence import fit_order
    order = fit_order(hs, errs, what="psi (forward operator)")
    return hs, errs, order


def shear_noise_amplification(nxs=(8, 12, 16, 24, 32, 40), half_width=2.5, R=1.5,
                              p=6, noise_std=1e-3, seed=0):
    """Why the theoretical O(h^2) shear rate is not reachable from real catalogs.

    Extracting the shear is a SECOND derivative of psi, and a second derivative
    amplifies any perturbation in psi by h^-2. So with a fixed noise level in psi
    -- which is what shape noise on a galaxy catalog leaves behind -- the total
    shear error behaves like

        err(h)  ~  C h^2   +   sigma / h^2,
                   \\_____/      \\_______/
                 discretisation   amplified noise

    which is U-shaped: it improves with refinement only down to an optimal h and
    then gets WORSE. This measures exactly that curve by perturbing the nodal psi
    with Gaussian noise of scale `noise_std` before extraction.

    Returns dict with h, err_clean, err_noisy (variational recovery in both cases)
    and the h at which the noisy error is minimised.
    """
    from .operators import build_recovered_shear_ops

    hs, e_clean, e_noisy = [], [], []
    for nx in nxs:
        ops = square_ops(nx, half_width)
        nodes = np.array(ops.mesh.nodes); M = ops.M
        psi = manufactured_potential(nodes, R=R, p=p)
        _, t1, t2 = manufactured_bump(nodes, R=R, p=p)
        rec = build_recovered_shear_ops(ops)

        rng = np.random.default_rng(seed)
        psi_n = psi + rng.normal(0, noise_std, len(psi))

        L2 = lambda a, b: np.sqrt(max(a @ (M @ a) + b @ (M @ b), 0.0))
        den = L2(t1, t2) + 1e-30
        c1, c2 = rec(psi);   e_clean.append(L2(c1 - t1, c2 - t2) / den)
        n1, n2 = rec(psi_n); e_noisy.append(L2(n1 - t1, n2 - t2) / den)
        hs.append(2.0 * half_width / nx)

    hs = np.array(hs); e_clean = np.array(e_clean); e_noisy = np.array(e_noisy)
    return dict(h=hs, err_clean=e_clean, err_noisy=e_noisy, noise_std=noise_std,
                h_opt=float(hs[int(np.argmin(e_noisy))]))


def shear_convergence(nxs=(16, 24, 32, 40), half_width=2.5, R=1.5, p=6):
    """Convergence of the SHEAR extracted from psi, nodal vs variationally
    recovered -- the P0.3 experiment.

    psi is the EXACT compactly-supported manufactured potential, so this isolates
    the shear-EXTRACTION operator from the solve, and both psi and grad psi vanish
    identically near the boundary -- which is what makes the variational recovery
    legitimate here (it drops a boundary term that would otherwise matter; see
    operators.RecoveredShear).

    Error is the true L2 norm (via the mass matrix), not a node sum, so the rate
    is a statement about the field and not about node placement.

    Theory (approximation theory for P3, MATH.md 18.3): the second derivative of a
    P3 field is O(h^2) in L2. Sampling it at the nodes -- exactly where the
    piecewise-cubic Hessian jumps -- pays a much larger constant and only reaches
    that rate asymptotically.

    Returns dict with h, err_nodal, err_recovered, the two fitted orders, and the
    per-interval local orders (which is where the asymptotic rate is visible).
    """
    from .operators import build_recovered_shear_ops

    hs, e_nod, e_rec = [], [], []
    for nx in nxs:
        ops = square_ops(nx, half_width)
        nodes = np.array(ops.mesh.nodes); M = ops.M
        psi = manufactured_potential(nodes, R=R, p=p)
        _, g1t, g2t = manufactured_bump(nodes, R=R, p=p)

        g1n = np.asarray(ops.S1 @ psi); g2n = np.asarray(ops.S2 @ psi)
        g1r, g2r = build_recovered_shear_ops(ops)(psi)

        L2 = lambda a, b: np.sqrt(max(a @ (M @ a) + b @ (M @ b), 0.0))
        den = L2(g1t, g2t) + 1e-30
        e_nod.append(L2(g1n - g1t, g2n - g2t) / den)
        e_rec.append(L2(g1r - g1t, g2r - g2t) / den)
        hs.append(2.0 * half_width / nx)

    hs = np.array(hs); e_nod = np.array(e_nod); e_rec = np.array(e_rec)
    from .convergence import fit_order, local_orders
    fit = lambda e: fit_order(hs, e, what="shear extraction")
    loc = lambda e: local_orders(hs, e)
    return dict(h=hs, err_nodal=e_nod, err_recovered=e_rec,
                order_nodal=fit(e_nod), order_recovered=fit(e_rec),
                local_nodal=loc(e_nod), local_recovered=loc(e_rec))


# --------------------------------------------------------------------------- #
# C^1 elements -- shear extraction without the second-derivative problem
# --------------------------------------------------------------------------- #
def manufactured_potential_derivs(p, dx=0, dy=0, R=1.5, c=1.0, pw=6):
    """The compactly-supported manufactured potential of manufactured_potential,
    plus its first and second partials -- the DOF data a C^1 element needs.

    psi = c u^pw, u = 1 - r^2/R^2 (and 0 outside r=R)."""
    x, y = float(p[0]), float(p[1])
    r2 = x * x + y * y
    if r2 >= R * R:
        return 0.0
    u = 1.0 - r2 / R**2
    if (dx, dy) == (0, 0):
        return c * u**pw
    if (dx, dy) == (1, 0):
        return c * pw * u**(pw - 1) * (-2 * x / R**2)
    if (dx, dy) == (0, 1):
        return c * pw * u**(pw - 1) * (-2 * y / R**2)
    if (dx, dy) == (2, 0):
        return c * pw * (4 * (pw - 1) * x * x / R**4 * u**(pw - 2)
                         - 2 / R**2 * u**(pw - 1))
    if (dx, dy) == (0, 2):
        return c * pw * (4 * (pw - 1) * y * y / R**4 * u**(pw - 2)
                         - 2 / R**2 * u**(pw - 1))
    if (dx, dy) == (1, 1):
        return 4 * c * pw * (pw - 1) * x * y / R**4 * u**(pw - 2)
    raise ValueError(f"unsupported derivative ({dx}, {dy})")


def element_shear_convergence(kind="argyris", nxs=(8, 12, 16, 24, 32),
                              half_width=2.5, R=1.5, pw=6, quad_order=5):
    """L2 convergence of the shear extracted from a C^1 element ('argyris' or
    'hct'), by interpolating the manufactured potential and differentiating twice.

    This is the element's own approximation power, isolated from any solve -- the
    direct counterpart of shear_convergence for P3, and measured the same way
    (true L2 norm, here by quadrature over every element).

    Expected rates: a degree-k element gives O(h^(k-1)) in the second derivative,
    so HCT (cubic) matches P3 at O(h^2) while ARGYRIS (quintic) reaches O(h^4).
    Returns dict with h, err, fitted order, and local orders.
    """
    from .elements import C1Space, structured_triangulation
    from .assembly import get_gauss_quadrature_triangle

    qp, qw = get_gauss_quadrature_triangle(order=quad_order)
    qp = np.asarray(qp); qw = np.asarray(qw)
    f = lambda p, dx=0, dy=0: manufactured_potential_derivs(p, dx, dy, R=R, pw=pw)

    hs, errs = [], []
    for nx in nxs:
        verts, tris = structured_triangulation(nx, half_width)
        S = C1Space(verts, tris, kind=kind)
        u = S.interpolate(f)
        num = den = 0.0
        for t in range(len(tris)):
            v = verts[tris[t]]
            area = abs(np.linalg.det(np.array([v[1] - v[0], v[2] - v[0]]))) / 2.0
            pts = v[0] + qp[:, 0:1] * (v[1] - v[0]) + qp[:, 1:2] * (v[2] - v[0])
            a1, a2 = S.eval_shear(u, t, pts)
            e1 = np.array([0.5 * (f(q, 2, 0) - f(q, 0, 2)) for q in pts])
            e2 = np.array([f(q, 1, 1) for q in pts])
            num += area * np.sum(qw * ((a1 - e1)**2 + (a2 - e2)**2))
            den += area * np.sum(qw * (e1**2 + e2**2))
        hs.append(2.0 * half_width / nx)
        errs.append(np.sqrt(num / (den + 1e-300)))

    hs = np.array(hs); errs = np.array(errs)
    from .convergence import fit_order, local_orders
    return dict(h=hs, err=errs, kind=kind,
                order=fit_order(hs, errs, what=f"{kind} shear"),
                local=local_orders(hs, errs))


# --------------------------------------------------------------------------- #
# P1.6 -- DC-mode identifiability (the mass-sheet degeneracy, at the operator level)
# --------------------------------------------------------------------------- #
def constant_mode_response(ops):
    """||F.1|| / ||1|| -- how strongly the forward responds to a UNIFORM mass sheet.
    FEMMI: O(sigma_max) (the far-field BC makes the DC mode observable); an
    FFT/Kaiser-Squires forward annihilates it exactly (-> 0)."""
    n = ops.n_nodes
    g1, g2 = femmi_forward_shear(ops, np.ones(n))
    return float(np.linalg.norm(np.concatenate([g1, g2])) / np.sqrt(n))


def ks_constant_mode_response(grid_size=32):
    """The KS/FFT forward's response to a uniform mass sheet -- identically zero
    (the DC mode is in its null space)."""
    kx = np.fft.fftfreq(grid_size)[None, :]
    ky = np.fft.fftfreq(grid_size)[:, None]
    k2 = kx**2 + ky**2; k2[0, 0] = 1.0
    d1 = (kx**2 - ky**2) / k2; d2 = 2 * kx * ky / k2
    d1[0, 0] = 0.0; d2[0, 0] = 0.0
    kh = np.fft.fft2(np.ones((grid_size, grid_size)))
    g1 = np.fft.ifft2(d1 * kh).real; g2 = np.fft.ifft2(d2 * kh).real
    return float(np.hypot(g1, g2).std())
