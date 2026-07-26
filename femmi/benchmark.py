"""
femmi/benchmark.py
A matrix runner over {element} x {prior} x {reconstruction method}.

The design goal for FEMMI is to carry every reasonable option and then FIND the
best combination empirically rather than arguing about it. That only works if
running the whole grid is one command and the results are directly comparable, so
this module fixes the comparison protocol:

  * the truth is INDEPENDENT (femmi.truth -- analytic GalSim NFW, a tapered
    log-normal field, or a MassiveNuS map), never either method's own forward, so
    no configuration is scored on an inverse crime;
  * the FIELD TYPE is a knob, because it decides the answer. A single smooth NFW
    halo is exactly the case a Gaussian/Wiener prior is built to win, so ranking
    priors on it says little; source='lognormal' gives the peaked, non-Gaussian
    field where TV/sparsity/learned priors are supposed to pay off. Run both
    before concluding anything about a prior;
  * the same truth, the same noise realisation and the same galaxy positions are
    reused across every configuration in a sweep;
  * cost is reported as GLOBAL DOFS and wall-clock next to accuracy, because
    "accuracy per DOF" is the only fair axis when comparing elements -- Argyris
    looks expensive per element (21 DOF) and is not (about 1.04x P3 globally).

Metrics per run: relative L2 error, DC-removed relative L2 error (shape only,
which is what survives the mass-sheet limitation of MATH.md 6.3a), mean-kappa
error, DOFs, and seconds.

    from femmi.benchmark import sweep, DEFAULT_GRID
    df = sweep(DEFAULT_GRID, nx=14)
"""

from __future__ import annotations
import itertools
import time
import numpy as np


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _metrics(kappa_rec, kappa_true, weak=None):
    k = np.asarray(kappa_rec, float); t = np.asarray(kappa_true, float)
    if weak is None:
        weak = np.ones(len(t), bool)
    k = k[weak]; t = t[weak]
    dm = lambda a: a - np.nanmean(a)
    den = np.linalg.norm(t) + 1e-300
    return dict(
        rel_l2=float(np.linalg.norm(k - t) / den),
        rel_l2_dc_removed=float(np.linalg.norm(dm(k) - dm(t))
                                / (np.linalg.norm(dm(t)) + 1e-300)),
        mean_err=float(abs(np.nanmean(k) - np.nanmean(t))),
        mean_truth=float(np.nanmean(t)),
    )


# --------------------------------------------------------------------------- #
# configurations
# --------------------------------------------------------------------------- #
DEFAULT_GRID = dict(
    element=["p3"],                       # "argyris" also works (C^1, circular)
    prior=["wiener", "tv", "sparse", "maxent"],
    method=["map"],                       # + "rto", "langevin" for sampling runs
)


def _run_one(element, prior, method, *, nx, half_width, noise_std, seed,
             truth_kw, wiener_length, ks_grid, source):
    """One cell of the grid. Returns a metrics dict (or an 'error' entry)."""
    from .experiments import square_ops, femmi_map, ks_map
    from .truth import independent_truth

    t0 = time.perf_counter()

    if element == "argyris":
        # Full C^1 path: circular domain, BEM far-field coupling, MAP with the
        # adjoint, and Wiener / TV / sparsity priors defined on the C^1 DOF
        # vector (c1_inverse.make_c1_prior). max-entropy is still P3-only.
        if prior not in ("wiener", "-", "tv", "sparse"):
            return dict(element=element, prior=prior, method=method,
                        error=f"prior {prior!r} not yet defined on the C^1 DOF vector")
        if method != "map":
            return dict(element=element, prior=prior, method=method,
                        error=f"method {method!r} not available for C^1 yet")
        from .elements import C1Space, circular_triangulation
        from .c1_inverse import C1MAPReconstructor

        # match the square mesh's DOF budget roughly, so the grid rows compare
        v, t = circular_triangulation(max(12, 3 * nx), radius=half_width)
        S = C1Space(v, t, kind="argyris")
        kt, g1t, g2t = independent_truth(v, source=source, half_width=half_width,
                                         seed=seed, **truth_kw)
        rng = np.random.default_rng(seed)
        g1 = g1t + rng.normal(0, noise_std, len(g1t))
        g2 = g2t + rng.normal(0, noise_std, len(g2t))
        rec = C1MAPReconstructor(S, lam=0.3, wiener_length=wiener_length,
                                 prior=(None if prior in ("wiener", "-") else prior))
        k, _ = rec.reconstruct(g1, g2)
        out = dict(element=element, prior=prior, method=method, source=source,
                   dofs=int(S.n_dofs), n_obs=int(S.n_vertices),
                   seconds=time.perf_counter() - t0)
        out.update(_metrics(rec.kappa_at_vertices(k), kt, kt < 1.0))
        return out

    if element != "p3":
        return dict(element=element, prior=prior, method=method,
                    error=f"unknown element {element!r} (expected 'p3' or 'argyris')")

    ops = square_ops(nx, half_width)
    nodes = np.array(ops.mesh.nodes)
    kt, g1t, g2t = independent_truth(nodes, source=source, half_width=half_width,
                                     seed=seed, **truth_kw)
    rng = np.random.default_rng(seed)
    g1 = g1t + rng.normal(0, noise_std, len(g1t))
    g2 = g2t + rng.normal(0, noise_std, len(g2t))
    weak = kt < 1.0                                  # drop the strong-lensing core

    if method == "ks":
        rec = ks_map(g1, g2, nodes, grid_size=ks_grid)
    elif method == "map":
        rec = femmi_map(ops, g1, g2, noise_std, wiener_length=wiener_length,
                        weight=np.ones(len(nodes)),
                        prior=(None if prior == "wiener" else prior))
    else:
        return dict(element=element, prior=prior, method=method,
                    error=f"unknown method {method!r}")

    out = dict(element=element, prior=prior, method=method, source=source,
               dofs=int(ops.n_nodes), n_obs=int(len(nodes)),
               seconds=time.perf_counter() - t0)
    out.update(_metrics(rec, kt, weak))
    return out


def sweep(grid=None, nx=14, half_width=2.5, noise_std=0.02, seed=0,
          truth_kw=None, wiener_length=1.0, ks_grid=48, include_ks=True,
          source="nfw", verbose=True):
    """Run every combination in `grid` on the SAME independent truth and noise.

    grid: dict of lists, keys 'element', 'prior', 'method'. Returns a list of
    result dicts; use `to_table` to print it.
    """
    grid = grid or DEFAULT_GRID
    if truth_kw is None:
        truth_kw = {"halos": ((2.0e14, 4.0, (0.0, 0.0)),)} if source == "nfw" else {}

    combos = list(itertools.product(grid.get("element", ["p3"]),
                                    grid.get("prior", ["wiener"]),
                                    grid.get("method", ["map"])))
    if include_ks:
        combos.append(("p3", "-", "ks"))            # the baseline every run needs

    rows = []
    for element, prior, method in combos:
        if verbose:
            print(f"  running {element:8s} {prior:8s} {method:8s} ...", flush=True)
        try:
            rows.append(_run_one(element, prior, method, nx=nx,
                                 half_width=half_width, noise_std=noise_std,
                                 seed=seed, truth_kw=truth_kw, source=source,
                                 wiener_length=wiener_length, ks_grid=ks_grid))
        except Exception as exc:                     # one bad cell must not kill the sweep
            rows.append(dict(element=element, prior=prior, method=method,
                             error=f"{type(exc).__name__}: {exc}"))
    return rows


def to_table(rows, sort_by="rel_l2_dc_removed"):
    """Render sweep results as a fixed-width table, best first."""
    ok = [r for r in rows if "error" not in r]
    bad = [r for r in rows if "error" in r]
    ok.sort(key=lambda r: r.get(sort_by, np.inf))

    hdr = (f"{'element':<9}{'prior':<9}{'method':<8}{'rel L2':>9}"
           f"{'shape L2':>10}{'mean err':>10}{'DOFs':>8}{'n_obs':>8}{'sec':>8}")
    lines = [hdr, "-" * len(hdr)]
    for r in ok:
        lines.append(f"{r['element']:<9}{r['prior']:<9}{r['method']:<8}"
                     f"{r['rel_l2']:>9.4f}{r['rel_l2_dc_removed']:>10.4f}"
                     f"{r['mean_err']:>10.4f}{r['dofs']:>8d}"
                     f"{r.get('n_obs', r['dofs']):>8d}{r['seconds']:>8.1f}")
    for r in bad:
        lines.append(f"{r['element']:<9}{r['prior']:<9}{r['method']:<8}  -- {r['error']}")
    return "\n".join(lines)
