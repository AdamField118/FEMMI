"""
tests/test_density_extensions.py
The pieces added to make the density claim testable beyond its first setting:
catalog geometry (masks, clustering, weights), alternative truth fields, the HCT
arm, Vandermonde equilibration, KS calibration, and paired statistics.

Each of these exists to ATTACK the claim rather than decorate it, so the tests
are about the properties that would let a bug flatter the result.

Run:
    python -m pytest tests/test_density_extensions.py -v
"""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.elements import (C1Space, catalog_triangulation, equilibrated_inverse,
                            structured_triangulation)
from femmi.c1_inverse import (shear_operators, shear_selection_operators,
                              shear_evaluation_operators)
from femmi.density import (sample_catalog, mesh_quality, _truth_at,
                           ks_params_for_density, KS_CALIBRATION,
                           paired_comparison, paired_table, average_over_seeds)


# ------------------------------------------------------- reproducibility ---

def test_unmasked_sampling_is_bit_identical_to_the_original_draw():
    """LOAD-BEARING. Every number in MATH.md 18.3i/18.3j is tied to the exact
    catalogs `sample_catalog(n, radius, seed)` produces. Adding mask support
    tempted a rewrite to rejection sampling for ALL cases, which consumes the RNG
    stream differently and silently changes every catalog for a given seed -- it
    did, and was caught only because a spot-check disagreed with a committed
    number. The unmasked path must stay bit-identical forever."""
    for n, radius, seed in ((141, 3.0, 0), (848, 3.0, 5), (200, 2.0, 3)):
        rng = np.random.default_rng(seed)
        th = rng.uniform(0, 2 * np.pi, n)
        rr = radius * np.sqrt(rng.uniform(0, 1, n))
        ex, ey = rr * np.cos(th), rr * np.sin(th)
        gx, gy = sample_catalog(n, radius=radius, seed=seed)
        assert np.array_equal(ex, gx) and np.array_equal(ey, gy)


def test_sampling_is_deterministic_in_the_seed():
    a = sample_catalog(300, radius=3.0, seed=7)
    b = sample_catalog(300, radius=3.0, seed=7)
    c = sample_catalog(300, radius=3.0, seed=8)
    assert np.array_equal(a[0], b[0])
    assert not np.array_equal(a[0], c[0])


# ------------------------------------------------------- survey geometry ---

def test_masks_are_empty_and_the_count_is_preserved():
    """A masked survey with a target n_eff is DENSER in what survives, so the
    count is held fixed rather than the area -- otherwise masking would silently
    also be a density change and the two effects could not be separated."""
    masks = ((1.0, 0.0, 0.7), (-1.2, 1.0, 0.5))
    x, y = sample_catalog(500, radius=3.0, seed=0, masks=masks)
    assert len(x) == 500
    for cx, cy, cr in masks:
        assert np.hypot(x - cx, y - cy).min() > cr
    assert np.hypot(x, y).max() <= 3.0


def test_clustering_makes_the_mesh_worse_not_better():
    """Clustering is the PESSIMISTIC case: tight groups make sliver triangles.
    If it improved the mesh the knob would be measuring something else.

    Judged on the MEDIAN angle, not the minimum. The minimum is a single extreme
    order statistic over ~840 elements and is wildly noisy -- measured
    0.699 -> 0.190 -> 1.477 degrees at clustering 0 / 0.2 / 0.4, not even
    monotone. The median moves cleanly (31.03 -> 29.14 -> 28.12) because it
    describes the bulk of the mesh rather than its single worst triangle."""
    q = []
    for c in (0.0, 0.4):
        x, y = sample_catalog(400, radius=3.0, seed=0, clustering=c)
        v, t, _, _ = catalog_triangulation(x, y)
        q.append(mesh_quality(C1Space(v, t, kind="argyris")))
    assert q[1]["median_angle"] < q[0]["median_angle"]


def test_weights_are_positive_and_mean_one():
    x, y, w = sample_catalog(300, radius=3.0, seed=0, return_weights=True,
                             weight_scatter=0.3)
    assert len(w) == 300 and np.all(w > 0)
    assert w.mean() == pytest.approx(1.0)
    x, y, w1 = sample_catalog(300, radius=3.0, seed=0, return_weights=True)
    assert np.all(w1 == 1.0)                       # no scatter -> equal weights


# ------------------------------------------------------------------ truth ---

def test_truth_selector_routes_and_is_shared():
    """All three arms must score against the SAME field; the helper exists so
    there is one call site instead of three that can drift."""
    pytest.importorskip("galsim")
    pts = np.random.default_rng(0).normal(scale=1.0, size=(50, 2))
    halos = ((2.0e14, 4.0, (0.0, 0.0)),)
    k1, a1, _ = _truth_at(pts, "nfw", halos, 3.0, 0, None)
    k2, a2, _ = _truth_at(pts, "nfw", halos, 3.0, 0, None)
    assert np.array_equal(k1, k2) and np.array_equal(a1, a2)
    kl, _, _ = _truth_at(pts, "lognormal", halos, 3.0, 0, None)
    assert kl.shape == k1.shape
    assert not np.allclose(kl, k1)                 # genuinely a different field


def test_unknown_truth_is_rejected():
    with pytest.raises(ValueError, match="unknown independent-truth source"):
        _truth_at(np.zeros((4, 2)), "not-a-field", (), 3.0, 0, None)


# ---------------------------------------------------------------- HCT arm ---

def _quadratic(p, dx=0, dy=0):
    p = np.atleast_2d(np.asarray(p, float))
    table = {(0, 0): p[:, 0] ** 2 - p[:, 1] ** 2,
             (1, 0): 2 * p[:, 0], (0, 1): -2 * p[:, 1],
             (2, 0): np.full(len(p), 2.0), (1, 1): np.zeros(len(p)),
             (0, 2): np.full(len(p), -2.0)}
    r = table.get((dx, dy), np.zeros(len(p)))
    return r if len(r) > 1 else float(r[0])


@pytest.mark.parametrize("kind", ["argyris", "hct"])
def test_shear_operators_are_exact_on_a_quadratic(kind):
    """psi = x^2 - y^2 has constant Hessian, so gamma1 = 2 and gamma2 = 0
    everywhere. Both elements contain P2, so both must be EXACT -- this is what
    validates the HCT evaluation operator against the Argyris selection one on a
    case where the answer is known independently."""
    v, t = structured_triangulation(4, 2.0)
    S = C1Space(v, t, kind=kind)
    S1, S2 = shear_operators(S)
    psi = S.interpolate(_quadratic)
    assert np.abs(S1 @ psi - 2.0).max() < 1e-10
    assert np.abs(S2 @ psi).max() < 1e-10


def test_shear_operator_dispatch():
    v, t = structured_triangulation(3, 2.0)
    arg = C1Space(v, t, kind="argyris")
    hct = C1Space(v, t, kind="hct")
    # Argyris takes the selection path: 2 and 1 entries per row, no quadrature
    S1, S2 = shear_operators(arg)
    assert S1.nnz == 2 * arg.n_vertices and S2.nnz == arg.n_vertices
    # HCT has no Hessian DOF, so selection must refuse rather than mis-index
    with pytest.raises(ValueError, match="Hessian vertex DOFs"):
        shear_selection_operators(hct)
    H1, _ = shear_evaluation_operators(hct)
    assert H1.shape == (hct.n_vertices, hct.n_dofs)
    assert H1.nnz > 2 * hct.n_vertices             # genuinely evaluated


def test_hct_is_cheaper_per_element_than_argyris():
    """The reason HCT is worth testing at all: 12 DOF against 21."""
    v, t = structured_triangulation(4, 2.0)
    assert C1Space(v, t, kind="hct").n_dofs < C1Space(v, t, kind="argyris").n_dofs


# --------------------------------------------------------- equilibration ---

def test_equilibrated_inverse_is_an_exact_inverse():
    """The scaling is an algebraic identity, not an approximation. If it ever
    became approximate the basis would silently stop being interpolatory.

    Measured RELATIVELY: these matrices are deliberately unbalanced across 13
    orders of magnitude, so an absolute tolerance on V @ Vinv - I is meaningless.
    The scale-free statement is that it agrees with a plain inverse."""
    rng = np.random.default_rng(0)
    for _ in range(10):
        V = rng.normal(size=(12, 12))
        V[3] *= 1e-7
        V[:, 5] *= 1e6
        Vinv, _, _ = equilibrated_inverse(V)
        assert np.allclose(Vinv, np.linalg.inv(V), rtol=1e-6, atol=0.0)


def test_equilibration_improves_conditioning_on_unbalanced_matrices():
    rng = np.random.default_rng(1)
    V = rng.normal(size=(21, 21))
    V[0] *= 1e-8
    V[:, 1] *= 1e8
    _, c_raw, c_eq = equilibrated_inverse(V)
    assert c_eq < c_raw / 100.0


def test_mesh_quality_reports_both_conditioning_numbers():
    """They mean different things -- raw is how bad the geometry is, eq is how
    much of that reaches the arithmetic -- so both are reported."""
    x, y = sample_catalog(300, radius=3.0, seed=0)
    v, t, _, _ = catalog_triangulation(x, y)
    q = mesh_quality(C1Space(v, t, kind="argyris"))
    assert q["max_cond_eq"] <= q["max_cond"]
    assert q["n_ill_eq"] <= q["n_ill"]


def test_mesh_quality_handles_an_element_with_no_condition_number():
    """HCT solves a least-squares system instead of inverting a square
    Vandermonde, so it reports no condition number at all. That must come back as
    nan without a warning storm, not crash."""
    x, y = sample_catalog(200, radius=3.0, seed=0)
    v, t, _, _ = catalog_triangulation(x, y)
    q = mesh_quality(C1Space(v, t, kind="hct"))
    assert np.isnan(q["max_cond"])
    assert q["n_elements"] == len(t) and q["min_angle"] > 0


# --------------------------------------------------------- KS calibration ---

def test_ks_params_rise_with_density_and_are_interpolated():
    """Grid resolution must increase with source density -- more galaxies support
    finer pixels before shot noise dominates. A flat rule would be the untuned
    default this replaced."""
    gs = [ks_params_for_density(n)[0] for n in (5.0, 10.0, 20.0, 30.0)]
    assert gs == [KS_CALIBRATION[n][0] for n in (5.0, 10.0, 20.0, 30.0)]
    assert gs[-1] > gs[0]
    assert gs[1] <= ks_params_for_density(15.0)[0] <= gs[2]
    assert ks_params_for_density(1.0)[0] >= 4        # clamped, still usable
    assert ks_params_for_density(500.0)[0] >= gs[-1]


def test_ks_default_is_the_calibration_not_the_old_pin():
    """Regression for the fairness bug: KS ran at a hardcoded (32, 1.0) that was
    never measured, costing it 15-38% and flattering every comparison against
    it."""
    pytest.importorskip("galsim")
    from femmi.density import ks_catalog_run
    r = ks_catalog_run(20.0, seed=0)
    assert r["ks_grid_size"] == 16 and r["ks_smoothing_px"] == 1.0
    old = ks_catalog_run(20.0, seed=0, grid_size=32, smoothing_px=1.0)
    assert r["shape_l2"] < old["shape_l2"]


# ------------------------------------------------------ paired statistics ---

def _paired_rows(true_gap=0.02, shared=0.10, indep=0.005, n=6, seed=0):
    """Two arms on shared catalogs: large COMMON scatter, small true gap."""
    rng = np.random.default_rng(seed)
    rows = []
    for ne in (5.0, 10.0):
        for s in range(n):
            base = dict(n_eff_nominal=ne, seed=s, radius_arcmin=3.0, n_eff=ne,
                        n_gal=100, dofs=10, rel_l2=0.6, mean_err=0.04, seconds=1.0)
            c = rng.normal(0, shared)      # catalog effect, common to both arms
            rows.append(dict(base, method="Argyris (catalog)",
                             shape_l2=0.50 + c + rng.normal(0, indep)))
            rows.append(dict(base, method="P3 (catalog)",
                             shape_l2=0.50 + true_gap + c + rng.normal(0, indep)))
    return rows


def test_paired_beats_marginal_when_catalog_scatter_is_shared():
    """THE REASON THIS EXISTS. With scatter common to both arms 5x the true gap,
    a marginal test sees nothing and a paired test sees it clearly. Every sigma
    quoted in MATH.md 18.3i/ja/jb was computed the marginal way."""
    rows = _paired_rows()
    for st in paired_comparison(rows, "Argyris", "P3"):
        assert st["t"] > 5.0
        assert st["wins"] == st["n_pairs"]

    for ne in (5.0, 10.0):
        a = np.array([r["shape_l2"] for r in rows
                      if r["method"].startswith("Argyris") and r["n_eff_nominal"] == ne])
        b = np.array([r["shape_l2"] for r in rows
                      if r["method"].startswith("P3") and r["n_eff_nominal"] == ne])
        marginal = (b.mean() - a.mean()) / np.sqrt(a.var(ddof=1) / len(a)
                                                   + b.var(ddof=1) / len(b))
        assert marginal < 1.0          # the same data, and it looks like nothing


def test_paired_sign_convention_and_pairing():
    """Positive mean_diff means the FIRST method is better, and only catalogs
    present for BOTH arms are compared."""
    rows = _paired_rows()
    st = paired_comparison(rows, "Argyris", "P3")[0]
    assert st["mean_diff"] > 0
    assert st["n_pairs"] == 6 and sorted(st["seeds"]) == list(range(6))

    rows2 = [r for r in rows
             if not (r["method"].startswith("P3") and r["seed"] == 3)]
    st2 = paired_comparison(rows2, "Argyris", "P3")[0]
    assert st2["n_pairs"] == 5 and 3 not in st2["seeds"]


def test_paired_requires_per_seed_rows():
    """Averaged rows carry no `seed`, so pairing them silently would compare one
    aggregate against another and report a meaningless t."""
    with pytest.raises(ValueError, match="per-seed rows"):
        paired_comparison(average_over_seeds(_paired_rows()), "Argyris", "P3")


def test_paired_table_renders():
    txt = paired_table(_paired_rows(), "Argyris", "P3")
    assert "paired t" in txt and "wins" in txt and "positive = Argyris better" in txt


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
