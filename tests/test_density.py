"""Catalogue geometry, source counts and mesh quality contracts."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.elements import C1Space, catalog_triangulation
from femmi.c1_coupling import boundary_loop
from femmi.density import (mesh_quality, sample_catalog,
                           average_over_seeds,
                           to_table, n_gal_for_density, density_for_n_gal,
                           SURVEY_NEFF)

pytest.importorskip("galsim", reason="density sweep scores against GalSim truth")


def test_density_count_conversion_round_trips():
    """n_eff is the reported quantity but a catalog is drawn as a count, so the
    conversion sits between the experiment and every number it prints."""
    for n_eff in (5.0, 10.0, 27.0, 30.0):
        for radius in (1.5, 3.0, 5.0):
            n = n_gal_for_density(n_eff, radius)
            assert isinstance(n, int)
            # exact up to the rounding to a whole galaxy
            assert abs(density_for_n_gal(n, radius) - n_eff) < 1.0 / (np.pi * radius**2)
    assert n_gal_for_density(10.0, 3.0) == round(10.0 * np.pi * 9.0)


def test_survey_densities_are_plausible():
    """These are quoted next to the measured densities, so a typo here would
    silently mis-scale the claim."""
    assert set(SURVEY_NEFF) >= {"DES Y3", "HSC Y3", "Euclid"}
    assert all(1.0 < v < 60.0 for v in SURVEY_NEFF.values())
    assert SURVEY_NEFF["DES Y3"] < SURVEY_NEFF["HSC Y3"] < SURVEY_NEFF["Euclid"]




def test_catalog_triangulation_puts_a_vertex_on_every_galaxy():
    x, y = sample_catalog(300, radius=2.0, seed=0)
    v, t, ring, gal_index = catalog_triangulation(x, y)

    assert ring.sum() > 0 and (~ring).sum() > 0
    assert np.all(gal_index[gal_index >= 0] < (~ring).sum())
    # every galaxy that survived dedup/clipping maps to a real vertex, and that
    # vertex is where the galaxy is
    mapped = gal_index >= 0
    assert mapped.mean() > 0.95
    got = v[gal_index[mapped]]
    want = np.stack([x, y], 1)[mapped]
    assert np.abs(got - want).max() < 0.05        # dedup tolerance


def test_boundary_is_made_only_of_ring_vertices():
    """If a galaxy landed on the boundary loop, the BEM trace would be sampling a
    data point and the far-field condition would be imposed on real signal."""
    x, y = sample_catalog(400, radius=2.0, seed=1)
    v, t, ring, _ = catalog_triangulation(x, y)
    S = C1Space(v, t, kind="argyris")
    loop = boundary_loop(S)
    assert set(loop.tolist()) <= set(np.where(ring)[0].tolist())


def test_mesh_quality_reports_the_sliver_problem():
    """Random positions make slivers and Argyris inverts a 21x21 Vandermonde per
    element. The diagnostic must surface that, since the density claim is not
    quotable without it."""
    x, y = sample_catalog(400, radius=2.0, seed=0)
    v, t, ring, _ = catalog_triangulation(x, y)
    q = mesh_quality(C1Space(v, t, kind="argyris"))

    assert q["n_elements"] == len(t)
    assert 0.0 < q["min_angle"] < q["median_angle"]
    assert q["median_cond"] < 1e8                  # the typical element is fine
    assert q["max_cond"] > q["median_cond"]        # and the worst one is not


def test_ring_vertices_carry_no_data_weight():
    """The reconstruction must not fit shear at the guard ring."""
    from femmi.c1_inverse import C1MAPReconstructor
    x, y = sample_catalog(200, radius=2.0, seed=0)
    v, t, ring, _ = catalog_triangulation(x, y)
    S = C1Space(v, t, kind="argyris")
    rec = C1MAPReconstructor(S, lam=0.3, data_weight=(~ring).astype(float))
    assert np.all(rec.w[ring] == 0.0)
    assert np.all(rec.w[~ring] == 1.0)










if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
