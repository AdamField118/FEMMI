"""
tests/test_density.py
Catalog-native reconstruction and the source-density comparison (femmi.density).

This is the experiment behind the candidate paper claim, so the tests are about
the things that would invalidate it rather than the headline number:

  * the density <-> count conversion is exact and round-trips, since the whole
    experiment is parameterised by n_eff [gal/arcmin^2] and every reported
    density is derived from a count through it;
  * catalog geometry really does put a vertex on every galaxy, with the boundary
    made only of ring vertices that carry no data;
  * the mesh-quality diagnostic actually reports the sliver problem, because the
    claim is not quotable without it;
  * accuracy improves with source density for every method (a curve that did not
    would mean the comparison is noise);
  * the sweep really is multi-seed and the spread survives into the reported
    row, because a single realisation of this experiment is not reproducible in
    the direction the claim depends on;
  * ring vertices are excluded from the data term -- weighting them in would let
    the reconstruction fit shear that was never observed.

Run:
    python -m pytest tests/test_density.py -v
"""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from femmi.elements import C1Space, catalog_triangulation
from femmi.c1_coupling import boundary_loop
from femmi.density import (mesh_quality, sample_catalog, argyris_catalog_run,
                           ks_catalog_run, density_sweep, average_over_seeds,
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


def test_runs_report_the_density_they_actually_used():
    """Argyris drops the guard ring from the data term, so its EFFECTIVE density
    is below the nominal one. Reporting the nominal number would overstate how
    much data it was given."""
    drawn = n_gal_for_density(10.0, 3.0)
    a = argyris_catalog_run(10.0, noise_std=0.05, seed=0, radius=3.0)
    assert a["n_eff_nominal"] == 10.0
    assert a["n_gal"] <= drawn                 # dedup and ring clipping only drop
    assert a["n_eff"] == pytest.approx(density_for_n_gal(a["n_gal"], 3.0))
    k = ks_catalog_run(10.0, noise_std=0.05, seed=0, radius=3.0)
    assert k["n_gal"] == drawn                 # KS uses every galaxy
    assert abs(k["n_eff"] - 10.0) < 0.05       # to within one whole galaxy


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


def test_accuracy_improves_with_source_density():
    """The load-bearing property of the sweep: if error did not fall with density
    for every method, the comparison would be measuring noise."""
    rows = density_sweep(n_effs=(5.0, 15.0), noise_std=0.05, radius=3.0,
                         seeds=(0,), methods=("argyris", "ks"), verbose=False)
    for name in ("Argyris", "Kaiser"):
        c = sorted([r for r in rows if r["method"].startswith(name)],
                   key=lambda z: z["n_eff_nominal"])
        assert len(c) == 2
        assert c[1]["n_eff"] > c[0]["n_eff"]
        assert c[1]["shape_l2"] < c[0]["shape_l2"]


def test_seeds_are_averaged_with_a_standard_error():
    """A single realisation of this experiment is not reproducible in the
    direction that matters -- the equivalence factor swings by a factor of five
    across seeds -- so the sweep is multi-seed and the spread must survive into
    the reported row rather than being averaged away silently."""
    raw = density_sweep(n_effs=(5.0,), noise_std=0.05, radius=3.0,
                        seeds=(0, 1), methods=("ks",), verbose=False)
    assert [r["seed"] for r in raw] == [0, 1]
    assert raw[0]["shape_l2"] != raw[1]["shape_l2"]      # different catalogs

    avg = average_over_seeds(raw)
    assert len(avg) == 1
    a = avg[0]
    assert a["n_seeds"] == 2 and a["seeds"] == [0, 1]
    assert a["shape_l2"] == pytest.approx(
        0.5 * (raw[0]["shape_l2"] + raw[1]["shape_l2"]))
    # standard error of the mean, not the sample spread
    assert a["shape_l2_std"] == pytest.approx(
        0.5 * abs(raw[0]["shape_l2"] - raw[1]["shape_l2"]))
    assert "+/-" in to_table(avg)


def test_argyris_beats_ks_on_the_same_catalog():
    """Same galaxies, same noise, same truth, at a DES-like source density."""
    a = argyris_catalog_run(10.0, noise_std=0.05, seed=0)
    k = ks_catalog_run(10.0, noise_std=0.05, seed=0)
    assert a["shape_l2"] < k["shape_l2"]


def test_table_renders_and_survives_a_failing_cell():
    rows = density_sweep(n_effs=(5.0,), seeds=(0,), methods=("ks",), verbose=False)
    rows.append(dict(method="broken", n_eff_nominal=1.0, n_gal=1, error="boom"))
    txt = to_table(rows)
    assert "Kaiser" in txt and "boom" in txt
    assert "n_eff" in txt and "/arcmin2" in txt
    # a failing cell must not poison the average either
    assert [r["method"] for r in average_over_seeds(rows)] == ["Kaiser-Squires"]


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
