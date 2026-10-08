"""Executable FITS-to-FITS survey contracts, including the pinned SMPy behavior."""

import json
from dataclasses import replace
import numpy as np
import pytest

fits = pytest.importorskip("astropy.io.fits")
from astropy.table import Table
from astropy.wcs import WCS
from femmi import map_mass, read_fits_catalog, FEMMapper
from femmi.survey import _prepare, _sky_to_xy, _coverage


def catalogue(path, n=24, wrap=False, unit="deg", pixel=False):
    rng = np.random.default_rng(34)
    x, y = rng.uniform(-0.7, 0.7, (2, n))
    ra, dec = (
        ((359.999 if wrap else 35) + x / (60 * np.cos(np.deg2rad(-30)))) % 360,
        -30 + y / 60,
    )
    t = Table(
        dict(
            ra=ra,
            dec=dec,
            g1=0.02 * x + 0.01 * y,
            g2=0.03 * y - 0.01 * x,
            weight=np.linspace(0.5, 2, n),
            ID=np.arange(n) + 1000,
            Z=np.linspace(0.6, 2, n),
            FLAGS=np.zeros(n, dtype=np.int32),
            R11=np.full(n, 0.7),
            R22=np.full(n, 0.8),
        )
    )
    if unit == "rad":
        t["ra"] = np.deg2rad(ra)
        t["dec"] = np.deg2rad(dec)
    t["ra"].unit = t["dec"].unit = unit
    t["X_IMAGE"] = x * 100 + 51
    t["Y_IMAGE"] = y * 100 + 71
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(np.zeros((2, 2))),
            fits.BinTableHDU(t, name="SHAPES"),
        ]
    ).writeto(path)
    return t


def options(path, **kwargs):
    return dict(
        data=path,
        input_hdu="SHAPES",
        lam=0.3,
        length=0.6,
        pixel_scale=0.2,
        save_plots=False,
        **kwargs,
    )


@pytest.mark.parametrize("method", ["p3", "argyris", "hct"])
def test_fits_to_fits_each_estimator_and_registration(tmp_path, method):
    path = tmp_path / "catalog.fits"
    table = catalogue(path)
    out = map_mass(
        **options(
            path,
            method=method,
            mode=["E", "B"],
            weight_col="weight",
            id_col="ID",
            z_col="Z",
            create_counts_map=True,
            save_fits=True,
            output_dir=tmp_path,
        )
    )
    c = out["catalogue"]
    np.testing.assert_array_equal(c.g2, -table["g2"])
    np.testing.assert_array_equal(c.weight, table["weight"])
    assert not out["metadata"]["catalogue"]["redshifts_used_in_operator"]
    base = tmp_path / method / f"femmi_output_{method}"
    for m in ("E", "B"):
        with fits.open(str(base) + f"_{m.lower()}_mode.fits", checksum=True) as hdul:
            assert hdul[0].verify_checksum() == 1
            wcs = WCS(hdul[0].header)
            yy, xx = np.indices(hdul[0].data.shape)
            ra, dec = wcs.all_pix2world(xx, yy, 0)
            x, y = _sky_to_xy(ra, dec, c.center, "smpy")
            expected = (
                out["reconstructions"][m]
                .evaluate(np.c_[x.ravel(), y.ravel()])
                .reshape(xx.shape)
            )
            expected[out["coverage"] == 0] = np.nan
            np.testing.assert_allclose(
                hdul[0].data, expected, atol=1e-10, equal_nan=True
            )
            np.testing.assert_array_equal(hdul["COVERAGE"].data, out["coverage"])
            assert hdul[0].header["BUNIT"] == "1"
        assert out["metadata"]["diagnostics"][m]["relative_residual"] < 1e-6
    selected = Table.read(str(base) + "_sources.fits", hdu="SOURCES")
    np.testing.assert_array_equal(selected["OBJECT_ID"], table["ID"])
    np.testing.assert_array_equal(selected["REDSHIFT"], table["Z"])
    np.testing.assert_allclose(selected["E_RES1"], c.g1 - selected["E_PRED1"])
    np.testing.assert_allclose(selected["B_RES1"], c.g2 - selected["B_PRED1"])
    assert out["counts_map"].sum() == len(table)
    manifest = json.loads(
        (tmp_path / method / f"femmi_output_{method}_run.json").read_text()
    )
    assert manifest["catalogue"]["hdu"] == "SHAPES"
    assert len(manifest["input_sha256"]) == 64
    assert manifest["timings"]["total_s"] > 0
    with pytest.raises(FileExistsError):
        map_mass(**options(path, method=method, save_fits=True, output_dir=tmp_path))


def test_units_wrap_selection_response_identity(tmp_path):
    path = tmp_path / "wrap.fits"
    table = catalogue(path, wrap=True, unit="rad")
    with fits.open(path, mode="update") as h:
        h["SHAPES"].data["FLAGS"][0] = 1
        h["SHAPES"].data["R11"][0] = 0  # discarded rows need no valid response
        h["SHAPES"].data["weight"][1] = -1
    c = read_fits_catalog(
        path,
        hdu="SHAPES",
        id_col="ID",
        z_col="Z",
        flag_col="FLAGS",
        reject_bits=1,
        response="auto",
        input_kind="ellipticity",
    )
    assert c.row_index[0] == 2
    np.testing.assert_array_equal(c.object_id, table["ID"][2:])
    np.testing.assert_allclose(c.g1, table["g1"][2:] / 0.7)
    assert min(c.center()[0], 360 - c.center()[0]) < 0.1
    flat = c.to_tangent_plane()
    assert np.max(np.abs(flat.x)) < 2
    out = map_mass(
        **options(
            path,
            weight_col="weight",
            id_col="ID",
            z_col="Z",
            flag_col="FLAGS",
            reject_bits=1,
            z_range=[0.9, 1.8],
            masks=[(0, 0, 0.1)],
        )
    )
    assert out["metadata"]["catalogue"]["n_dropped"] == len(table) - out["catalogue"].n
    assert out["metadata"]["catalogue"]["selection"]["z_range"] == [0.9, 1.8]
    assert "spatial_mask" in out["metadata"]["catalogue"]["rejected_rows"]
    assert out["counts_map"].sum() == out["catalogue"].n


def test_response_is_opt_in_and_no_hidden_amplitude_cut(tmp_path):
    path = tmp_path / "raw.fits"
    table = catalogue(path)
    with fits.open(path, mode="update") as h:
        h["SHAPES"].data["g1"][0] = 3.0
    c = read_fits_catalog(path, hdu="SHAPES")
    assert c.g1[0] == 3.0
    np.testing.assert_allclose(c.g2, table["g2"])
    with pytest.raises(ValueError, match="explicit response"):
        read_fits_catalog(path, hdu="SHAPES", input_kind="ellipticity")
    c = read_fits_catalog(path, hdu="SHAPES", max_shear=2)
    assert c.row_index[0] == 1
    with pytest.raises(ValueError, match="reduced shear"):
        map_mass(**options(path, input_kind="reduced_shear"))


def test_reference_image_rotation_and_pixel_axis_coordinates(tmp_path):
    path = tmp_path / "input.fits"
    table = catalogue(path)
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [35, -30]
    w.wcs.crpix = [4.25, 5.75]
    theta = 0.4
    w.wcs.cd = (
        np.array([[-np.cos(theta), np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        * 0.004
    )
    reference = tmp_path / "reference.fits"
    fits.PrimaryHDU(np.zeros((9, 7)), w.to_header()).writeto(reference)
    out = map_mass(**options(path, reference_image=reference))
    assert out["maps"]["E"].shape == (9, 7)
    np.testing.assert_allclose(
        out["wcs"].all_pix2world([[0, 0], [3, 5]], 0),
        w.all_pix2world([[0, 0], [3, 5]], 0),
    )
    px, py = w.all_world2pix(table["ra"], table["dec"], 0)
    expected = _coverage(np.c_[px, py], np.ones(len(table)), (9, 7))[0]
    np.testing.assert_array_equal(out["counts_map"], expected)
    pixel = map_mass(
        **options(
            path,
            coord_system="pixel",
            pixel_scale_arcmin=0.01,
            downsample_factor=20,
            save_fits=True,
            output_dir=tmp_path,
        )
    )
    np.testing.assert_array_equal(pixel["catalogue"].g2, table["g2"])
    assert not pixel["wcs"].has_celestial
    assert (tmp_path / "p3/femmi_output_p3_e_mode.fits").exists()
    assert pixel["counts_map"].sum() == len(table)
    with pytest.raises(ValueError, match="pixel_scale_arcmin"):
        map_mass(**options(path, coord_system="pixel", downsample_factor=20))


@pytest.mark.parametrize("shuffle", ["orientation", "spatial"])
def test_seeded_snr_matches_explicit_population_variance(tmp_path, shuffle):
    import random
    from femmi.survey import _smooth

    path = tmp_path / "input.fits"
    catalogue(path)
    kw = options(
        path,
        create_snr=True,
        num_shuffles=3,
        seed=17,
        shuffle_type=shuffle,
        mode=["E", "B"],
        smoothing=0.4,
        snr_smoothing=0.5,
        weight_col="weight",
    )
    a = map_mass(**kw)
    b = map_mass(**kw)
    for m in ("E", "B"):
        np.testing.assert_array_equal(a["snr_maps"][m], b["snr_maps"][m])
    c = a["catalogue"]
    mapper = a["reconstructions"]["E"].mapper
    yy, xx = np.indices(a["maps"]["E"].shape)
    x, y = _sky_to_xy(*a["wcs"].all_pix2world(xx, yy, 0), c.center, "smpy")
    pts = np.c_[x.ravel(), y.ravel()]
    rng = np.random.default_rng(17)
    null = []
    for i in range(3):
        if shuffle == "orientation":
            angle = np.arctan2(c.g2, c.g1) + rng.uniform(0, 2 * np.pi, c.n)
            amp = np.hypot(c.g1, c.g2)
            g1, g2 = amp * np.cos(angle), amp * np.sin(angle)
            worker = mapper
        else:
            order = list(range(c.n))
            random.Random(17 + i).shuffle(order)
            worker = FEMMapper(replace(c, x=c.x[order], y=c.y[order]), mapper.config)
            g1, g2 = c.g1, c.g2
        image = _smooth(worker.reconstruct(g1, g2).evaluate(pts).reshape(xx.shape), 0.4)
        image[a["coverage"] == 0] = np.nan
        null.append(_smooth(image, 0.5))
    valid = a["coverage"] > 0
    np.testing.assert_allclose(
        a["variance_maps"]["E"][valid], np.var(null, axis=0)[valid], rtol=1e-10
    )
    np.testing.assert_allclose(
        a["snr_maps"]["E"][valid],
        (_smooth(a["maps"]["E"], 0.5) / np.sqrt(np.var(null, axis=0)))[valid],
    )


def test_cli_fits_workflow_and_plot(tmp_path):
    import yaml
    from femmi.cli import main
    from femmi.survey_config import load_survey_config

    path = tmp_path / "input.fits"
    catalogue(path)
    ctr = tmp_path / "xray.ctr"
    ctr.write_text("fk5\nline\n34.999 -30.001\n35.001 -29.999\n")
    cfg = dict(
        general=dict(
            input_path=str(path),
            input_hdu="SHAPES",
            output_directory=str(tmp_path),
            method="p3",
            coordinate_system="radec",
            radec=dict(resolution=0.2),
            save_fits=True,
            save_plots=True,
            create_counts_map=True,
            overlay_counts_map=True,
        ),
        methods=dict(p3=dict(lam=0.3, length=0.6)),
        plotting=dict(
            figsize=[4, 3],
            xray_contours=dict(ctr_file=str(ctr), show_on_convergence=True),
        ),
    )
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(cfg))
    main(["map", "--config", str(config)])
    assert (tmp_path / "p3/femmi_output_p3_e_mode.png").stat().st_size > 1000
    assert (tmp_path / "p3/femmi_output_p3_counts.fits").exists()
    cfg["general"]["typo"] = 3
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="unknown survey options"):
        load_survey_config(config)


def test_smpy_input_projection_contract(tmp_path):
    pytest.importorskip("smpy")
    from smpy.utils import load_shear_data
    from smpy.coordinates.radec import RADecSystem

    path = tmp_path / "input.fits"
    catalogue(path)
    ref = RADecSystem().transform_coordinates(
        load_shear_data(path, "ra", "dec", "g1", "g2", None, hdu="SHAPES")
    )
    c, f, _, _, scaled, true = _prepare(
        path,
        "radec",
        "ra",
        "dec",
        "g1",
        "g2",
        None,
        "SHAPES",
        "smpy",
        "smpy",
        None,
        None,
        1,
        {},
    )
    np.testing.assert_allclose(f.x, ref["coord1_scaled"] * 60, atol=1e-10)
    np.testing.assert_allclose(f.y, ref["coord2_scaled"] * 60, atol=1e-10)
    np.testing.assert_array_equal(f.weight, ref["weight"])
    np.testing.assert_array_equal(f.g1, ref["g1"])
    np.testing.assert_array_equal(f.g2, -ref["g2"])


def test_pinned_smpy_wcs_and_orientation_seed_defects(tmp_path):
    """Evidence for the documented exceptions; intentionally pinned to upstream commit."""
    pytest.importorskip("smpy")
    from smpy.utils import save_fits, generate_multiple_shear_dfs
    import pandas as pd

    path = tmp_path / "upstream.fits"
    save_fits(
        np.zeros((8, 10)),
        dict(ra_min=34.99, ra_max=35.01, dec_min=-30.01, dec_max=-29.99),
        path,
    )
    w = WCS(fits.getheader(path))
    sky = w.all_pix2world([[0, 4], [9, 4]], 0)
    assert sky[0, 0] > sky[1, 0]  # Opposite to increasing-RA gridding.
    np.testing.assert_array_equal(
        w.wcs.crpix, [5, 4]
    )  # Correct centered reference: [5.5,4.5].
    df = pd.DataFrame(
        dict(
            g1=[0.1, 0.2],
            g2=[0.2, -0.1],
            coord1_scaled=[0.0, 1.0],
            coord2_scaled=[0.0, 1.0],
        )
    )
    state = np.random.get_state()
    try:
        np.random.seed(99)
        a = generate_multiple_shear_dfs(df, 1, "orientation", seed=7)[0]
        np.random.seed(98)
        b = generate_multiple_shear_dfs(df, 1, "orientation", seed=7)[0]
        assert not np.array_equal(a.g1, b.g1)  # Same declared seed, different output.
    finally:
        np.random.set_state(state)


def test_fits_integer_nulls_are_not_observations(tmp_path):
    path = tmp_path / "null.fits"
    fits.BinTableHDU.from_columns(
        [
            fits.Column(name="ra", format="J", null=-999, array=[35, -999, 36]),
            fits.Column(name="dec", format="D", array=[-30.0, -30.0, -30.0]),
            fits.Column(name="g1", format="D", array=[0.1, 0.1, 0.1]),
            fits.Column(name="g2", format="D", array=[0.2, 0.2, 0.2]),
        ]
    ).writeto(path)
    c = read_fits_catalog(path)
    np.testing.assert_array_equal(c.row_index, [0, 2])
    assert c.meta["rejected_rows"]["nonfinite"] == [1]


def test_tan_shear_convention_conversion_precedes_rotation():
    from femmi.io import ShearCatalog, rotate_shear, projection_rotation

    c = ShearCatalog.from_arrays(
        np.array([31.0, 32.0]),
        np.array([65.0, 66.0]),
        np.array([0.1, 0.2]),
        np.array([0.2, -0.1]),
    )
    flat = c.to_tangent_plane(center=(30.0, 60.0), flip_g2=True)
    angle = projection_rotation(
        np.deg2rad(c.ra), np.deg2rad(c.dec), *np.deg2rad([30.0, 60.0])
    )
    a, b = rotate_shear(c.g1, -c.g2, angle)
    np.testing.assert_allclose(flat.g1, a)
    np.testing.assert_allclose(flat.g2, b)


def test_zero_signal_nulls_do_not_create_infinite_significance(tmp_path):
    path = tmp_path / "zero.fits"
    catalogue(path)
    with fits.open(path, mode="update") as h:
        h["SHAPES"].data["g1"][:] = 0
        h["SHAPES"].data["g2"][:] = 0
    out = map_mass(
        **options(path, create_snr=True, shuffle_type="orientation", num_shuffles=2)
    )
    assert np.all(np.isnan(out["snr_maps"]["E"]))
    assert np.nanmax(out["variance_maps"]["E"]) == 0


def test_default_grid_boundary_counts_and_effective_density(tmp_path):
    path = tmp_path / "data.fits"
    table = catalogue(path)
    out = map_mass(**options(path, weight_col="weight", masks=[(0, 0, 0.3)]))
    c = out["catalogue"]
    ra, dec = out["input_catalogue"].ra, out["input_catalogue"].dec
    ra, dec = ra[c.row_index], dec[c.row_index]
    pix = np.column_stack(out["wcs"].all_world2pix(ra, dec, 0))
    counts, weight, eff, _ = _coverage(pix, c.weight, out["counts_map"].shape)
    np.testing.assert_array_equal(out["counts_map"], counts)
    assert counts.sum() == c.n
    assert weight.sum() == pytest.approx(c.weight.sum())
    assert np.all(eff <= counts + 1e-12)
    assert np.any(out["coverage"] == 3)
    assert np.all(np.isfinite(out["maps"]["E"][out["coverage"] == 3]))
    with pytest.raises(ValueError, match="max_pixels"):
        map_mass(**options(path, max_pixels=1))


@pytest.mark.parametrize("method", ["p3", "argyris", "hct"])
def test_analytic_sky_peak_location_and_b_rotation_contract(tmp_path, method):
    from femmi.catalog import analytic_gaussian_shear

    rng = np.random.default_rng(5)
    xy = rng.uniform(-1.4, 1.4, (180, 2))
    peak = (0.25, -0.15)
    _, g1, g2 = analytic_gaussian_shear(xy, sigma=0.4, amp=0.1, center=peak)
    dec = -30 + xy[:, 1] / 60
    ra = 35 + xy[:, 0] / (60 * np.cos(np.deg2rad(dec)))
    path = tmp_path / "gaussian.fits"
    Table(dict(ra=ra, dec=dec, g1=g1, g2=-g2)).write(path)
    out = map_mass(
        path,
        method=method,
        pixel_scale=0.15,
        lam=0.03,
        length=0.3,
        mode=["E", "B"],
        centre=(35, -30),
        save_plots=False,
    )
    yy, xx = np.indices(out["maps"]["E"].shape)
    x, y = _sky_to_xy(*out["wcs"].all_pix2world(xx, yy, 0), (35, -30), "smpy")
    truth, _, _ = analytic_gaussian_shear(
        np.c_[x.ravel(), y.ravel()], sigma=0.4, amp=0.1, center=peak
    )
    image = out["maps"]["E"]
    valid = np.isfinite(image) & (np.hypot(x, y) < 1.1)
    assert np.corrcoef(truth.reshape(xx.shape)[valid], image[valid])[0, 1] > 0.98
    j, i = np.unravel_index(np.nanargmax(image), image.shape)
    assert np.hypot(x[j, i] - peak[0], y[j, i] - peak[1]) < 0.2
    mapper = out["reconstructions"]["E"].mapper
    c = out["catalogue"]
    # Finite-field B is a rotated-shear diagnostic, not a claimed exact E/B decomposition.
    np.testing.assert_allclose(
        out["reconstructions"]["B"].kappa,
        mapper.reconstruct(c.g2, -c.g1).kappa,
        atol=1e-10,
    )


def test_reference_wcs_frame_conversion(tmp_path):
    from astropy.coordinates import SkyCoord
    from astropy.wcs.utils import pixel_to_skycoord

    path = tmp_path / "data.fits"
    catalogue(path)
    origin = SkyCoord(35, -30, unit="deg").galactic
    w = WCS(naxis=2)
    w.wcs.ctype = ["GLON-TAN", "GLAT-TAN"]
    w.wcs.crval = [origin.l.deg, origin.b.deg]
    w.wcs.crpix = [4, 4]
    w.wcs.cdelt = [-0.004, 0.004]
    ref = tmp_path / "galactic.fits"
    fits.PrimaryHDU(np.zeros((7, 7)), w.to_header()).writeto(ref)
    out = map_mass(**options(path, reference_image=ref))
    yy, xx = np.indices((7, 7))
    sky = pixel_to_skycoord(xx, yy, w).icrs
    x, y = _sky_to_xy(sky.ra.deg, sky.dec.deg, out["catalogue"].center, "smpy")
    expected = (
        out["reconstructions"]["E"]
        .evaluate(np.c_[x.ravel(), y.ravel()])
        .reshape((7, 7))
    )
    expected[out["coverage"] == 0] = np.nan
    np.testing.assert_allclose(out["maps"]["E"], expected, atol=1e-10, equal_nan=True)


def test_pinned_smpy_upper_grid_edge_loss():
    pytest.importorskip("smpy")
    import pandas as pd
    from smpy.coordinates.radec import RADecSystem

    coords = RADecSystem()
    df = pd.DataFrame(
        dict(
            coord1=[0.0, 0.01, 0.0, 0.01],
            coord2=[0.0, 0.0, 0.01, 0.01],
            g1=[0.1] * 4,
            g2=[0.2] * 4,
            weight=[1.0] * 4,
        )
    )
    scaled, _ = coords.calculate_boundaries(df.coord1, df.coord2)
    coords.create_grid(
        coords.transform_coordinates(df),
        scaled,
        dict(general=dict(radec=dict(resolution=0.3), create_counts_map=True)),
    )
    assert coords._last_count_grid.sum() < len(df)
    count, _, _, _ = _coverage(
        np.array([[-0.5, -0.5], [1.5, -0.5], [-0.5, 1.5], [1.5, 1.5]]),
        np.ones(4),
        (2, 2),
    )
    assert count.sum() == 4


def test_masks_preserve_grid_and_sip_coverage_registration(tmp_path):
    from astropy.wcs import Sip

    path = tmp_path / "input.fits"
    catalogue(path)
    base = map_mass(**options(path))
    masked = map_mass(**options(path, masks=[(0.6, 0.6, 0.5)]))
    np.testing.assert_allclose(
        masked["wcs"].all_pix2world([[0, 0], [2, 3]], 0),
        base["wcs"].all_pix2world([[0, 0], [2, 3]], 0),
    )
    assert (
        masked["metadata"]["mapper"]["radius"] == base["metadata"]["mapper"]["radius"]
    )
    w = base["wcs"].deepcopy()
    a = np.zeros((3, 3))
    b = np.zeros((3, 3))
    a[2, 0] = 0.001
    b[0, 2] = -0.001
    w.sip = Sip(a, b, None, None, w.wcs.crpix)
    w.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    ref = tmp_path / "sip.fits"
    fits.PrimaryHDU(np.zeros((7, 7)), w.to_header(relax=True)).writeto(ref)
    out = map_mass(
        **options(path, reference_image=ref, save_fits=True, output_dir=tmp_path)
    )
    with fits.open(tmp_path / "p3/femmi_output_p3_e_mode.fits") as hdul:
        for hdu in hdul:
            copy = WCS(hdu.header)
            assert copy.sip is not None
            np.testing.assert_allclose(
                copy.all_pix2world([[1.0, 2.0], [3.0, 4.0]], 0),
                w.all_pix2world([[1.0, 2.0], [3.0, 4.0]], 0),
                atol=1e-12,
            )


def test_unsupported_plot_option_fails_before_outputs(tmp_path):
    path = tmp_path / "input.fits"
    catalogue(path)
    with pytest.raises(ValueError, match="unsupported plotting"):
        map_mass(
            **options(
                path, save_fits=True, output_dir=tmp_path, plotting={"threshold": 3.0}
            )
        )
    assert not (tmp_path / "p3").exists()


def test_conditional_b_products(tmp_path):
    path = tmp_path / 'catalog.fits'
    catalogue(path)
    out = map_mass(**options(path, mode=['E','B'], b_diagnostics=True,
        save_fits=True,output_dir=tmp_path,smoothing=.5))
    assert out['metadata']['b_response']['closure_relative'] < 2e-5
    np.testing.assert_allclose(out['maps']['B'],
        out['b_diagnostic_maps']['leakage']+out['b_diagnostic_maps']['residual'],
        atol=1e-8,equal_nan=True)
    for name in ('b_leakage','b_residual'):
        with fits.open(tmp_path/'p3'/f'femmi_output_p3_{name}.fits',checksum=True) as hdul:
            assert hdul[0].verify_checksum()==1
    with pytest.raises(ValueError,match='requires B'):
        map_mass(**options(path,mode='E',b_diagnostics=True))
