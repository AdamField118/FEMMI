# API reference

## `MapperConfig`

`MapperConfig(method, lam, length, radius, **options)` is immutable. It can be
saved with `config.save(path)` and read with `MapperConfig.from_file(path)`.

| Field | Default | Meaning |
|---|---|---|
| `method` | Required | `p3`, `argyris`, or `hct` |
| `lam` | Required | Positive penalty strength in the supplied weight convention |
| `length` | Required | Nonnegative correlation length in arcminutes |
| `radius` | Required | Positive physical field radius in arcminutes |
| `center` | `(0., 0.)` | Field centre in computational arcminutes |
| `rtol` | `1e-8` | Internal CG relative target |
| `residual_tolerance` | `1e-6` | Independently recomputed relative residual acceptance |
| `maxiter` | `2000` | Maximum CG iterations |
| `observable` | `shear` | Only linear shear is supported |
| `source_plane` | `effective` | Only one effective source plane is supported |
| `boundary_padding` | `1.12` | Ring radius divided by physical field radius; must exceed one |
| `boundary_nodes` | `None` | Ring vertex count, at least 12; otherwise chosen from catalogue size |

`rtol <= residual_tolerance < 1` is required. Invalid settings raise `ValueError`.

## `FEMMapper(catalogue, config)`

`catalogue` supplies one-dimensional `x`, `y`, `g1`, `g2`, and `weight` arrays
with equal lengths of at least three. `FlatCatalog` is the standard container.
Coordinates must be in arcminutes; arrays must be finite, positions distinct,
and weights nonnegative with a positive total. All sources must lie within the
configured radius. Inputs are copied so later edits cannot change the fitted model.

Construction assembles the mesh and forward system. The public attributes
`catalogue`, `config`, `dofs`, `vertices`, `triangles`, and `setup_seconds`
describe that geometry. Internal solver objects are implementation details.

| Method | Arguments and result |
|---|---|
| `reconstruct(g1=None, g2=None, *, lam=None, length=None)` | Fit stored shear or two replacement component arrays; optionally override the prior; return `MassMap` |
| `evaluate(coefficients, points)` | Evaluate a finite coefficient vector at finite `(n,2)` points; return an `(n,)` array with NaN outside the mesh |
| `select_regularization(lambdas, lengths, *, folds=3, seed=0)` | Return `best`, all `candidates`, `boundary_unresolved`, and split metadata using held-out shear MSE |
| `diagnose_b(g1=None, g2=None, *, e_fit=None, b_fit=None)` | Return E, raw B, predicted leakage, residual fits, and L2/closure diagnostics |

Both replacement shear components must be supplied together. Reuse requires the
same positions and weights; construct a new mapper when either changes.
A failed reconstruction raises rather than returning an unconverged estimate.
Supplied diagnostic fits must belong to the same mapper and match the data and
prior. See [B interpretation](observation-model.md#interpreting-rotated-shear-b-maps).

`map_catalogue(catalogue, config)` constructs a mapper and returns one fit.

## `MassMap`

| Attribute | Contents |
|---|---|
| `kappa` | Convergence at source positions |
| `coefficients` | Full FE coefficient vector, including derivative degrees of freedom for C1 elements |
| `observed_g1`, `observed_g2` | Shear arrays fitted in this reconstruction |
| `predicted_g1`, `predicted_g2` | Forward prediction at source positions |
| `config` | Effective configuration, including prior overrides |
| `diagnostics` | Convergence, fresh residual, objective, iterations, prior, and setup/solve times |
| `mapper` | Reusable geometry and solver owner |

`result.evaluate(points)` evaluates the FE field. `result.save(path)` writes an
NPZ of observations, weights, source convergence, coefficients, predictions,
mesh, and JSON metadata. The parent directory must exist. Load with
`numpy.load(path, allow_pickle=False)`; rebuild a mapper to evaluate new points.

## `map_mass`

`map_mass(data, ..., lam=..., length=...)` is the survey interface. `data` is a
FITS catalogue path. Use keyword arguments for clarity. The options below are
grouped by purpose; the [FITS guide](survey-io.md) gives the conventions and
file formats.

| Group | Options and defaults |
|---|---|
| Estimator | `method='p3'`, required `lam`, `length`; `rtol=1e-8`, `residual_tolerance=1e-6`, `maxiter=2000` |
| Input | `input_hdu=1`, `coord_system='radec'`, `coord1=None`, `coord2=None`, `g1_col='g1'`, `g2_col='g2'`, `weight_col=None` |
| Angular units | `ra_unit=None`, `dec_unit=None` honor FITS units; missing units mean degrees |
| Calibration | `input_kind='shear'`, `response=None`, `shear_convention='smpy'`, `source_plane='effective'` |
| Selection | `selection=None`, `flag_col=None`, `reject_bits=0`, `z_col=None`, `z_range=None`, `max_shear=None`, `masks=()` |
| Identity | `id_col=None` preserves original source IDs when specified |
| Projection | `projection='smpy'`, `centre=None` (sky RA/Dec), `radius=None` (inferred physical arcminutes) |
| Sky grid | `pixel_scale=None` must be supplied in arcmin/output pixel unless a reference image defines the grid |
| Pixel grid | `pixel_scale_arcmin=None` must supply arcmin/input pixel; `downsample_factor=None` must be supplied; `pixel_origin=1` |
| Reference grid | `reference_image=None`, `reference_hdu=0`, `max_pixels=4_000_000` |
| Mesh ring | `boundary_padding=1.12`, `boundary_nodes=None` |
| Map modes | `mode='E'` or a list containing E and/or B; `b_diagnostics=False` |
| Noise maps | `create_snr=False`, `num_shuffles=100`, `shuffle_type='spatial'`, `seed=0`, `snr_smoothing=2.0` pixels |
| Display | `smoothing=None`, `create_counts_map=False`, `overlay_counts_map=False`, `plotting=None`, `snr_plot_title='Signal-to-Noise Map'` |
| Overlays | `xray_image=None`, `xray_levels=None`, `xray_contours=None` |
| Files | `output_dir='.'`, `output_base_name='femmi_output'`, `save_fits=False`, `save_plots=True`, `overwrite=False` |
| Timing | `print_timing=False` |

`coord1`/`coord2` default to RA/Dec column names for sky input and
`X_IMAGE`/`Y_IMAGE` for pixel input. `weight_col=None` uses unit weights.
`response=None` means no response correction. Ellipticity input requires an
explicit response. Paths in YAML are relative to the working directory.

The result dictionary contains `maps` keyed by E/B, `snr_maps`, `counts_map`,
`scaled_boundaries`, `true_boundaries`, `wcs`, `coverage`, `weight_map`,
`effective_density_map`, `reconstructions`, `catalogue`, `input_catalogue`,
`variance_maps`, `null_means`, `metadata`, and `timings`.
Optional `b_diagnostic_maps` contains leakage and residual images. Arrays use
`[y,x]` order. Check the source and prior metadata before interpreting masked
pixels. Output paths are listed under `output_files` when products are requested.

Bad column mappings, unsupported calibration/model choices, and invalid options
raise `ValueError`; existing output files raise `FileExistsError` unless overwrite
is enabled. Sparse solve failures raise numerical errors. The CLI uses the same
validation and does not silently substitute another estimator.

## Catalogue utilities

`read_fits_catalog` returns a `ShearCatalog`. Use its `to_tangent_plane` method
for explicit lower-level projection, or let `map_mass` perform the survey
convention conversion. `FlatCatalog.select(mask)` returns selected rows while
preserving row provenance. The lower-level reader is distinct from the survey
workflow: it does not automatically imply SMPy's sign convention.

## Research interfaces

`DifferentiableForward` supplies the P3 JAX forward and custom adjoint.
`MAPReconstructor`, `make_prior`, and `sample_posterior` are described in
[priors and sampling](priors-and-sampling.md). The production mapper's NumPy/SciPy
call is not JAX-traceable. Array shape, mask, and noise conventions for research
calls are summarized in the [observation model](observation-model.md).
