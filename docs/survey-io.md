# Survey FITS workflow

The production `femmi.map_mass` API and `femmi map` command take FITS shear
catalogues and produce registered astronomical images. The public vocabulary,
RA/Dec flattening and default shear sign follow
[SMPy at 26d231f](https://github.com/GeorgeVassilakis/SMPy/tree/26d231f5b4b22b41e76cb3bd3143d799c2e7ebfe).
The inverse is FEMMI's catalogue-based quadratic estimator.
`map_catalogue(flat, MapperConfig(...))` is the separate low-level Python call.

```python
from femmi import map_mass

result = map_mass(
    'shapes.fits', input_hdu='SHAPES', method='p3',
    coord1='ALPHA_J2000', coord2='DELTA_J2000',
    g1_col='g1', g2_col='g2', weight_col='weight', id_col='ID',
    pixel_scale=0.4, lam=0.3, length=0.6, mode=['E', 'B'],
    create_counts_map=True, save_fits=True, save_plots=True,
    output_dir='results', output_base_name='cluster')
```

The prior values above illustrate syntax; they are not calibrated survey choices.
Use independent calibration or held-out shear selection. Lambda has the same
weight convention as the observations; length is in arcminutes. Numerical
stopping and residual acceptance remain FEMMI's 1e-8 and 1e-6, respectively.
FFT-based KS has no corresponding CG residual tolerance to copy.

For the same workflow from a configuration file:

```bash
pip install -e '.[io,speed]'
femmi map --config configs/survey.yaml --catalogue shapes.fits --output-dir results
```

[`configs/survey.yaml`](https://github.com/AdamField118/FEMMI/blob/main/configs/survey.yaml) uses SMPy's `general`, `methods`,
`snr` and `plotting` sections, plus a `catalogue` section for explicit selection
and calibration. Set `general.method` to `p3`, `argyris`, or `hct`. Each has its
own `lam` and `length`; the examples are not a recommendation to reuse tuning
across elements. Paths are relative to the working directory, as in SMPy.
Unknown options raise. Outputs are protected from accidental replacement;
`overwrite=True` or CLI `--overwrite` explicitly replaces them.

## Input contract

| Concern | Behavior |
|---|---|
| FITS structure | Binary table HDU, index or extension name; scalar columns only |
| Sky positions | ICRS RA/Dec; honor FITS `TUNIT`, with explicit `ra_unit`/`dec_unit` override; missing units mean degrees |
| RA wrapping | Unwrap around the spherical mean before SMPy's midrange centering |
| Pixel positions | SMPy's `X_IMAGE`/`Y_IMAGE` defaults; require `pixel_scale_arcmin` to supply physical prior units and `downsample_factor` for the output grid |
| Pixel origin | Native catalogue coordinates are preserved, not shifted; `pixel_origin` records whether they use 0 or 1, default 1 |
| Shear signs | `shear_convention='smpy'` flips `g2` for RA/Dec, leaves pixel components unchanged; `east_north` disables that conversion |
| Projection | `projection='smpy'`: `(RA-RA0)*cos(Dec), Dec-Dec0`; `tan`: gnomonic coordinates with spin-2 basis rotation after sign conversion |
| Weights | `weight_col=None` means unit weights, exactly as in SMPy's public API; an explicit column is used without normalization |
| Response | `response=None` never recalibrates, even if response columns exist; `response='auto'` explicitly requests per-object `R11/R22[/R12/R21]`; a scalar, diagonal pair, mean matrix or per-source matrix is also accepted |
| Ellipticities | `input_kind='ellipticity'` requires an explicit response; `shear` declares the chosen columns already calibrated unless response is separately requested |
| Quality selection | Boolean `selection`, `flag_col`/`reject_bits`, optional `z_col`/`z_range`, optional positive `max_shear`; no hidden amplitude cut |
| Invalid data | Nonfinite positions/shears/weights, FITS numeric `TNULL`, and nonpositive weights are removed with recorded row reasons |
| Spatial masks | Circular exclusions `(x,y,r)` in computational arcminutes, or arbitrary source selections supplied as a boolean array |
| IDs | `id_col` preserved as `OBJECT_ID`; every selected source also retains its zero-based original `ROW_INDEX` |
| Redshifts | Preserved and usable for selection; never interpreted as source-by-source lensing efficiencies |

The reader's column names can be mapped for any survey; this is not a claim that
all surveys share a single FITS schema. Survey-level selection response,
additive/multiplicative bias calibration and calibrated shear weights remain
caller responsibilities. A per-object response inverse alone is not a complete
survey metacalibration prescription. The low-level reader's optional covariance
columns must describe calibrated shear variance; they are not transformed when
response correction is applied.

The forward model supports **linear shear at one effective source plane**.
Reduced-shear input is rejected. Convert non-ICRS catalogues and shear bases
before ingestion. Pixel catalogues use the supplied physical angular scale and
retain their Cartesian shear axes; no unknown detector WCS is guessed.
Duplicate source positions still fail explicitly in the estimator rather than
being silently combined with an undocumented noise model.

## WCS and image products

Every sky output has a genuine celestial WCS. Pixel values are evaluated from
the actual finite-element field at the sky positions specified by that WCS.
The default grid has SMPy's dimensions, with its WCS extent set from the source
positions projected onto a TAN plane. Therefore its sky registration is exact
within the declared projection; arrays are not merely labelled with an
approximate bounding-box header. With `projection='smpy'`, TAN output samples
are converted back to SMPy's computational coordinates before evaluating the FE
field. Increasing array x corresponds to increasing RA in the default frame.

`reference_image='image.fits', reference_hdu=0` adopts an existing image's shape
and celestial WCS, including rotation, handedness, SIP and sky-frame conversion.
FITS lookup-table distortions are rejected explicitly because writing only their
header would lose required auxiliary tables. Pixel inputs produce a LINEAR WCS
in the original catalogue pixel coordinates. Pixel FITS output is supported.

Files live in `output_dir/method/`, with the SMPy naming pattern:

| Suffix after `base_method_` | Contents |
|---|---|
| `e_mode.fits`, `b_mode.fits` | Dimensionless convergence from the original and rotated `(g2,-g1)` catalogue |
| `snr_e_mode.fits`, `snr_b_mode.fits` | Optional randomized-catalogue SNR |
| `null_variance_*_mode.fits`, `null_mean_*_mode.fits` | Noise estimator and null mean underlying SNR |
| `counts.fits` | Source count per pixel, when `create_counts_map=True` |
| `weight_sum.fits` | Sum of selected source weights in each output pixel |
| `effective_density.fits` | `(sum w)^2 / sum(w^2)` divided by local pixel area, in arcmin^-2 |
| `coverage.fits` | 0 outside support; 1 unobserved/interpolated; 2 data present; 3 explicitly masked |
| `sources.fits` | Original row/ID, coordinates, calibrated computational shear, weight, optional redshift, predicted shear and residuals |
| `run.json` | Source hash, resolved columns/units, response, selection/rejected rows, geometry, prior, solver diagnostics, versions, null settings and timings |

Image FITS files include checksums and an identically registered `COVERAGE`
extension. Unsupported image pixels are NaN. Unobserved and masked pixels within
the physical field are prior-dependent predictions, not zero measurements.
The computational mesh ring extends beyond the declared physical field; those
extra mesh pixels are excluded from survey maps. For a reference image smaller
than the source field, the manifest separately records sources inside the image.

E/B images and optional SNR/count images also have PNG products. Plots use
WCSAxes, SMPy's viridis/12x8 defaults and count-label option. SMPy's `.ctr`
configuration (`plotting.xray_contours.ctr_file`, `show_on_convergence`,
`show_on_snr`, color/linewidth/alpha) is accepted. Celestial contours are
transformed through WCS. Image/physical contours require pixel output; a sky
registration is never inferred from their coordinate extrema. Alternatively,
`xray_image` accepts a primary-HDU celestial FITS image for registered contours.

The result dictionary retains SMPy's `maps`, `scaled_boundaries`,
`true_boundaries`, `snr_maps`, `counts_map`, plus `wcs`, coverage, weights,
effective density, reconstructed FE objects, selected catalogue, metadata,
timings and output paths. Spatial boundary dictionaries describe the catalogue,
not the footprint of a separately supplied reference image.

## Noise and B diagnostics

`create_snr=True` follows SMPy's null-catalogue construction and uses the
population variance (`ddof=0`), with 100 shuffles by default. Spatial shuffling
uses the same Python `random` permutation with `seed+i`; shear and weight remain
paired while their positions move. Orientation shuffling preserves shear
amplitudes and uses a local, correctly seeded NumPy generator. `seed='random'`
generates a seed and saves its resolved integer. Science and null maps use the
same estimator, prior, geometry, signs, selected rows, masks, evaluation support
and smoothing. Orientation shuffles reuse the mapper; spatial permutations
rebuild it because the weight/position relation changes.

`snr_smoothing` defaults to SMPy's two-pixel Gaussian width; `smoothing` controls
optional science-map smoothing. Null variance and SNR include the same two
smoothing stages. Zero null variance yields NaN SNR, never infinity. Streaming
variance avoids storing the full stack of null images.

These are **randomized-catalogue noise estimates, not posterior intervals or
calibrated detection probabilities**. The regularization is held fixed during
null runs. Survey-specific selection bias, correlated shape noise, cosmic shear,
source redshift uncertainty and prior uncertainty are not included.

The B map is the same estimator applied to rotated shear. On a finite,
irregularly sampled domain it is not an exact orthogonal E/B decomposition.
Set `b_diagnostics=True` to save the response to the rotated E prediction as
`b_leakage.fits`, and the response to the rotated shear residual as
`b_residual.fits`. Their sum reproduces the raw B map within solve error.
The manifest records finite-element L2 norms and the closure error. Both maps
are conditional on the fitted E model and prior; neither is a pure-B projector.
See the [observation model](observation-model.md) for the defining equation.

## Compatibility with SMPy

The public vocabulary follows the pinned SMPy version. The following differences
affect interpretation or file registration:

1. **WCS registration:** upstream `utils.save_fits` writes negative RA scale for an
   array gridded in increasing RA, uses `CRPIX=(nx/2,ny/2)` rather than the centred
   FITS pixel `(nx+1)/2,(ny+1)/2`, and treats raw RA span as tangent angular span.
   FEMMI evaluates the field on the WCS grid instead.
2. **Orientation seeding:** upstream `generate_multiple_shear_dfs`
   ignores the numeric seed for orientation shuffles. FEMMI seeds a local RNG.
3. **Grid-edge loss:** upstream `digitize` excludes the final upper bin edge.
   FEMMI includes both outer boundaries with a 1e-8 pixel roundoff tolerance.
4. **Pixel FITS omission:** SMPy's run path only writes FITS in its RA/Dec branch.
   FEMMI honors `save_fits` for pixel catalogues too, using an honest LINEAR WCS.
5. **Estimator differences:** FEMMI operates on individual source locations,
   requires explicit lambda/physical length, and uses its own convergence checks.
   Its nulls use that same FEM estimator, not a KS approximation. FE output
   evaluation, source residuals and masked predictions cannot be made identical
   to an FFT estimator without changing the scientific method.
6. **Additional safeguards/products:** unit-aware ingestion, explicit calibration,
   rejected-row provenance, protected overwrites, actual WCS overlays, reference
   images and supplementary diagnostic FITS files extend SMPy's workflow.

Core settings and product names are matched; this does not implement every
SMPy plotting convenience. Peak annotations/cluster-centre markers, plot-only
`pixel_axis_reference='map'`, and arbitrary SMPy method-specific options raise
rather than being silently ignored. The supplied FEMMI config shows the supported
vocabulary. Inactive method/coordinate sections may remain in a shared YAML.


## Geometry controls

`boundary_padding` sets the unobserved mesh ring radius relative to the physical
field (default 1.12). `boundary_nodes` optionally fixes its resolution. Vary them
independently when assessing exterior sensitivity; changing padding is not a
substitute for testing uncertain exterior mass. These options can be placed in
the selected `methods` section of the YAML file.

## Joint E/B products

For a simultaneous fit with separate priors, set `joint_eb=True`. See
[joint E/B mapping](eb-mapping.md) for products, configuration, and interpretation.
These are additional posterior-mode maps; the ordinary rotated-fit SNR does
not apply to them.
