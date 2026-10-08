# Development and troubleshooting

## Code layout

| Modules | Responsibility |
|---|---|
| `io`, `survey`, `survey_config`, `survey_output` | Catalogue ingestion, coordinate conventions, survey orchestration, and products |
| `mapping`, `quadratic` | Reusable production estimator and accepted quadratic solve |
| `mesh`, `elements`, `operators`, `c1_assembly`, `c1_coupling` | Element geometry, FE assembly, and coupling |
| `bem`, `bem_hp`, `_bem_assembly`, `_bem_numba` | Boundary quadrature and CPU assembly |
| `forward`, `inverse`, `priors`, `sampling`, `pipeline` | Research adjoint, nonquadratic reconstruction, and sampling |
| `protocol`, `calibration`, `comparison`, `aperture`, `smpy` | Frozen comparisons, tuning, scoring, and upstream adapters |
| `diagnostics` | Operator identities and conditional B-response decomposition |

Keep catalogue cleaning separate from numerical assembly. Any new source
selection must preserve row provenance and apply to all paired benchmark arms.
An element implementation needs consistent value evaluation, volume quadrature,
Hessian recovery, and adjoint tests. It also needs a manufactured reconstruction
and an independent-truth check; polynomial reproduction alone is insufficient.

## Tests and documentation

```bash
python -m pip install -e '.[dev,io,galsim,speed,docs]'
python -m pip install -r requirements-benchmark.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 FEMMI_REQUIRE_OPTIONAL=io,galsim \
  python -m pytest -q -m 'not slow'
mkdocs build --strict
```

Install `.[neural]` and set `FEMMI_REQUIRE_OPTIONAL=1` to require every optional
scientific dependency. Run the marked slow tests for training/sampling changes.
Use the NumPy backend as a numerical reference when changing Numba assembly.

For boundary and multi-halo diagnostics:

```bash
python examples/diagnostics/scientific_checks.py --truth nfw \
  --sources 64 180 --padding 1.12 1.5 --lengths .3 --strict \
  --output results/scientific-checks
```

The driver checks adjoint and skew identities, fixed-geometry superposition,
and B-response closure. `--strict` measures sensitivity to tighter solves.
Source count, ring padding, and prior length can be varied independently.
These algebraic thresholds do not impose an arbitrary reconstruction-quality
pass mark on an uncalibrated model. Inspect independent-truth errors and then
use held-out calibration for accuracy comparisons.

## Common problems

| Symptom | Action |
|---|---|
| Missing FITS or independent-truth dependency | Install `.[io]` or `.[galsim]` in the same Python environment |
| SMPy revision rejected | Install `requirements-benchmark.txt`; the adapter requires its immutable Git revision |
| Sources outside radius | Check angular units and centre; choose a radius covering selected sources |
| Duplicate positions | Combine or select duplicates explicitly with an appropriate noise model |
| CG residual rejected | Inspect geometry/weights and parameter scale; retain the failed diagnostics and adjust numerical settings deliberately |
| Strong B structure | Inspect the conditional response and finite-field limitations; a raw B fit is not a pure-B projection |
| NaNs in output maps | Check coverage and WCS; unsupported pixels are intentionally NaN |
| Calibration stops at an edge | Widen the relevant search axis and rerun before evaluation |
| KS+ stability gate fails | Investigate iteration budget and threshold-schedule sensitivity, then recalibrate |
| Existing-output error | Choose a new run directory, or explicitly allow replacement for survey products |

Report reproducible issues on
[GitHub](https://github.com/AdamField118/FEMMI/issues). Include the configuration,
versions, smallest failing input, and full error. See the repository's
[contribution guide](https://github.com/AdamField118/FEMMI/blob/main/CONTRIBUTING.md)
for the review workflow.
