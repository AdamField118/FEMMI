# Quickstart

Install FEMMI, then run `python examples/quickstart.py`. It creates an analytic
shear catalogue, reconstructs it and writes `mass-map.npz`.

```python
from femmi import read_fits_catalog, MapperConfig, FEMMapper

flat = read_fits_catalog('shear.fits').to_tangent_plane(units='arcmin')
config = MapperConfig('hct', lam=0.3, length=0.6, radius=3.)
mapper = FEMMapper(flat, config)
result = mapper.reconstruct()
result.save('mass-map.npz')
```

These prior settings illustrate the API. See [Catalogue mapper](mapper.md) for
held-out shear selection, units, source-plane assumptions and spatial evaluation.
The result includes source convergence, full FE coefficients, predicted shear,
mesh geometry and solver diagnostics. No plot or file is generated unless asked.

For survey FITS input and [SMPy-style output](survey-io.md):

```bash
femmi map --catalogue shear.fits --config configs/survey.yaml --output-dir results
```

The separate [research pipeline](configuration.md) supports nonquadratic priors
and posterior sampling. Use [SMPy comparisons](smpy-benchmarks.md) for held-out
benchmarks and [production performance](production-performance.md) for timings.
