# FEMMI

FEMMI reconstructs weak-lensing convergence from shear catalogues using finite
and boundary elements. It fits the observations at their source positions and
supports cubic Lagrange (P3), Argyris, and Hsieh–Clough–Tocher (HCT) elements.
The survey interface reads FITS tables and writes convergence images, source
residuals, coverage, and optional randomized-catalogue noise maps.

The model assumes linear shear on a flat field at one effective source plane.
Regularization supplies information where observations are insufficient; a
reconstructed value inside a mask is a prediction under that prior.

## Install

Use Python 3.10 or later in a virtual environment:

```bash
git clone https://github.com/AdamField118/FEMMI.git
cd FEMMI
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[io,speed]'
python examples/quickstart.py
```

Numba accelerates CPU boundary assembly. The package retains a NumPy backend
and uses SciPy for sparse solves. No GPU is required.

## Map a survey catalogue

Copy and edit `configs/survey.yaml` to set the column names, shear convention,
angular scale, and prior for your data, then run:

```bash
femmi map --config configs/survey.yaml --catalogue shapes.fits --output-dir results/cluster
```

In Python, `femmi.map_mass` provides the same survey workflow. Use `FEMMapper`
when working directly with arrays or reusing a mesh for several shear catalogues.

## Documentation

Start with [installation](docs/installation.md), the runnable
[quickstart](docs/quickstart.md), and the [FITS guide](docs/survey-io.md).
The [API reference](docs/api.md) describes arguments, return values, and errors.
The [mathematical model](MATH.md) defines the operator and its limitations.

```bash
python -m pip install -e '.[docs]'
mkdocs serve
```

For experiments, see the [benchmark protocol](docs/smpy-benchmarks.md) and
[profiling guide](docs/production-performance.md). Keep recipes in Git and
write generated data to `results/`. Archive results separately with the exact
configuration and source revision when publishing a comparison.

## Contributing and support

Report bugs or ask usage questions in
[GitHub Issues](https://github.com/AdamField118/FEMMI/issues). Include a small
reproducer, your configuration, dependency versions, and the traceback.
See [CONTRIBUTING.md](CONTRIBUTING.md) for development and testing instructions.
FEMMI is distributed under the [MIT license](LICENSE.md).
When citing unreleased work, identify the repository and the exact commit used.
