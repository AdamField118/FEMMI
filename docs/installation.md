# Installation

FEMMI requires Python 3.10 or later. Create an isolated environment from the
repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[io,speed]'
```

This installs the `femmi` command, FITS support, and optional CPU acceleration.
For core array-based reconstruction, `python -m pip install -e .` is sufficient.

| Extra | Use |
|---|---|
| `io` | FITS catalogues, image products, and simulation maps via Astropy |
| `speed` | Float64 CPU BEM kernels via Numba |
| `galsim` | Independent NFW truth for synthetic comparisons |
| `mesh` | Triangle-based adaptive meshing |
| `neural` | Learned score priors via Flax and Optax |
| `dev` | Tests and notebooks |
| `docs` | MkDocs documentation site |
| `paper` | Convenience group containing `galsim` and `io` |

Extras compose, for example `python -m pip install -e '.[dev,io,galsim,speed]'`.
For SMPy comparisons, also install the pinned upstream revision:

```bash
python -m pip install -r requirements-benchmark.txt
```

Check the installation with `python examples/quickstart.py`. That example needs
no external catalogue. Use `femmi --help` to list commands.

FEM and boundary calculations use float64. Importing FEMMI enables JAX's x64
mode. Neural networks use float32 internally. The production mapper runs on CPU;
installing a GPU JAX build does not move its SciPy solves to a GPU.

`FEMMI_BEM_BACKEND` accepts `auto` (default), `numpy`, or `numba`. Automatic
selection uses Numba when available. An explicit Numba request fails if its
optional dependency is missing. See [profiling](production-performance.md) for
cache configuration and backend comparisons.
