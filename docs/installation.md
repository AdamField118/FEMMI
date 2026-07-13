# Installation

FEMMI requires Python ≥ 3.10.

```bash
git clone https://github.com/AdamField118/FEMMI.git
cd FEMMI
pip install -e .
```

This installs the core package and the `femmi` command.

## Optional extras

Install only what a given run needs:

```bash
pip install -e ".[neural]"   # learned score prior (Flax + optax)
pip install -e ".[io]"       # FITS shear catalogs / Frontier Fields maps (astropy)
pip install -e ".[galsim]"   # independent-truth NFW benchmark
pip install -e ".[mesh]"     # Triangle-based adaptive meshing
pip install -e ".[dev]"      # test suite (pytest)
pip install -e ".[docs]"     # build this documentation site (mkdocs-material)
```

Extras compose, e.g. `pip install -e ".[neural,io]"`.

## Verifying the install

```bash
femmi run --config configs/default.yaml   # a synthetic MAP reconstruction
pytest -q -m "not slow"                    # the fast test suite
```

!!! note "float64"
    The FEM forward runs in float64 (it needs the precision to converge). The
    neural score net runs in float32 for speed; both are handled automatically.
