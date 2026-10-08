# Contributing to FEMMI

Open an issue for a bug, API proposal, or scientific-model question. For a bug,
include a minimal catalogue or synthetic reproducer, configuration, package
versions, and expected versus observed behavior. Do not attach private survey data.

Create a branch from main and keep each change focused. For numerical changes,
state the equation or invariant being changed and add a test that would detect
an incorrect result. Compare forward and transpose paths together. Document
new arguments, units, return fields, and failure conditions alongside the code.

```bash
python -m pip install -e '.[dev,io,galsim,speed,docs]'
python -m pip install -r requirements-benchmark.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 FEMMI_REQUIRE_OPTIONAL=io,galsim \
  python -m pytest -q -m 'not slow'
mkdocs build --strict
```

For the neural tests, also install `.[neural]`. Set `FEMMI_REQUIRE_OPTIONAL=1`
to require all optional scientific test dependencies rather than skipping them.
Run marked slow tests when changing training or sampling behavior.

Keep small deterministic fixtures and their generation instructions in `tests/`.
Keep benchmark code in `examples/` and configurations in `configs/benchmarks/`.
Write generated catalogues, maps, profiles, figures, and reports under `results/`.
Do not commit run products or put performance tables in the user guides. Archive
publication outputs separately with source revision, environment, raw results,
and commands needed to reproduce them.

Use ordinary descriptive prose. Explain what a function expects and returns,
then the assumptions a caller needs to use it correctly. Avoid development
history, unsupported performance claims, and descriptions of fixes that no
longer explain current behavior.
