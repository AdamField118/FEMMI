# Contributing to FEMMI

Contributions are welcome — bug reports, fixes, new priors, better examples, docs.

## Getting set up

```bash
git clone https://github.com/AdamField118/FEMMI.git
cd FEMMI
pip install -e ".[dev]"        # add ,neural for the learned prior, ,galsim for the benchmark
python -m pytest tests/ -q      # everything should pass
```

## Ground rules

- **Keep it approachable.** FEMMI should read like something you can pick up in an
  afternoon. New examples in `examples/` should be short, self-contained, and teach
  one idea — not a framework. If a change makes the public API harder to explain,
  reconsider it.
- **Add a test.** New behaviour needs a test in `tests/` (see the existing ones for
  the plain, dependency-light style). Correctness-critical maths (operators,
  priors, samplers) should include a check against an analytic or finite-difference
  reference.
- **Match the surrounding style.** PEP 8, small functions, comments that explain
  *why* not *what*. No new hard dependencies without discussion — optional features
  (the neural prior, GalSim benchmark) live behind extras in `pyproject.toml`.
- **Document the maths.** If you add or change an operator, update `MATH.md` and,
  where relevant, the module docstring.

## Pull requests

1. Branch from the default branch.
2. Make the change, add tests, run `python -m pytest tests/ -q`.
3. Open a PR describing what changed and why. Small, focused PRs are easiest to review.

## Reporting issues

Please include the smallest reproducer you can, the full traceback, and your
Python / JAX / SciPy versions.
