# FEMMI examples

Two groups: a short **teaching set** here in `examples/` that walks the public
API, and `examples/diagnostics/` — the internal validation, benchmark, and
figure-generation scripts used while developing FEMMI (kept for reproducibility,
not needed to learn the API).

All scripts run from the repo root, e.g. `python examples/quickstart.py`.

## Start here (teaching)

| script | what it teaches |
|---|---|
| `quickstart.py` | ~15 lines: catalog → reconstruction. The minimal end-to-end path. |
| `catalog_comparison.py` | FEMMI vs Fourier Kaiser–Squires on the same catalog (head-to-head). |
| `prior_comparison.py` | Swapping priors (Wiener / TV / sparse / max-entropy / neural). |
| `uncertainty_demo.py` | Posterior sampling: mean + per-pixel uncertainty map. |
| `plot_npz.py` | Plot the `.npz` a `femmi run` writes (truth / kappa / std / samples). |
| `compare_runs.py` | Head-to-head of several runs on one field (Wiener vs neural vs hybrid): L2 table + side-by-side means + appearance-frequency maps. |
| `paper_artifacts.py` | Flagship: reproduces the Remy et al. 2020 figure structure, config-driven. |

The recommended path into the library, though, is the CLI + a config file:

```bash
femmi run --config configs/default.yaml
python examples/plot_npz.py runs/run.npz   # see the result
```

## diagnostics/ (development & validation)

Forward-operator accuracy (`bem_dtn_diagnostic`, `bem_scaling_diagnostic`),
systematics null tests (`eb_modes_demo`, `bmode_dipole_diagnostic`),
regularization/ablation studies (`bc_ablation`, `prior_bakeoff`), independent-truth
benchmarks (`galsim_nfw_benchmark`, `smpy_comparison`), and figure/plot generators
for the README and talks (`generate_figures`, `generate_presentation_figures`,
`pme_talk_plots`, `visualize_results`). These are heavier, some need optional extras
(`galsim`, `astropy`), and they encode research decisions rather than API usage.
