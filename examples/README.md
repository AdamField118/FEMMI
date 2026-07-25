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
| `benchmark_grid.py` | Runs the {element} × {prior} × {method} grid on one independent truth and ranks the results — accuracy, DC-removed shape error, DOFs and wall-clock, with KS as the automatic baseline. The "which combination wins" command. |

The recommended path into the library, though, is the CLI + a config file:

```bash
femmi run --config configs/default.yaml
python examples/plot_npz.py runs/run.npz   # see the result
```

## paper/ (the thesis: FEMMI vs Kaiser–Squires)

The structural results that justify choosing FEMMI over KS in a real pipeline —
each is a standalone figure backed by `femmi.experiments`.

| script | result |
|---|---|
| `paper/independent_truth.py` | **[central]** both claims on truth NEITHER method generated (analytic GalSim NFW, or MassiveNuS + aperiodic shear). FEMMI's boundary advantage survives; its DC-mode advantage does not. Read this before `mass_sheet.py`. |
| `paper/mass_sheet.py` | the same absolute-normalisation comparison on FEMMI's OWN forward shear. Shows the mechanism cleanly, but it is an inverse crime — quote `independent_truth.py` for the headline number. |
| `paper/boundary_bias.py` | reconstruction error vs distance-from-edge — FEMMI's exact BEM far-field beats KS's truncation near the boundary (also self-consistent shear; the neutral version is in `independent_truth.py`). |
| `paper/injectivity.py` | the DC mode at the operator level: FEMMI's forward observes a uniform sheet (`‖F·1‖>0`); the KS/FFT forward annihilates it. Operator-level fact, and it holds — but see the caveat below. |
| `paper/forward_convergence.py` | the potential ψ converges at the P3 theory rate `O(h⁴)` — the forward operator's validation. |
| `paper/shear_recovery.py` | shear extraction reaches `O(h²)`; variational recovery beats nodal sampling by ~1.8× in constant; and noise amplified by `h⁻²` makes that rate unreachable catalog-native. |
| `paper/element_comparison.py` | **element choice for shear**: P3 nodal / P3 recovered / HCT / Argyris on one plot. Argyris reaches `O(h⁴)` — 42× more accurate at `h=0.156` for `1.04×` the DOFs. |

**Scope of the mass-sheet claim.** `‖F·1‖ > 0` is true and KS's is exactly zero,
so the DC mode is formally observable to FEMMI and formally invisible to KS. But
~99.9% of that response is concentrated in the *corners* of the square domain and
it grows under refinement rather than converging, so it does not turn into
practical DC recovery: on independent truth FEMMI's mean-κ error is comparable to
KS's. State the injectivity result as an operator property, not as a solved
mass-sheet degeneracy.

All plots use the shared paper style (`femmi.plotstyle`, white background,
colorblind-safe colormaps).

## diagnostics/ (development & validation)

Forward-operator accuracy (`bem_dtn_diagnostic`, `bem_scaling_diagnostic`),
systematics null tests (`eb_modes_demo`, `bmode_dipole_diagnostic`),
regularization/ablation studies (`bc_ablation`, `prior_bakeoff`), independent-truth
benchmarks (`galsim_nfw_benchmark`, `smpy_comparison`), and figure/plot generators
for the README and talks (`generate_figures`, `generate_presentation_figures`,
`pme_talk_plots`, `visualize_results`). These are heavier, some need optional extras
(`galsim`, `astropy`), and they encode research decisions rather than API usage.
