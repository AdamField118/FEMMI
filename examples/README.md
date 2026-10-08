# Examples

Run scripts from the repository root after installing FEMMI. Use `--help` to
inspect configurable drivers. Save generated files under `results/` or another
ignored output directory.

| Task | Entry point |
|---|---|
| First catalogue reconstruction | `quickstart.py` |
| Survey FITS mapping | `femmi map --config configs/survey.yaml` |
| Held-out FEM/SMPy comparisons | `paper/benchmark_smpy.py` |
| Rebuild comparison summaries | `paper/report_calibration.py` |
| Adjoint, halo-superposition, and B-response diagnostics | `diagnostics/scientific_checks.py` |
| NumPy/Numba fresh-process comparison | `diagnostics/run_profiles.py` |
| Detailed CPU profile | `diagnostics/profile_cpu.py` |
| FITS ingestion and output timing | `diagnostics/benchmark_survey_io.py` |
| Posterior-sampling example | `uncertainty_demo.py` |
| Plot saved research outputs | `plot_npz.py`, `compare_runs.py` |
| Direct/iterative and dense/compressed solver experiments | `diagnostics/numerical_followup.py` |

The remaining `paper/` scripts are focused numerical experiments, including
manufactured potentials, forward convergence, and Hessian recovery. Manufactured
shear generated with the same FEM operator tests algebraic consistency; use
independent truth for reconstruction comparisons. A nonzero response to a finite
constant sheet does not establish that observations fix an infinite mass sheet.

`paper_artifacts.py` and the plot-generation scripts are research plotting
utilities. Their layouts or defaults do not define a publication benchmark.
Use the frozen protocol and retain raw paired results for method comparisons.

See the [benchmark guide](../docs/smpy-benchmarks.md),
[profiling guide](../docs/production-performance.md), and
[sampling guide](../docs/priors-and-sampling.md) for complete workflows.
