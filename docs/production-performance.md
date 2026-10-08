# Profiling and backend comparisons

Use the production mapper profiler to measure setup and repeated reconstructions
on a fixed mesh. Each fit receives a deterministic new shear realization and
starts from zero. Settings and observations are saved with the timings.

```bash
python -m pip install -e '.[speed]'
python examples/diagnostics/run_profiles.py \
  --sources 200 1000 --processes 3 --repeats 3 --output results/cpu-profile
```

The runner executes cases serially in fresh processes with one BLAS and Numba
thread. Each element/source-count case gets an empty Numba cache for its first
run; later processes reuse it. A NumPy run supplies the matched baseline.
The output directory must be empty. Start with small source counts before
allocating a larger job.

On Turing or another Slurm cluster, submit from the checkout root:

```bash
export PYTHON_BIN="$PWD/.venv/bin/python"
sbatch --time=02:00:00 --mem=16G scripts/profile_cpu.sbatch \
  --sources 200 1000 --processes 3 --repeats 3 --output results/cpu-profile
```

Adjust time, memory, account, and partition to the cluster and workload. The
script does not load site-specific modules or require a GPU. The time and memory
above are submission examples, not measured resource requirements.

## Recorded phases

| Field | Meaning |
|---|---|
| `numba_import_s` | Numba kernel import and initialization |
| `jit_compile_or_cache_load_s` | Compilation into a new cache or loading an existing one |
| `jit_first_execution_s` | First call after explicit compilation/loading |
| `catalogue_s` | Synthetic catalogue generation |
| `setup_s` | Meshing, assembly, and coupled factorization |
| `single_layer_s`, `double_layer_s` | BEM assembly subtimers |
| `volume_assembly_s`, `trace_s` | C1 setup subtimers |
| `coupled_factorization_s` | Sparse coupled LU |
| `prior_factorization_s` | Prior factorization during reconstruction |
| `reconstruction_s` | Sum of repeated reconstruction calls |
| `total_s` | Catalogue + setup + reconstruction |
| `total_including_jit_s` | Total plus explicitly separated Numba startup |
| `peak_rss_mib` | Process high-water memory, including imports |

Subtimers are nested: do not add factorization or assembly a second time.
Totals exclude process/module startup, output writing, and profiler overhead.
`plan.json` records the workload and source provenance; individual JSON files
record environment, dependency versions, thread settings, and solve diagnostics.

`comparisons.json` is written only after matching the workload, accepted
residuals, objectives, coefficients, and predicted shear. A failed parity check
stops the runner. Small test cases validate the workflow; use representative
catalogue sizes on the intended machine to establish performance claims.

## Locate remaining cost

Use a separate profiled run, since profiling overhead invalidates speed ratios:

```bash
python examples/diagnostics/profile_cpu.py --kind hct --sources 200 \
  --repeats 3 --warm-numba --profile --output results/profile/hct.json
python -m pstats results/profile/hct.prof
```

The CPU kernels preserve float64 and singular quadrature. NumPy/SciPy remain
responsible for vectorized assembly and sparse linear algebra. Numba operates
on geometry-dependent assembly, so the research custom adjoint remains intact.
Do not infer GPU gains or iterative-solver crossover from a boundary-kernel
speedup; measure the complete workload before changing those components.

For FITS ingestion and product-writing cost, use
`examples/diagnostics/benchmark_survey_io.py --help`. Write its JSON to
`results/`; the outputs include timings and process memory. Keep numerical
performance records in run archives rather than copying them into this guide.
