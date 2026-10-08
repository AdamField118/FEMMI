#!/bin/bash
#SBATCH --job-name=femmi-profile
#SBATCH --cpus-per-task=1
#SBATCH --output=femmi-profile-%j.log
set -euo pipefail
# Submit from the checkout; set PYTHON_BIN to the environment's Python.
# Supply partition, memory and time using your cluster's sbatch options.
cd "${SLURM_SUBMIT_DIR:?Submit from the FEMMI repository root}"
: "${PYTHON_BIN:?Set PYTHON_BIN to the absolute path of your environment Python}"
exec "$PYTHON_BIN" examples/diagnostics/run_profiles.py "$@"
