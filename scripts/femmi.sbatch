#!/bin/bash
#SBATCH --job-name=femmi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH -o slurm_outfiles/femmi-%j.out
#SBATCH -e slurm_outfiles/femmi-%j.err

# General FEMMI run: one config describes the whole pipeline (forward, data,
# inverse, prior, sampler, output). Submit:
#
#   sbatch scripts/femmi.sbatch configs/my_run.yaml
#   sbatch scripts/femmi.sbatch configs/my_run.yaml inverse.method=sample prior.kind=neural
#
# The first arg is the config; any further args are `--set key=value` overrides.

set -euo pipefail
# SLURM copies the batch script to a spool dir before running it, so
# ${BASH_SOURCE[0]} points there, not at the repo. SLURM_SUBMIT_DIR is the
# directory you ran `sbatch` from (the repo root, if you submit from there).
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
echo "repo=$REPO  cwd=$(pwd)"

CONFIG="${1:-configs/default.yaml}"
shift || true
SETS=()
for kv in "$@"; do SETS+=(--set "$kv"); done

# --- JAX / GPU memory ------------------------------------------------------
# Don't let JAX pre-grab most of the GPU (looks like an OOM on a shared card);
# grow on demand instead. The FEM forward stays in float64 on the CPU where it
# needs the precision to converge; only the neural score net runs in float32.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"

# Activate your environment:
source /cm/shared/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-13.2.0/miniconda3-25.1.1-24g7bpuxyyxo5pfd4zn5sldbomvz736a/bin/activate
conda activate femmi

echo "host=$(hostname)  config=$CONFIG  overrides=${*:-none}"
python -c "import jax; print('JAX devices:', jax.devices())"

# -u = unbuffered so progress streams into the .log live.
srun python -u -m femmi.cli run --config "$CONFIG" "${SETS[@]}"
