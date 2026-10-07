#!/bin/bash
#SBATCH --job-name=bench
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out

# Runs scripts/run_benchmarks.sh inside a full-node allocation, as the LUMI AI Guide
# does (05-multi-gpu-and-node/run_ddp_srun_4.sh). The CPU bind masks address 7 cores in
# each of 8 GPU groups; a bare srun from a login node was given 4 per group and aborted
# with "Unable to satisfy cpu bind request". Each template's srun becomes a job step here.
#
#   sbatch --account=$PROJECT_NAME scripts/run_benchmarks.sbatch
#
# Environment (PROJECT_NAME, RESULTS_ROOT, OLD_CONTAINER, NEW_CONTAINER, SKIP_EXISTING,
# EXCLUDE_NODES) is passed through from the submitting shell.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
exec ./scripts/run_benchmarks.sh
