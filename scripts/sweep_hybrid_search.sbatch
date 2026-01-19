#!/bin/bash
#SBATCH -A deep_learning
#SBATCH -t 20:00:00
#SBATCH -J sweep_vllm_hybrid
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Note: GPU request syntax varies by cluster.
# If your cluster supports GPU types via Slurm, keep this:
#SBATCH --gpus=5060ti:1
# Otherwise you may need one of these instead (ask admin / check docs):
# #SBATCH --gpus=1
# #SBATCH --gres=gpu:1

set -euo pipefail

# Prefer running from the submission directory if available.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR}"
else
  cd /home/mipham/tt_scale
fi

mkdir -p logs

# If you want to force venv usage, uncomment:
# if [[ -f .env/bin/activate ]]; then
#   source .env/bin/activate
# fi

# Current entrypoint uses YAML config (+ runtime overrides).
CONFIG_FILE="${CONFIG_FILE:-tt_scale/config/example.yaml}"
SUMMARY_CSV="${SUMMARY_CSV:-logs/sweep_summary.csv}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

python3 -m tt_scale.scripts.benchmarks \
  --config "${CONFIG_FILE}" \
  --detail-log none \
  --summary-csv "${SUMMARY_CSV}" \
  --run-tag "${RUN_TAG}"
