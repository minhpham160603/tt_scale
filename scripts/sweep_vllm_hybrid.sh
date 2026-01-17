#!/usr/bin/env bash
set -euo pipefail

# Example parameter sweep for tt_scale.vllm_hybrid_parallel_engine
# Usage:
#   bash scripts/sweep_vllm_hybrid.sh

cd "$(dirname "$0")/.."

# If you use a venv, uncomment:
if [[ -f .env/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .env/bin/activate
fi

MODEL="Qwen/Qwen2.5-3B-Instruct-AWQ"
DATASETS=(
  "MathArena/hmmt_nov_2025"
  "MathArena/aime_2025"
  "MathArena/cmimc_2025"
  "MathArena/brumo_2025"
  "MathArena/apex_2025"
  "MathArena/hmmt_feb_2025"

)

DATASET_ARGS=()
for ds in "${DATASETS[@]}"; do
  DATASET_ARGS+=(--dataset "$ds")
done

TAUS=(0.6)
MS=6
K_TOKENS=256
PASSING_MINIMUMS=2
AGGS=(last mean_last mean_step)

# sweep backtracking on/off
BACKTRACK_MODES=(backtrack)

# sweep warmup on/off (step-1 uses 2x k-tokens when enabled)
WARMUP_MODES=(warmup)

# fixed unless you also want to sweep it
KEEPING_BRANCHES=2

mkdir -p logs
SUMMARY_CSV="logs/sweep_summary_warmup.csv"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"

for tau in "${TAUS[@]}"; do
  for m in "${MS[@]}"; do
    for kt in "${K_TOKENS[@]}"; do
      for pm in "${PASSING_MINIMUMS[@]}"; do
        for agg in "${AGGS[@]}"; do
          for bt in "${BACKTRACK_MODES[@]}"; do
            for wu in "${WARMUP_MODES[@]}"; do
            echo "==== datasets=${#DATASETS[@]} tau=${tau} M=${m} K=${KEEPING_BRANCHES} k_tokens=${kt} passing_min=${pm} agg=${agg} bt=${bt} warmup=${wu} ===="

              if [[ "$bt" == "backtrack" ]]; then
                BT_FLAG="--backtrack"
              else
                BT_FLAG="--no-backtrack"
              fi

              if [[ "$wu" == "warmup" ]]; then
                WARMUP_FLAG="--warmup"
              else
                WARMUP_FLAG="--no-warmup"
              fi

            python3 -m tt_scale.vllm_hybrid_parallel_engine \
              --model "$MODEL" \
              "${DATASET_ARGS[@]}" \
              --tau "$tau" \
              --max-total-branches "$m" \
              --keeping-branches "$KEEPING_BRANCHES" \
              --k-tokens "$kt" \
              --passing-minimum "$pm" \
              --agg "$agg" \
              $BT_FLAG \
              $WARMUP_FLAG \
              --detail-log none \
              --summary-csv "$SUMMARY_CSV" \
              --run-tag "$RUN_TAG"
            done
          done
        done
      done
    done
  done
done
