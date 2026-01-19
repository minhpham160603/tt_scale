#!/usr/bin/env bash
set -euo pipefail

# Example parameter sweep for current tt_scale entrypoint:
#   python3 -m tt_scale.scripts.benchmarks --config <yaml>
# Usage:
#   bash scripts/sweep_vllm_hybrid.sh

cd "$(dirname "$0")/.."

# If you use a venv, uncomment:
if [[ -f .env/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .env/bin/activate
fi

MODEL="Qwen/Qwen2.5-3B-Instruct-AWQ"
BASE_CONFIG="${BASE_CONFIG:-tt_scale/config/example.yaml}"
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
MS=(6)
K_TOKENS=(256)
PASSING_MINIMUMS=(2)
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

            tmp_cfg="$(mktemp -t tt_scale_sweep_XXXXXX.yaml)"

            python3 - "$BASE_CONFIG" "$tmp_cfg" \
              "$MODEL" "$tau" "$m" "$KEEPING_BRANCHES" "$kt" "$pm" "$agg" "$bt" "$wu" <<'PY'
import sys, yaml

base_path = sys.argv[1]
out_path = sys.argv[2]
model_name = sys.argv[3]
tau = float(sys.argv[4])
max_total_branches = int(sys.argv[5])
keeping_branches = int(sys.argv[6])
k_tokens = int(sys.argv[7])
passing_minimum = int(sys.argv[8])
agg = sys.argv[9]
bt = sys.argv[10]
wu = sys.argv[11]

with open(base_path, "r") as f:
    cfg = yaml.safe_load(f) or {}

# Use the hybrid/backtrack-style searcher.
cfg["method"] = cfg.get("method") or "collective"
cfg["model_name"] = model_name
cfg["tau"] = tau
cfg["max_total_branches"] = max_total_branches
cfg["keeping_branches"] = keeping_branches
cfg["k_tokens"] = k_tokens
cfg["passing_minimum"] = passing_minimum
cfg["agg"] = agg
cfg["backtrack"] = (bt == "backtrack")
cfg["warmup"] = (wu == "warmup")

with open(out_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

            python3 -m tt_scale.scripts.benchmarks \
              --config "$tmp_cfg" \
              "${DATASET_ARGS[@]}" \
              --detail-log none \
              --summary-csv "$SUMMARY_CSV" \
              --run-tag "$RUN_TAG"

            rm -f "$tmp_cfg"
            done
          done
        done
      done
    done
  done
done
