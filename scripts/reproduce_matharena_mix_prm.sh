#!/bin/bash
# Reproduce PRM Reliability Analysis for all MathArena Mix datasets
#
# This script runs the complete PRM analysis for all datasets used in
# the paper's MathArena Mix benchmark.

set -e

# Configuration
GENERATOR_MODEL="Qwen/Qwen2.5-7B-Math-Instruct"
PRM_MODEL="Qwen/Qwen2.5-Math-PRM-7B"
N=64
OUTPUT_DIR="prm_analysis_data"

# MathArena Mix datasets
DATASETS=(
    "MathArena/aime_2025"
    "MathArena/hmmt_feb_2025"
    "MathArena/cmimc_2025"
    "MathArena/brumo_2025"
    "MathArena/apex_2025"
    "MathArena/hmmt_nov_2025"
)

echo "=========================================="
echo "Reproducing PRM Reliability Analysis"
echo "for MathArena Mix (Section 3)"
echo "=========================================="
echo "Generator: $GENERATOR_MODEL"
echo "PRM: $PRM_MODEL"
echo "N: $N"
echo "Datasets: ${#DATASETS[@]}"
echo "=========================================="
echo ""

# Step 1: Generate samples for all datasets
echo "STEP 1: Generating Best-of-$N samples"
echo "=========================================="
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "Processing: $dataset"
    echo "----------------------------------------"

    python -m tt_scale.prm_correlation.generate_best_of_n \
        --model_name "$GENERATOR_MODEL" \
        --dataset "$dataset" \
        --N "$N" \
        --output_dir "$OUTPUT_DIR" \
        --gpu_memory_utilization 0.9

    echo "Completed: $dataset"
done

echo ""
echo "STEP 1 COMPLETE"
echo ""

# Step 2: Analyze correlations for all datasets
echo "STEP 2: Analyzing correlations"
echo "=========================================="

for dataset in "${DATASETS[@]}"; do
    dataset_clean=$(echo "$dataset" | tr '/' '_')
    generator_clean=$(echo "$GENERATOR_MODEL" | tr '/' '_')
    data_file="${OUTPUT_DIR}/${dataset_clean}_${generator_clean}_N${N}.json"

    if [ ! -f "$data_file" ]; then
        echo "Warning: Data file not found: $data_file"
        echo "Skipping $dataset"
        continue
    fi

    echo ""
    echo "Analyzing: $dataset"
    echo "----------------------------------------"

    python -m tt_scale.prm_correlation.analyze_prm_correlation \
        --data_file "$data_file" \
        --prm_model "$PRM_MODEL" \
        --token_thresholds 128 256 512 1024 1536 2048 \
        --aggregation_methods mean min last whole

    echo "Completed: $dataset"
done

echo ""
echo "=========================================="
echo "ANALYSIS COMPLETE"
echo "=========================================="
echo ""
echo "Results saved in: $OUTPUT_DIR/"
echo ""
echo "To aggregate results across all datasets:"
echo "  python -m tt_scale.scripts.aggregate_prm_results --input_dir $OUTPUT_DIR"
echo ""
