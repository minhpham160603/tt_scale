#!/bin/bash
# PRM Reliability Analysis - Complete Workflow
#
# This script automates the two-step process for PRM reliability analysis:
# 1. Generate Best-of-N samples with a generator model
# 2. Analyze correlations using different PRMs
#
# Usage: ./run_prm_analysis.sh [options]

set -e  # Exit on error

# Default parameters
GENERATOR_MODEL="Qwen/Qwen2.5-7B-Math-Instruct"
PRM_MODEL="Qwen/Qwen2.5-Math-PRM-7B"
DATASET="MathArena/aime_2025"
N=64
NUM_SAMPLES=""
OUTPUT_DIR="prm_analysis_data"
GPU_UTIL=0.9
TOKEN_THRESHOLDS="128 256 512 1024 1536 2048"
AGG_METHODS="mean min last whole"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --generator)
            GENERATOR_MODEL="$2"
            shift 2
            ;;
        --prm)
            PRM_MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --N)
            N="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --generator MODEL       Generator model (default: Qwen/Qwen2.5-7B-Math-Instruct)"
            echo "  --prm MODEL             PRM model (default: Qwen/Qwen2.5-Math-PRM-7B)"
            echo "  --dataset DATASET       Dataset name (default: MathArena/aime_2025)"
            echo "  --N NUM                 Number of solutions per problem (default: 64)"
            echo "  --num_samples NUM       Number of problems to sample (default: all)"
            echo "  --output_dir DIR        Output directory (default: prm_analysis_data)"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate dataset name for files
DATASET_CLEAN=$(echo "$DATASET" | tr '/' '_')
GENERATOR_CLEAN=$(echo "$GENERATOR_MODEL" | tr '/' '_')
PRM_CLEAN=$(echo "$PRM_MODEL" | tr '/' '_')

DATA_FILE="${OUTPUT_DIR}/${DATASET_CLEAN}_${GENERATOR_CLEAN}_N${N}.json"

echo "=========================================="
echo "PRM Reliability Analysis Workflow"
echo "=========================================="
echo "Generator: $GENERATOR_MODEL"
echo "PRM: $PRM_MODEL"
echo "Dataset: $DATASET"
echo "N: $N"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Step 1: Generate Best-of-N samples
echo "Step 1: Generating Best-of-N samples..."
echo "----------------------------------------"

GENERATE_CMD="python -m tt_scale.prm_correlation.generate_best_of_n \
    --model_name $GENERATOR_MODEL \
    --dataset $DATASET \
    --N $N \
    --output_dir $OUTPUT_DIR \
    --gpu_memory_utilization $GPU_UTIL"

if [ -n "$NUM_SAMPLES" ]; then
    GENERATE_CMD="$GENERATE_CMD --num_samples $NUM_SAMPLES"
fi

echo "Running: $GENERATE_CMD"
echo ""

if [ -f "$DATA_FILE" ]; then
    echo "Data file already exists: $DATA_FILE"
    read -p "Overwrite? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        eval $GENERATE_CMD
    else
        echo "Skipping generation step..."
    fi
else
    eval $GENERATE_CMD
fi

echo ""
echo "Step 1 complete: $DATA_FILE"
echo ""

# Step 2: Analyze correlations
echo "Step 2: Analyzing PRM correlations..."
echo "----------------------------------------"

ANALYZE_CMD="python -m tt_scale.prm_correlation.analyze_prm_correlation \
    --data_file $DATA_FILE \
    --prm_model $PRM_MODEL \
    --token_thresholds $TOKEN_THRESHOLDS \
    --aggregation_methods $AGG_METHODS"

echo "Running: $ANALYZE_CMD"
echo ""

eval $ANALYZE_CMD

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "=========================================="
echo "Data file: $DATA_FILE"
echo "Results: ${DATA_FILE%.json}_correlation_${PRM_CLEAN}_4bit.json"
echo "=========================================="
