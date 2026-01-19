# PRM Reliability Analysis Scripts

This directory contains scripts for reproducing the PRM reliability analysis from Section 3 of the paper.

## Overview

The analysis consists of two steps:

1. **Generate Best-of-N samples** (`generate_best_of_n.py`): Generate multiple solution candidates for each problem using a generator model
2. **Analyze correlations** (`analyze_prm_correlation.py`): Score the solutions with PRMs at different token thresholds and compute Pearson correlations

## Step 1: Generate Best-of-N Samples

This script generates N solutions for each problem in a dataset and saves them as JSON.

### Usage

```bash
python -m tt_scale.prm_correlation.generate_best_of_n \
    --model_name Qwen/Qwen2.5-7B-Math-Instruct \
    --dataset MathArena/aime_2025 \
    --N 64 \
    --temperature 0.8 \
    --max_tokens 2048 \
    --output_dir prm_analysis_data \
    --gpu_memory_utilization 0.9
```

### Arguments

- `--model_name`: Generator model (default: `Qwen/Qwen2.5-7B-Math-Instruct`)
- `--dataset`: HuggingFace dataset name (default: `MathArena/aime_2025`)
- `--split`: Dataset split (default: `train`)
- `--num_samples`: Number of problems to sample (default: `None` for all)
- `--N`: Number of solutions per problem (default: `64`)
- `--temperature`: Sampling temperature (default: `0.8`)
- `--max_tokens`: Maximum tokens per solution (default: `2048`)
- `--output_dir`: Output directory for JSON files (default: `prm_analysis_data`)
- `--tensor_parallel_size`: Tensor parallel size for vLLM (default: `1`)
- `--gpu_memory_utilization`: GPU memory utilization (default: `0.9`)
- `--dtype`: Data type (default: `float16`)
- `--quantization`: Quantization method (default: `None`)

### Output

Creates a JSON file in the output directory with the following structure:

```json
{
  "metadata": {
    "model_name": "...",
    "dataset": "...",
    "N": 64,
    ...
  },
  "results": [
    {
      "index": 0,
      "question": "...",
      "gold_answer": "...",
      "solutions": [
        {"text": "...", "num_tokens": 456},
        ...
      ]
    },
    ...
  ]
}
```

## Step 2: Analyze PRM Correlations

This script loads the generated samples, scores them with PRMs at different token thresholds, and computes Pearson correlations.

### Usage

```bash
python -m tt_scale.prm_correlation.analyze_prm_correlation \
    --data_file prm_analysis_data/MathArena_aime_2025_Qwen_Qwen2.5-7B-Math-Instruct_N64.json \
    --prm_model Qwen/Qwen2.5-Math-PRM-7B \
    --token_thresholds 128 256 512 1024 1536 2048 \
    --aggregation_methods mean min last whole
```

### Arguments

- `--data_file`: Path to JSON file from Step 1 (required)
- `--prm_model`: PRM model name (default: `Qwen/Qwen2.5-Math-PRM-7B`)
- `--token_thresholds`: Token thresholds to analyze (default: `[128, 256, 512, 1024, 1536, 2048]`)
- `--aggregation_methods`: Aggregation methods to test (default: `["mean", "min", "last", "whole"]`)
- `--output_file`: Output JSON file (default: auto-generated)
- `--no_quantization`: Disable 4-bit quantization
- `--device`: Device to use (default: `cuda`)

### Aggregation Methods

- **`mean`**: Average of all step scores
- **`min`**: Minimum score across all steps (weakest link)
- **`last`**: Score of the last step only
- **`whole`**: Treat the entire partial solution as a single block

### Output

The script produces:

1. **Console output**: Real-time progress and correlation values
2. **JSON file**: Detailed results with metadata and correlations
3. **Summary table**: Pearson correlations printed at the end

Example summary table:

```
================================================================================
SUMMARY: Pearson Correlation (Partial → Correctness)
================================================================================
Threshold   mean           min            last           whole
--------------------------------------------------------------------------------
τ=128       0.155          0.124          0.170          0.018
τ=256       0.293          0.216          0.264          0.025
τ=512       0.511          0.350          0.483          0.132
τ=1024      0.819          0.555          0.733          0.449
τ=1536      0.954          0.648          0.777          0.642
τ=2048      0.997          0.668          0.772          0.742
================================================================================
```

## Complete Example Workflow

```bash
# Step 1: Generate samples (takes ~1-2 hours depending on dataset size)
python -m tt_scale.prm_correlation.generate_best_of_n \
    --model_name Qwen/Qwen2.5-7B-Math-Instruct \
    --dataset MathArena/aime_2025 \
    --N 64 \
    --num_samples 100 \
    --output_dir prm_analysis_data

# Step 2: Analyze correlations (takes ~30-60 minutes)
python -m tt_scale.prm_correlation.analyze_prm_correlation \
    --data_file prm_analysis_data/MathArena_aime_2025_Qwen_Qwen2.5-7B-Math-Instruct_N64.json \
    --prm_model Qwen/Qwen2.5-Math-PRM-7B \
    --token_thresholds 128 256 512 1024 1536 2048 \
    --aggregation_methods mean min last whole
```

## Reproducing Paper Results

To reproduce the results from **Section 3** (PRM Reliability Analysis):

### MathArena Mix Dataset

```bash
# Generate samples for all MathArena datasets
for dataset in "MathArena/aime_2025" "MathArena/hmmt_feb_2025" "MathArena/cmimc_2025" "MathArena/brumo_2025" "MathArena/apex_2025" "MathArena/hmmt_nov_2025"; do
    python -m tt_scale.prm_correlation.generate_best_of_n \
        --model_name Qwen/Qwen2.5-7B-Math-Instruct \
        --dataset $dataset \
        --N 64 \
        --output_dir prm_analysis_data
done

# Analyze each dataset
for file in prm_analysis_data/MathArena_*_N64.json; do
    python -m tt_scale.prm_correlation.analyze_prm_correlation \
        --data_file $file \
        --prm_model Qwen/Qwen2.5-Math-PRM-7B
done
```

### Testing Different PRMs

To compare different PRM models (as in Table 7):

```bash
# Qwen 2.5 7B PRM (4-bit)
python -m tt_scale.prm_correlation.analyze_prm_correlation \
    --data_file prm_analysis_data/MathArena_aime_2025_Qwen_Qwen2.5-7B-Math-Instruct_N64.json \
    --prm_model Qwen/Qwen2.5-Math-PRM-7B

# Qwen 2.5 72B PRM (4-bit)
python -m tt_scale.prm_correlation.analyze_prm_correlation \
    --data_file prm_analysis_data/MathArena_aime_2025_Qwen_Qwen2.5-7B-Math-Instruct_N64.json \
    --prm_model Qwen/Qwen2.5-Math-PRM-72B

# RLHFlow PRM (full precision)
python -m tt_scale.prm_correlation.analyze_prm_correlation \
    --data_file prm_analysis_data/MathArena_aime_2025_Qwen_Qwen2.5-7B-Math-Instruct_N64.json \
    --prm_model RLHFlow/Llama3.1-8B-PRM \
    --no_quantization
```

## Key Findings

From the paper (Section 3.1):

1. **Contextual Threshold Hypothesis**: Correlation between partial scores and final correctness is negligible at τ=128 (r≈0.16) and only becomes reliable around τ=512
2. **Prediction-Verification Trade-off**: Mean aggregation outperforms Min for intermediate guidance, despite Min being optimal for final verification
3. **Best-of-N Performance**: Qwen 2.5 PRM significantly outperforms RLHFlow (16.86% vs 11.63%)

## Notes

- The analysis requires significant compute resources (GPU with at least 24GB VRAM recommended)
- For 4-bit quantization, ensure `bitsandbytes` is installed
- The grading uses the MathArena competition grader for accurate answer matching
- Results may vary slightly due to sampling randomness and hardware differences
