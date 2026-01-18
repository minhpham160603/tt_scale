# Hybrid searcher for Test-time scaling

A scalable framework for tree search algorithms with vLLM, supporting multiple search strategies including collective backtracking, independent backtracking, beam search, and best-of-N.

## Repository Structure

```
tt_scale/
├── tt_scale/              # Main package
│   ├── config.py         # Configuration dataclass
│   ├── argparse.py        # Command-line argument parsing
│   ├── base_classes.py   # Abstract base classes (Generator, PRM, Searcher)
│   ├── generator/         # LLM generators (vLLM, HuggingFace)
│   ├── prm/              # Process Reward Models (PRMs)
│   ├── searcher/         # Search algorithms
│   │   ├── collective_backtrack.py
│   │   ├── independent_backtrack.py
│   │   ├── beam_search.py
│   │   └── best_of_n.py
│   ├── scripts/
│   │   └── benchmarks.py # Main benchmark script
│   └── utils.py          # Utilities and evaluation
├── tt_scale/config/
│   └── example.yaml      # Example configuration file
└── logs/                 # Output logs (CSV files)
```

## Installation

```bash
# Create virtual environment
python -m venv .env
source .env/bin/activate  # On Windows: .env\Scripts\activate

# Install package
pip install -e .

# Core dependencies (if not auto-resolved)
pip install torch transformers datasets vllm pyyaml
```

## Quick Start

### 1. Using YAML Configuration (Recommended)

Create a YAML configuration file (see `tt_scale/config/example.yaml` for a template):

```yaml
model_name: "Qwen/Qwen2.5-3B-Instruct-AWQ"
method: "collective"  # Options: "collective", "independent", "beam_search", "best_of_n"
tau: 0.6
max_total_branches: 9
keeping_branches: 3
passing_minimum: 3
max_steps: 30
k_tokens: 256
datasets:
  - "MathArena/hmmt_nov_2025"
split: "test"
```

Run the benchmark:

```bash
python -m tt_scale.scripts.benchmarks --config your_config.yaml
```

### 2. Using Command-Line Overrides

You can override runtime settings via command-line arguments:

```bash
python -m tt_scale.scripts.benchmarks \
    --config your_config.yaml \
    --gpu-mem-util 0.8 \
    --tensor-parallel-size 2 \
    --dataset "MathArena/aime_2025" \
    --summary-csv results/summary.csv \
    --run-tag "experiment_1"
```

### 3. Using Default Configuration

If no config file is provided, default values will be used:

```bash
python -m tt_scale.scripts.benchmarks
```

## Results on MATH-500

Performance comparison across different algorithms and difficulty levels:
Generator: meta-llama/Llama-3.2-1B-Instruct
PRM: Qwen/Qwen2.5-Math-PRM-7B

| Level | Algorithm | Accuracy | Avg. Tokens | Samples |
|-------|-----------|----------|-------------|---------|
| **Overall** | Collective Backtrack | 0.4600 | 3661.68 | 50 |
| | Beam Search | 0.4000 | 5304.84 | 50 |
| | Best of N | 0.3400 | 3461.44 | 50 |
| **Level 1** | Collective Backtrack | 0.5000 | 2733.50 | 4 |
| | Beam Search | 0.5000 | 4595.00 | 4 |
| | Best of N | 0.0000 | 2402.25 | 4 |
| **Level 2** | Collective Backtrack | 0.5385 | 3146.00 | 13 |
| | Beam Search | 0.3846 | 3927.08 | 13 |
| | Best of N | 0.3077 | 3063.46 | 13 |
| **Level 3** | Collective Backtrack | 0.6667 | 3171.83 | 12 |
| | Best of N | 0.6667 | 3220.00 | 12 |
| | Beam Search | 0.5833 | 4958.67 | 12 |
| **Level 4** | Collective Backtrack | 0.5000 | 4036.38 | 8 |
| | Beam Search | 0.3750 | 5681.88 | 8 |
| | Best of N | 0.3750 | 2903.12 | 8 |
| **Level 5** | Beam Search | 0.2308 | 6988.54 | 13 |
| | Collective Backtrack | 0.1538 | 4684.54 | 13 |
| | Best of N | 0.1538 | 4751.77 | 13 |



## Configuration

### Configuration File (YAML)

The main configuration is defined in a YAML file. Key parameters:

**Model & Search Method:**
- `model_name`: Model identifier (e.g., "Qwen/Qwen2.5-3B-Instruct-AWQ")
- `method`: Search algorithm - `"collective"`, `"independent"`, `"beam_search"`, or `"best_of_n"`

**Search Parameters:**
- `tau`: Threshold score to accept a step (default: 0.6)
- `max_total_branches`: Maximum total branches at each step (default: 9)
- `keeping_branches`: Number of branches to keep when pruning (default: 3)
- `passing_minimum`: Minimum passing branches to avoid backtracking (default: 3)
- `max_steps`: Maximum number of search steps (default: 30)
- `k_tokens`: Maximum tokens to generate per step (default: 256)
- `backtrack`: Enable/disable backtracking (default: true)

**Best-of-N Parameters:**
- `N`: Number of candidates to generate (default: 4)
- `temperature`: Sampling temperature (default: 0.8)
- `max_tokens`: Maximum tokens per candidate (default: 2048)
- `agg`: Aggregation method - `"last"`, `"mean"`, or `"mean_only_final"` (default: "last")

**Beam Search Parameters:**
- `num_iterations`: Maximum number of iterations (default: uses `max_steps`)
- `beam_width`: Beam width M (default: 1)
- `n_beams`: Total beams to maintain (default: uses `keeping_branches`)
- `lookahead`: Lookahead steps (default: 0)

**Dataset & Evaluation:**
- `datasets`: List of dataset names (default: MathArena datasets)
- `split`: Dataset split to use (default: "train")
- `num_samples`: Optional subsample size (default: None = full dataset)
- `detail_log`: Write per-question CSV logs (default: true)

**Runtime Settings (set via CLI, not in YAML):**
- `--gpu-mem-util`: GPU memory utilization (default: 0.5)
- `--tensor-parallel-size`: Tensor parallel size (default: 1)
- `--dtype`: Data type (default: "float16")
- `--summary-csv`: Path to summary CSV file
- `--run-tag`: Optional tag for runs

See `tt_scale/config/example.yaml` for a complete example.

### Programmatic Configuration

You can also modify the `Config` class in `tt_scale/config.py` or create a custom configuration:

```python
from tt_scale.config import Config

config = Config(
    model_name="your-model",
    method="collective",
    tau=0.7,
    max_total_branches=12,
    # ... other parameters
)
```

## Output

The benchmark script generates:

1. **Summary CSV** (`--summary-csv`): Overall results with aggregated statistics
2. **Detail Logs** (`logs/` directory): Per-question results with token counts
3. **Console Output**: Real-time progress and per-question statistics

Each log entry includes:
- Question index, model answer, gold answer, correctness
- Total tokens generated per question
- Step counts, retries, backtracks, jumps

## Search Algorithms

- **Collective Backtrack**: All branches backtrack together if insufficient branches pass threshold
- **Independent Backtrack**: Each branch backtracks independently
- **Beam Search**: Traditional beam search with PRM scoring
- **Best-of-N**: Generate N candidates and select the best based on PRM scores

## Troubleshooting

**Out of Memory (OOM):**
- Reduce `--gpu-mem-util` (e.g., 0.3)
- Use a smaller model
- Reduce `k_tokens` or `max_total_branches`

**Slow Performance:**
- Increase `--tensor-parallel-size` if you have multiple GPUs
- Reduce `num_samples` for quick testing
- Adjust `max_steps` to limit search depth

**Poor Results:**
- Adjust `tau` threshold (higher = more selective)
- Increase `max_total_branches` for more exploration
- Try different `agg` methods for PRM score aggregation
- Experiment with different PRM models
