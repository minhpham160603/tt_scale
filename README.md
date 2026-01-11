# tt_scale

- Parallel hybrid search with vLLM for multi-branch expansion and backtracking

Key files and symbols:
- Parallel hybrid engine: [`tt_scale.tt_scale.vllm_hybrid_parallel_engine`](tt_scale/tt_scale/vllm_hybrid_parallel_engine.py)
  - Generator: [`tt_scale.vllm_hybrid_parallel_engine.VLLMGenerator`](tt_scale/tt_scale/vllm_hybrid_parallel_engine.py)
  - PRMs: [`tt_scale.vllm_hybrid_parallel_engine.VLLMPRM`](tt_scale/tt_scale/vllm_hybrid_parallel_engine.py)
  - Searcher: [`tt_scale.vllm_hybrid_parallel_engine.HybridSearcher`](tt_scale/tt_scale/vllm_hybrid_parallel_engine.py) with [`run_parallel`](tt_scale/tt_scale/vllm_hybrid_parallel_engine.py)
  - Answer extractors: [`tt_scale.vllm_hybrid_parallel_engine.extract_result`](tt_scale/tt_scale/vllm_hybrid_parallel_engine.py), [`tt_scale.vllm_hybrid_parallel_engine.extract_math_answer`](tt_scale/tt_scale/vllm_hybrid_parallel_engine.py)
  - Evaluations: [`tt_scale.vllm_hybrid_parallel_engine.test_gsm8k`](tt_scale/tt_scale/vllm_hybrid_parallel_engine.py), [`tt_scale.vllm_hybrid_parallel_engine.test_MathArena`](tt_scale/tt_scale/vllm_hybrid_parallel_engine.py)


- PRMs:
  - Logits PRM: [`tt_scale.prm.logits_prm.LogitsPRM`](tt_scale/tt_scale/prm/logits_prm.py)
  - Custom LLM-as-judge (HF): [`tt_scale.prm.custom_prm.CustomPRM`](tt_scale/tt_scale/prm/custom_prm.py)
  - Classifier head PRM: [`tt_scale.prm.classifier_prm.ClassifierPRM`](tt_scale/tt_scale/prm/classifier_prm.py)
  - Qwen Math PRM : [`tt_scale.prm.qwen_math_prm.QwenMathPRM`](tt_scale/tt_scale/prm/qwen_math_prm.py)
  - vLLM CoT PRM: [`tt_scale.prm.vllm_cot_prm.VLLMCoTPRM`](tt_scale/tt_scale/prm/vllm_cot_prm.py)


## Installation

- Python 3.10+
- CUDA GPU recommended (vLLM)
- Dependencies in [`pyproject.toml`](pyproject.toml)

```bash
python -m venv .env
source .env/bin/activate
pip install -e .

# Core deps (if not auto-resolved)
pip install torch transformers datasets bitsandbytes accelerate vllm
```

## Quickstart

- Parallel hybrid search (default: AIME/MATH datasets with Qwen Math PRM):
```bash
python -m tt_scale.vllm_hybrid_parallel_engine
```


## How it works

```text
Algorithm: Parallel Collective Search (Swarm)
Input: Prompt P, Min Passing K
Branches ← Init(P) × N
while not Finished:
    Outputs ← GenerateBatch(Branches)
    Scores ← PRM(Outputs)
    Passing ← Count(Scores > τ)
    if Passing ≥ K:
        Advance: Keep Passing Branches
    else:
        Global Backtrack: Revert ALL branches
        Branches ← KeepBestPrevious(Branches)
        Increase Temperature
return Best Answer
```
![Parallel Backtracking Illustration](./Sichao's_version_of_parallel_backtracking.jpg)

Core loop:
- Build context: [`tt_scale.vllm_hybrid_parallel_engine.VLLMGenerator.build_input_context`](tt_scale/tt_scale/vllm_hybrid_parallel_engine.py)
- Batch expand M candidates/branch: [`tt_scale.vllm_hybrid_parallel_engine.VLLMGenerator.generate_batch_step`](tt_scale/tt_scale/vllm_hybrid_parallel_engine.py)
- Batch scoring: e.g., [`tt_scale.prm.qwen_math_prm.QwenMathPRM.get_scores_batch`](tt_scale/tt_scale/prm/qwen_math_prm.py)
- Search and prune/backtrack: [`tt_scale.vllm_hybrid_parallel_engine.HybridSearcher.run_parallel`](tt_scale/tt_scale/vllm_hybrid_parallel_engine.py)

## Configuration (parallel engine)

Edit these at the top of [`tt_scale/tt_scale/vllm_hybrid_parallel_engine.py`](tt_scale/tt_scale/vllm_hybrid_parallel_engine.py):

- Model and sampling:
  - `MODEL_NAME`, `K_TOKENS`, `TEMP_ORIGIN`, `TEMP_STEP`
- PRM thresholding:
  - `TAU` (keep threshold), `MAX_BACKTRACKS`
- Parallel control:
  - `MAX_TOTAL_BRANCHES`, `PASSING_MINIMUM`, `KEEPING_BRANCHES`, `MAX_FINISHED_BRANCHES`
- Context limit:
  - `MAX_MODEL_LEN`

Switch PRM in `main()`:
```python
# filepath: /home/sichma/tt_scale/tt_scale/vllm_hybrid_parallel_engine.py
# ...existing code...
gen = VLLMGenerator(engine)
# prm = VLLMPRM(engine)     # LLM-as-judge with vLLM
# prm = LogitsPRM()         # Logits-based PRM
prm = QwenMathPRM()         # Step-wise Qwen Math PRM (default)
# prm = VLLMCoTPRM(engine)  # CoT-style judge
searcher = HybridSearcher(gen, prm, 512)
# ...existing code...
```

## Datasets and evaluation

- MATH-500 (default in some scripts): HuggingFaceH4/MATH-500
- AIME (MathArena): MathArena/aime_2025



## Troubleshooting

- OOM with vLLM:
  - Lower `gpu_memory_utilization` in `LLM(...)`, reduce `K_TOKENS`, or pick a smaller `MODEL_NAME`.
- Stuck mid-generation:
  - Check stop delimiter and final prefix in the system prompt of [`tt_scale.vllm_hybrid_parallel_engine.VLLMGenerator`](tt_scale/tt_scale/vllm_hybrid_parallel_engine.py).
- Over/under-pruning:
  - Adjust `TAU`, `PASSING_MINIMUM`, `KEEPING_BRANCHES`, or use a different PRM.




