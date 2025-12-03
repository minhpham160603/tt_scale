import time
import numpy as np
import re
from typing import List, Dict, Any
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams

# Import Utils and Config
try:
    from .config import (
        MODEL_NAME,
        LOGITS_PRM_MODEL,
        UNIFIED_SYS_PROMPT,
        PRM_SYS_PROMPT,
        extract_math_answer
    )
    from .experiment_utils import save_experiment_results, calculate_metrics
except ImportError:
    from config import (
        MODEL_NAME,
        LOGITS_PRM_MODEL,
        UNIFIED_SYS_PROMPT,
        PRM_SYS_PROMPT,
        extract_math_answer
    )
    from experiment_utils import save_experiment_results, calculate_metrics

# ==========================================
# CONFIGURATION
# ==========================================
EXPERIMENT_NAME = "best_of_n_parallel"
NUM_SAMPLES = 500         # Full MATH-500
MAX_TOKENS = 2048
N = 4                     # Number of candidates per problem
TEMPERATURE_GEN = 0.5     # Diversity for candidates

# --- PRM MODE SELECTION ---
# "self_correction": Generator scores itself (1 Model, High VRAM usage)
# "logits": External PRM scores (2 Models, Split VRAM usage)
PRM_MODE = "self_correction"

# Dynamic Memory Allocation
# If "logits", we reserve 40% VRAM for the HF Verifier, leaving 60% for vLLM
GPU_MEMORY_UTIL = 0.95 if PRM_MODE == "self_correction" else 0.6

# ==========================================
# Helpers
# ==========================================

def extract_score_text(text: str) -> float:
    """Extracts a score (0-10) from the judge's text response."""
    match = re.search(r"(?:Score:\s*)?(\d+(\.\d+)?)", text)
    if match:
        try:
            val = float(match.group(1))
            return max(0.0, min(10.0, val))
        except: pass
    return 0.0

def create_judge_prompt(tokenizer, problem, candidate_answer):
    """Formats the self-correction prompt."""
    user_content = f"Problem: {problem}\n\nProposed Answer:\n{candidate_answer}\n\nPlease rate the correctness of the answer above from 1 to 10."
    messages = [
        {"role": "system", "content": PRM_SYS_PROMPT},
        {"role": "user", "content": user_content}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ==========================================
# Main Execution
# ==========================================

def main():
    # 1. Configuration Object
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "model_name": MODEL_NAME,
        "prm_mode": PRM_MODE,
        "prm_model": LOGITS_PRM_MODEL if PRM_MODE == "logits" else MODEL_NAME,
        "n_candidates": N,
        "num_samples": NUM_SAMPLES,
        "temperature_gen": TEMPERATURE_GEN,
        "gpu_utilization": GPU_MEMORY_UTIL
    }

    print(f"--- Starting {EXPERIMENT_NAME} (N={NUM_SAMPLES}, Best-of-{N}) ---")
    print(f"--- Mode: {PRM_MODE} | GPU Util: {GPU_MEMORY_UTIL} ---")

    # 2. Load Data
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    dataset = dataset.select(range(min(len(dataset), NUM_SAMPLES)))

    problems = [sample.get('problem', sample.get('question')) for sample in dataset]
    ground_truths = [sample.get('solution', sample.get('answer')) for sample in dataset]

    # 3. Initialize Generator (vLLM)
    print(f"Loading Generator: {MODEL_NAME}")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEMORY_UTIL,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=4096,
        enforce_eager=False
    )
    tokenizer = llm.get_tokenizer()

    # 4. Initialize PRM (If External)
    prm_engine = None
    if PRM_MODE == "logits":
        print(f"Loading External PRM: {LOGITS_PRM_MODEL}")
        try:
            from tt_scale.prm.logits_prm import LogitsPRM
            prm_engine = LogitsPRM(model_name=LOGITS_PRM_MODEL, device="cuda")
        except ImportError:
            print("❌ Error: Could not import LogitsPRM.")
            return

    # ==========================================
    # PHASE 1: Parallel Generation
    # ==========================================
    print(f"\n[Phase 1] Generating {len(problems) * N} candidates...")

    formatted_prompts = []
    for p in problems:
        msgs = [{"role": "system", "content": UNIFIED_SYS_PROMPT}, {"role": "user", "content": p}]
        formatted_prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    gen_params = SamplingParams(
        temperature=TEMPERATURE_GEN,
        max_tokens=MAX_TOKENS,
        n=N, # vLLM generates N sequences per prompt efficiently
        top_p=0.9,
    )

    t0 = time.time()
    gen_outputs = llm.generate(formatted_prompts, gen_params, use_tqdm=True)
    time_gen = time.time() - t0

    # Organize: [[C1..CN], [C1..CN]]
    all_candidates_text = []
    for output in gen_outputs:
        all_candidates_text.append([o.text for o in output.outputs])

    # ==========================================
    # PHASE 2: Batch Scoring
    # ==========================================
    print(f"\n[Phase 2] Scoring Candidates (Mode: {PRM_MODE})...")

    # Flatten for batch processing: [P1_C1, P1_C2 ... P2_C1, P2_C2...]
    flat_candidates = [c for sublist in all_candidates_text for c in sublist]
    flat_problems = [p for p in problems for _ in range(N)]
    flat_scores = []

    t1 = time.time()

    if PRM_MODE == "self_correction":
        # Option A: Generator grades itself via vLLM
        judge_prompts = [
            create_judge_prompt(tokenizer, flat_problems[i], flat_candidates[i])
            for i in range(len(flat_candidates))
        ]
        score_params = SamplingParams(temperature=0.0, max_tokens=10)
        score_outputs = llm.generate(judge_prompts, score_params, use_tqdm=True)
        flat_scores = [extract_score_text(out.outputs[0].text) for out in score_outputs]

    elif PRM_MODE == "logits":
        # Option B: External HF PRM scores the batch
        # Note: LogitsPRM usually runs on HF, so it might be slower than vLLM
        qa_pairs = list(zip(flat_problems, flat_candidates))
        flat_scores = prm_engine.get_scores_batch(qa_pairs)

    time_score = time.time() - t1

    # ==========================================
    # PHASE 3: Selection & Evaluation
    # ==========================================
    print(f"\n[Phase 3] Selecting & Aggregating Results...")

    results = []

    for i in range(len(problems)):
        # Reconstruct structure for this problem
        start_idx = i * N
        p_scores = flat_scores[start_idx : start_idx + N]
        p_cands = all_candidates_text[i]

        # 1. Select Best
        best_idx = np.argmax(p_scores)
        best_cand_text = p_cands[best_idx]
        best_score = p_scores[best_idx]

        # 2. Evaluate
        truth_val = extract_math_answer(ground_truths[i])
        pred_val = extract_math_answer(best_cand_text)

        is_correct = False
        if pred_val is not None and truth_val is not None:
            is_correct = abs(pred_val - truth_val) < 1e-3

        # 3. Store Detailed Data
        # We save ALL candidates to analyze ranker performance later
        candidates_data = []
        for j in range(N):
            candidates_data.append({
                "text": p_cands[j],
                "score": p_scores[j],
                "extracted_val": extract_math_answer(p_cands[j])
            })

        results.append({
            "id": i,
            "problem": problems[i],
            "ground_truth_full": ground_truths[i],
            "ground_truth_val": truth_val,
            "selected_pred_full": best_cand_text,
            "selected_pred_val": pred_val,
            "selected_score": best_score,
            "is_correct": is_correct,
            "candidates": candidates_data # Rich data for analysis
        })

    # ==========================================
    # Final Metrics & Save
    # ==========================================
    total_time = time_gen + time_score
    metrics = calculate_metrics(results, total_time)

    # Add timing breakdown to metrics
    metrics["time_generation"] = time_gen
    metrics["time_scoring"] = time_score
    metrics["avg_time_per_problem"] = total_time / len(problems)

    save_path = save_experiment_results(
        experiment_name=EXPERIMENT_NAME,
        config=config,
        results=results,
        metrics=metrics
    )

    print(f"\n{'='*40}")
    print(f"BEST-OF-{N} SUMMARY ({PRM_MODE})")
    print(f"{'='*40}")
    print(f"Accuracy    : {metrics['accuracy']:.2%} ({metrics['correct_count']}/{metrics['total_samples']})")
    print(f"Time (Gen)  : {time_gen:.2f}s")
    print(f"Time (Score): {time_score:.2f}s")
    print(f"Total Time  : {total_time:.2f}s")
    print(f"Results     : {save_path}")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()
