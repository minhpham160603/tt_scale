import re
import numpy as np
import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset
from typing import List, Optional, Tuple
from tt_scale.prm.logits_prm import LogitsPRM

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_NAME = "Qwen/Qwen3-0.6B"
LOGITS_PRM_MODEL = "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"

NUM_SAMPLES = 30
MAX_TOKENS = 2048

# Best of N Settings
N = 4
TEMPERATURE_GEN = 0.8

# --- PRM SELECTION (Uncomment one) ---

# [OPTION A] Self-Correction: Use the Generator itself to score via prompting
PRM_MODE = "self_correction"
# GPU_MEMORY_UTIL = 0.95  # Use almost all VRAM for the single model

# [OPTION B] Logits PRM: Use a separate finetuned PRM (Loads a 2nd model!)
# PRM_MODE = "logits"
# GPU_MEMORY_UTIL = 0.6   # Reduce Generator VRAM to make room for PRM

# Automatically set memory utilization based on mode if not manually set above
if 'GPU_MEMORY_UTIL' not in locals():
    GPU_MEMORY_UTIL = 0.95 if PRM_MODE == "self_correction" else 0.6

# ==========================================
# Helper Functions
# ==========================================

def extract_math_answer(text: str) -> Optional[float]:
    """Robust extraction for MATH dataset answers (\boxed{...}) or simple numbers."""
    # 1. Prefer Boxed LaTeX
    boxed_match = re.search(r"\\boxed{([^}]+)}", text)
    if boxed_match:
        content = boxed_match.group(1).strip()
        content = content.replace('$', '').replace('\\', '').replace(',', '')
        try:
            nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", content)
            if nums: return float(nums[-1])
        except: pass

    # 2. Fallback: Look for the last number in the text
    try:
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text.replace(',', ''))
        if nums: return float(nums[-1])
    except: pass
    return None

def extract_score_text(text: str) -> float:
    """Extracts a score (0-10) from the judge's text response (Option A)."""
    match = re.search(r"(?:Score:\s*)?(\d+(\.\d+)?)", text)
    if match:
        try:
            val = float(match.group(1))
            return max(0.0, min(10.0, val))
        except: pass
    return 0.0

def create_judge_prompt(tokenizer, problem, candidate_answer):
    """Creates the prompt for Self-Correction (Option A)."""
    sys_prompt = "You are a strict math grader. Review the problem and the proposed answer. Assign a score from 1 to 10 based on correctness. Output ONLY the score, e.g., 'Score: 8'."
    user_content = f"Problem: {problem}\n\nProposed Answer:\n{candidate_answer}\n\nPlease rate the correctness of the answer above from 1 to 10."

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ==========================================
# Main Execution
# ==========================================

def main():
    print(f"--- Configuration: Mode=[{PRM_MODE}] | GPU Util=[{GPU_MEMORY_UTIL}] ---")

    # 1. Load Data
    print(f"Loading gsm8k (First {NUM_SAMPLES} samples)...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    # dataset = load_dataset("openai/gsm8k", "main", split="test")
    dataset = dataset.select(range(NUM_SAMPLES))
    problems = [sample.get('problem', sample.get('question')) for sample in dataset]
    ground_truths = [sample.get('solution', sample.get('answer')) for sample in dataset]
    truth_vals = [extract_math_answer(t) for t in ground_truths]

    # 2. Initialize Generator (vLLM)
    print(f"Loading Generator {MODEL_NAME}...")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEMORY_UTIL,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=4096,
    )
    tokenizer = llm.get_tokenizer()

    # 3. Initialize PRM (If Option B)
    prm_engine = None
    if PRM_MODE == "logits":
        print(f"Loading Logits PRM {LOGITS_PRM_MODEL}...")
        # Note: LogitsPRM uses standard HF, so it will allocate remaining VRAM
        prm_engine = LogitsPRM(model_name=LOGITS_PRM_MODEL, device="cuda")

    # 4. Phase 1: Generate N Candidates (Batch)
    print(f"\n--- Phase 1: Generating {len(problems) * N} candidates ---")
    gen_params = SamplingParams(
        temperature=TEMPERATURE_GEN,
        max_tokens=MAX_TOKENS,
        n=N, # Generate N sequences per prompt
        top_p=0.9,
    )

    # Pre-format prompts for Qwen
    formatted_prompts = []
    for p in problems:
        msgs = [{"role": "system", "content": "You are a helpful math solver. Solve step-by-step. Put final answer in \\boxed{}."},
                {"role": "user", "content": p}]
        formatted_prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    gen_outputs = llm.generate(formatted_prompts, gen_params, use_tqdm=True)

    # Organize candidates: List of Lists [[cand1, cand2...], [cand1, cand2...]]
    all_candidates: List[List[str]] = []
    for output in gen_outputs:
        all_candidates.append([o.text for o in output.outputs])

    # 5. Phase 2: Score Candidates (Batch)
    print(f"\n--- Phase 2: Scoring Candidates (Mode: {PRM_MODE}) ---")

    # Flatten for processing
    flat_candidates = [cand for sublist in all_candidates for cand in sublist]
    flat_problems = [p for p in problems for _ in range(N)]
    flat_scores = []

    if PRM_MODE == "self_correction":
        # Option A: Use vLLM to generate scores
        judge_prompts = [
            create_judge_prompt(tokenizer, flat_problems[i], flat_candidates[i])
            for i in range(len(flat_candidates))
        ]
        score_params = SamplingParams(temperature=0.0, max_tokens=10)
        score_outputs = llm.generate(judge_prompts, score_params, use_tqdm=True)
        flat_scores = [extract_score_text(out.outputs[0].text) for out in score_outputs]

    elif PRM_MODE == "logits":
        # Option B: Use external LogitsPRM
        # Prepare pairs for the PRM batch function
        qa_pairs = list(zip(flat_problems, flat_candidates))
        # LogitsPRM.get_scores_batch handles the HF inference
        flat_scores = prm_engine.get_scores_batch(qa_pairs)

    # 6. Phase 3: Select Best & Evaluate
    print(f"\n--- Phase 3: Selection & Evaluation ---")
    correct_count = 0

    for i in range(len(problems)):
        # Slice the flattened scores for this problem
        start_idx = i * N
        problem_scores = flat_scores[start_idx : start_idx + N]
        problem_candidates = all_candidates[i]

        # Argmax
        best_idx = np.argmax(problem_scores)
        best_candidate = problem_candidates[best_idx]
        best_score = problem_scores[best_idx]

        # Check Truth
        pred_val = extract_math_answer(best_candidate)
        is_correct = False
        if pred_val is not None and truth_vals[i] is not None:
            is_correct = abs(pred_val - truth_vals[i]) < 1e-3

        if is_correct: correct_count += 1

        status = "✅" if is_correct else "❌"
        print(f"Sample {i+1}: {status} | Score: {best_score:.3f} | Pred: {pred_val} | Truth: {truth_vals[i]}")

    accuracy = correct_count / NUM_SAMPLES
    print(f"\n=== Best-of-{N} ({PRM_MODE}) Accuracy: {accuracy:.2%} ({correct_count}/{NUM_SAMPLES}) ===")

if __name__ == "__main__":
    main()