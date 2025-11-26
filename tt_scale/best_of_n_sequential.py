import re
import numpy as np
from vllm import LLM, SamplingParams
from datasets import load_dataset

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
NUM_SAMPLES = 50
MAX_TOKENS = 2048

# Best of N Settings
N = 4  # Number of candidates to generate per problem
TEMPERATURE_GEN = 0.8  # Must be > 0 to get diverse candidates
TEMPERATURE_SCORE = 0.0 # Greedy for the grading phase

def extract_math_answer(text):
    """
    Robust extraction for MATH dataset answers (\boxed{...}) or simple numbers.
    """
    # 1. Prefer Boxed LaTeX
    boxed_match = re.search(r"\\boxed{([^}]+)}", text)
    if boxed_match:
        content = boxed_match.group(1).strip()
        content = content.replace('$', '').replace('\\', '').replace(',', '')
        try:
            nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", content)
            if nums: return float(nums[-1])
        except: pass

    # 2. Fallback: Look for the last number
    try:
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text.replace(',', ''))
        if nums: return float(nums[-1])
    except: pass
    return None

def extract_score(text):
    """
    Extracts a score (0-10) from the judge's response.
    """
    # Look for "Score: X" or just numbers at the end
    match = re.search(r"(?:Score:\s*)?(\d+(\.\d+)?)", text)
    if match:
        try:
            val = float(match.group(1))
            return max(0.0, min(10.0, val))
        except: pass
    return 0.0

def create_judge_prompt(tokenizer, problem, candidate_answer):
    """
    Creates the prompt for the model to evaluate its own answer.
    """
    sys_prompt = "You are a strict math grader. Review the problem and the proposed answer. Assign a score from 1 to 10 based on correctness. Output ONLY the score, e.g., 'Score: 8'."

    user_content = f"""
    Problem: {problem}

    Proposed Answer:
    {candidate_answer}

    Please rate the correctness of the answer above from 1 to 10.
    """

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    # 1. Load Data
    print(f"Loading MATH-500 (First {NUM_SAMPLES} samples)...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    dataset = dataset.select(range(NUM_SAMPLES))

    # 2. Initialize Model
    print(f"Loading {MODEL_NAME}...")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8, # Slightly lower to be safe
        trust_remote_code=True,
        dtype="float16",
        max_model_len=4096,
        enforce_eager=True # Helps with memory fragmentation in sequential loops sometimes
    )
    tokenizer = llm.get_tokenizer()

    # Define Sampling Params
    # 1. Generation Params (High Temp for Diversity)
    gen_params = SamplingParams(temperature=TEMPERATURE_GEN, max_tokens=MAX_TOKENS)
    # 2. Scoring Params (Greedy for Consistency)
    score_params = SamplingParams(temperature=TEMPERATURE_SCORE, max_tokens=20)

    correct_count = 0

    print(f"\n--- Starting Best-of-{N} Sequential Run ---")

    for i, sample in enumerate(dataset):
        problem = sample.get('problem', sample.get('question'))
        ground_truth = sample.get('solution', sample.get('answer'))
        truth_val = extract_math_answer(ground_truth)

        # --- Phase 1: Generate N Candidates Sequentially ---
        candidates = []

        # Prepare the prompt for generation
        gen_messages = [
            {"role": "system", "content": "You are a helpful math solver. Solve the problem step-by-step. Put your final answer within \\boxed{}."},
            {"role": "user", "content": problem}
        ]
        gen_prompt = tokenizer.apply_chat_template(gen_messages, tokenize=False, add_generation_prompt=True)

        # We loop N times to generate diverse answers
        # (Note: passing n=N to SamplingParams is faster, but this loop is stricter on memory if that's the concern)
        for _ in range(N):
            output = llm.generate([gen_prompt], gen_params, use_tqdm=False)
            candidates.append(output[0].outputs[0].text)

        # --- Phase 2: Score Candidates (Self-Correction) ---
        scores = []
        for cand in candidates:
            # Create a prompt asking the model to judge this specific candidate
            judge_prompt = create_judge_prompt(tokenizer, problem, cand)

            # Generate score
            score_output = llm.generate([judge_prompt], score_params, use_tqdm=False)
            score_text = score_output[0].outputs[0].text
            scores.append(extract_score(score_text))

        # --- Phase 3: Select Best ---
        best_idx = np.argmax(scores)
        best_candidate = candidates[best_idx]
        best_score = scores[best_idx]

        # Extract Answer from the best candidate
        pred_val = extract_math_answer(best_candidate)

        # Check correctness
        is_correct = False
        if pred_val is not None and truth_val is not None:
            is_correct = abs(pred_val - truth_val) < 1e-3

        if is_correct:
            correct_count += 1

        print(f"Sample {i+1}: {'✅' if is_correct else '❌'} | Score: {best_score} | Pred: {pred_val} | Truth: {truth_val}")
        # Optional: Print distribution of scores to see if the model is actually differentiating
        # print(f"    Scores: {scores}")

    accuracy = correct_count / NUM_SAMPLES
    print(f"\n=== Best-of-{N} Accuracy: {accuracy:.2%} ({correct_count}/{NUM_SAMPLES}) ===")

if __name__ == "__main__":
    main()
