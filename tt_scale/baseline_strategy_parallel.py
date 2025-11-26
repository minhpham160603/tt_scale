import re
from vllm import LLM, SamplingParams
from datasets import load_dataset
from typing import Optional

# --- Configuration (UPDATED MODEL) ---
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
NUM_SAMPLES = 100
MAX_TOKENS = 2048

def extract_math_answer(text: str) -> Optional[float]:
    """
    Robust extraction for MATH dataset answers (\boxed{...}) or simple numbers.
    """
    # 1. Prefer Boxed LaTeX
    boxed_match = re.search(r"\\boxed{([^}]+)}", text)
    if boxed_match:
        content = boxed_match.group(1).strip()
        content = content.replace('$', '').replace('\\', '').replace(',', '')
        try:
            # Try finding the last number in the box
            nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", content)
            if nums: return float(nums[-1])
        except: pass

    # 2. Fallback: Look for the last number in the text
    try:
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text.replace(',', ''))
        if nums: return float(nums[-1])
    except: pass
    return None

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
        gpu_memory_utilization=0.95, # Aggressively use GPU for large batch
        trust_remote_code=True,
        dtype="float16",
        max_model_len=4096
    )
    tokenizer = llm.get_tokenizer()

    # 3. Prepare Prompts
    prompts = []
    problems = []
    ground_truths = []

    for sample in dataset:
        problem = sample.get('problem', sample.get('question'))
        ground_truth = sample.get('solution', sample.get('answer'))

        problems.append(problem)
        ground_truths.append(ground_truth)

        # Apply Qwen's recommended math prompt template
        messages = [
            {"role": "system", "content": "You are a helpful math solver. Solve the problem step-by-step. Put your final answer within \\boxed{}."},
            {"role": "user", "content": problem}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    # 4. Generate (Standard Greedy Decoding in one large batch)
    print(f"Generating {len(prompts)} responses in a single batch...")
    # CRITICAL: temperature=0.0 ensures simple, deterministic greedy decoding
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, n=1)
    outputs = llm.generate(prompts, params, use_tqdm=True)

    # 5. Evaluate
    correct = 0
    print("\n--- Evaluation ---")

    for i, output in enumerate(outputs):
        gen_text = output.outputs[0].text

        gen_ans = extract_math_answer(gen_text)
        truth_ans = extract_math_answer(ground_truths[i])

        is_correct = False
        if gen_ans is not None and truth_ans is not None:
            # Standard tolerance check for math
            is_correct = abs(gen_ans - truth_ans) < 1e-3
            if is_correct: correct += 1

        status = "✅" if is_correct else "❌"
        print(f"Sample {i+1}: {status} | Truth: {truth_ans} | Pred: {gen_ans}")

    accuracy = correct / NUM_SAMPLES
    print(f"\n=== Greedy Baseline (T=0) Accuracy: {accuracy:.2%} ({correct}/{NUM_SAMPLES}) ===")

if __name__ == "__main__":
    main()
