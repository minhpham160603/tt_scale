import re
from vllm import LLM, SamplingParams
from datasets import load_dataset

# Configuration
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
NUM_SAMPLES = 50
MAX_TOKENS = 2048

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
    print("Loading MATH-500...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    dataset = dataset.select(range(NUM_SAMPLES)) # Take first N samples

    # 2. Initialize Model
    print(f"Loading {MODEL_NAME}...")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=4096
    )
    tokenizer = llm.get_tokenizer()

    # 3. Prepare Prompts
    prompts = []
    for sample in dataset:
        problem = sample.get('problem', sample.get('question'))
        messages = [
            {"role": "system", "content": "You are a helpful math solver. Solve the problem step-by-step. Put your final answer within \\boxed{}."},
            {"role": "user", "content": problem}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    # 4. Generate (Standard Greedy Decoding)
    print("Generating responses...")
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    outputs = llm.generate(prompts, params)

    # 5. Evaluate
    correct = 0
    print("\n--- Evaluation ---")
    for i, output in enumerate(outputs):
        gen_text = output.outputs[0].text
        truth_text = dataset[i].get('solution', dataset[i].get('answer'))

        gen_ans = extract_math_answer(gen_text)
        truth_ans = extract_math_answer(truth_text)

        is_correct = False
        if gen_ans is not None and truth_ans is not None:
            is_correct = abs(gen_ans - truth_ans) < 1e-3
            if is_correct: correct += 1

        print(f"Sample {i+1}: {'✅' if is_correct else '❌'} | Truth: {truth_ans} | Pred: {gen_ans}")

    accuracy = correct / NUM_SAMPLES
    print(f"\n=== Baseline Accuracy: {accuracy:.2%} ({correct}/{NUM_SAMPLES}) ===")

if __name__ == "__main__":
    main()
