import time
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams

# Import Utils and Config
try:
    from .config import MODEL_NAME, UNIFIED_SYS_PROMPT, extract_math_answer
    from .experiment_utils import save_experiment_results, calculate_metrics
except ImportError:
    from config import MODEL_NAME, UNIFIED_SYS_PROMPT, extract_math_answer
    from experiment_utils import save_experiment_results, calculate_metrics

# ==========================================
# CONFIGURATION
# ==========================================
EXPERIMENT_NAME = "baseline_greedy_parallel"

# Realistic settings for a solid benchmark run
NUM_SAMPLES = 500        # Full MATH-500 set is standard
MAX_TOKENS = 2048        # Sufficient for CoT reasoning
TEMPERATURE = 0.0        # Greedy = 0.0 for Baseline
GPU_MEMORY_UTIL = 0.90   # High utilization for max parallel batching

def main():
    # 1. Setup Configuration Object
    config = {
        "model_name": MODEL_NAME,
        "num_samples": NUM_SAMPLES,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "gpu_memory_utilization": GPU_MEMORY_UTIL,
        "strategy": "vllm_continuous_batching"
    }

    # 2. Load Data
    print(f"--- Starting {EXPERIMENT_NAME} (N={NUM_SAMPLES}) ---")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    # Select specific range (e.g., first N samples)
    # Using the full requested size, or the dataset max if smaller
    limit = min(len(dataset), NUM_SAMPLES)
    dataset = dataset.select(range(limit))

    # 3. Initialize Model (vLLM Parallel Engine)
    print(f"Initializing vLLM Engine [{MODEL_NAME}]...")
    print(f" -> GPU Memory Util: {GPU_MEMORY_UTIL}")

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEMORY_UTIL,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=4096,
        enforce_eager=False # False = Enable CUDA graph optimizations (Faster)
    )
    tokenizer = llm.get_tokenizer()

    # 4. Prepare Prompts (Batch Pre-processing)
    print("Formatting prompts...")
    prompts = []
    raw_problems = []
    ground_truths = []

    for sample in dataset:
        p = sample.get('problem', sample.get('question'))
        t = sample.get('solution', sample.get('answer'))

        raw_problems.append(p)
        ground_truths.append(t)

        messages = [
            {"role": "system", "content": UNIFIED_SYS_PROMPT},
            {"role": "user", "content": p}
        ]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(full_text)

    # 5. Parallel Generation (The "Realistic" Batch)
    print(f"Parallel Generation of {len(prompts)} requests...")

    params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        n=1 # One output per prompt (Greedy)
    )

    start_time = time.time()

    # This single line executes the whole batch in parallel on the GPU
    outputs = llm.generate(prompts, params, use_tqdm=True)

    duration = time.time() - start_time
    print(f"Generation finished in {duration:.2f}s ({(len(prompts)/duration):.1f} req/s)")

    # 6. Scoring & Processing
    results = []
    print("Scoring responses...")

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text

        # Parse
        pred_val = extract_math_answer(generated_text)
        truth_val = extract_math_answer(ground_truths[i])

        # Verify
        is_correct = False
        if pred_val is not None and truth_val is not None:
            is_correct = abs(pred_val - truth_val) < 1e-3

        # Store Result
        sample_result = {
            "id": i,
            "problem": raw_problems[i],
            "prediction_full": generated_text,
            "prediction_extracted": pred_val,
            "truth_full": ground_truths[i],
            "truth_extracted": truth_val,
            "is_correct": is_correct
        }
        results.append(sample_result)

    # 7. Metrics & Save
    metrics = calculate_metrics(results, duration)

    save_path = save_experiment_results(
        experiment_name=EXPERIMENT_NAME,
        config=config,
        results=results,
        metrics=metrics
    )

    # 8. Clean Summary
    print(f"\n{'='*40}")
    print(f"BASELINE PARALLEL SUMMARY")
    print(f"{'='*40}")
    print(f"Accuracy : {metrics['accuracy']:.2%} ({metrics['correct_count']}/{metrics['total_samples']})")
    print(f"TPS      : {(metrics['total_samples'] * MAX_TOKENS / duration):.0f} (Approx tokens/sec)")
    print(f"Saved to : {save_path}")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()
