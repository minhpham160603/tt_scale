import random
import re
import torch
from typing import Optional, List, Tuple
from datasets import load_dataset
from vllm import LLM, SamplingParams

# Import Unified Configuration
from .config import (
    MODEL_NAME,
    LOGITS_PRM_MODEL,
    UNIFIED_SYS_PROMPT,
    PRM_SYS_PROMPT,
    STOP_STRING,
    FINAL_ANSWER_PREFIX,
    extract_math_answer
)

    from tt_scale.prm.logits_prm import LogitsPRM
    from tt_scale.prm.vllm_cot_prm import VLLMCoTPRM

# --- Configuration ---
K_TOKENS = 256
TAU = 0.7                # Score threshold to keep a step
MAX_BACKTRACKS = 3       # Max retries per step before backtracking
MAX_STEPS = 5            # Max depth of reasoning
NUM_SAMPLES = 50
SEED = 69
EPSILON = 1e-3
VERBOSE = True

class VLLMGenerator:
    """Wraps vLLM engine to handle step-by-step generation with specific stop tokens."""

    def __init__(self, llm_engine):
        self.llm = llm_engine
        self.tokenizer = self.llm.get_tokenizer()

    def build_input_context(self, question: str, partial_answer: str = "") -> str:
        # Uses the UNIFIED_SYS_PROMPT (Helpful math solver + \boxed{})
        messages = [
            {"role": "system", "content": UNIFIED_SYS_PROMPT},
            {"role": "user", "content": question},
        ]
        context = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return context + partial_answer

    def generate_step(self, full_context: str, retry_attempt: int = 0) -> Tuple[str, bool]:
        # Adaptive temperature:
        # Start at 0.5 to ensure some diversity on the first try (for scoring potential).
        # Increase slightly on retries to find new paths, but cap at 0.9 to prevent hallucination.
        temp = min(0.5 + (0.1 * retry_attempt), 0.9)

        params = SamplingParams(
            temperature=temp,
            max_tokens=K_TOKENS,
            top_p=0.9,
            stop=[STOP_STRING, self.tokenizer.eos_token]
        )

        outputs = self.llm.generate([full_context], params, use_tqdm=False)
        new_text = outputs[0].outputs[0].text

        # Check if we hit a stop string or true EOS
        finish_reason = outputs[0].outputs[0].finish_reason
        is_eos = (finish_reason == "stop" and STOP_STRING not in new_text)

        return new_text, is_eos

class VLLMCoTPRMWrapper:
    """Extension of base VLLMCoTPRM with specific prompting logic."""
    def __init__(self, llm_engine):
        self.llm = llm_engine
        self.tokenizer = self.llm.get_tokenizer()

    def get_score(self, question: str, partial_answer: str) -> float:
        # Uses the PRM_SYS_PROMPT (Strict grader)
        judge_prompt = f"""
Review the following partial solution to a math problem.
---
Question: {question}
Partial Answer So Far: {partial_answer}
---
Rate the logical correctness of the MOST RECENT STEP in the Partial Answer on a scale of 1 to 10.
Output ONLY the number.
"""
        messages = [
            {"role": "system", "content": PRM_SYS_PROMPT},
            {"role": "user", "content": judge_prompt}
        ]
        full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Greedy generation for consistent scoring
        params = SamplingParams(temperature=0.0, max_tokens=10)
        outputs = self.llm.generate([full_prompt], params, use_tqdm=False)

        match = re.search(r"(\d+(\.\d+)?)", outputs[0].outputs[0].text)
        if match:
            try:
                return max(1.0, min(10.0, float(match.group(1)))) / 10.0
            except ValueError:
                pass
        return 0.0

class HybridSearcher:
    """Implements Depth-First Search with Adaptive Backtracking."""

    def __init__(self, generator, prm, max_len=2048):
        self.gen = generator
        self.prm = prm
        self.max_len = max_len

    def run(self, raw_prompt: str) -> str:
        print(f"\n--- New Run: {raw_prompt[:40]}... ---")

        # Stack stores tuples of: (current_text_history, backtrack_count_for_this_node)
        checkpoints = [("", 0)]
        step_count = 0
        final_response = ""

        while checkpoints:
            current_generated, bt_count = checkpoints[-1]

            # 1. Termination Checks
            if step_count >= MAX_STEPS:
                print("  -> Max steps reached.")
                final_response = current_generated
                break

            # Check for \boxed{} as per Unified Prompt instructions
            if "\\boxed{" in current_generated:
                final_response = current_generated
                break

            # 2. Generate Next Step
            full_context = self.gen.build_input_context(raw_prompt, current_generated)
            if len(full_context) > self.max_len * 4:
                print("  -> Context limit reached.")
                break

            new_chunk, _ = self.gen.generate_step(full_context, retry_attempt=bt_count)

            if not new_chunk:
                # Model stuck or finished silently; backtrack
                checkpoints.pop()
                continue

            # 3. Verify Step
            full_candidate = current_generated + new_chunk
            score = self.prm.get_score(raw_prompt, full_candidate)

            # --- DECISION LOGIC ---

            if score >= TAU:
                # Case A: Success -> Advance
                print(f"  -> KEEP (Score {score:.2f})")
                checkpoints.append((full_candidate, 0))
                step_count += 1

                if "\\boxed{" in new_chunk:
                    final_response = full_candidate
                    break

            elif bt_count < MAX_BACKTRACKS:
                # Case B: Soft Failure -> Retry same step (Pop current, push same parent with +1 attempt)
                print(f"  -> REJECT (Score {score:.2f}) - Retrying ({bt_count + 1}/{MAX_BACKTRACKS})")
                checkpoints.pop()
                checkpoints.append((current_generated, bt_count + 1))

            else:
                # Case C: Hard Failure -> Recursive Backtrack
                print(f"  -> MAX RETRIES ({bt_count}). Initiating Backtrack...")
                checkpoints.pop() # Discard current node

                # Recursively move up the tree until we find a parent with retries left
                resumed = False
                while checkpoints:
                    prev_text, prev_bt = checkpoints.pop()
                    step_count -= 1 # We are moving up

                    if prev_bt < MAX_BACKTRACKS:
                        # Found a parent with attempts remaining
                        checkpoints.append((prev_text, prev_bt + 1))
                        print(f"     Resuming from depth {step_count} (Attempt {prev_bt + 1})")
                        resumed = True
                        break
                    else:
                        print(f"     Depth {step_count} also exhausted. Continuing up...")

                if not resumed:
                    print("  -> Root exhausted. Exiting.")
                    final_response = current_generated # Return whatever we have
                    break

        return final_response

# ==========================================
# Evaluation & Main
# ==========================================

def main():
    print(f"Initializing Generator: {MODEL_NAME}")

    # Initialize Generator (vLLM)
    engine = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=4096,
        enforce_eager=True
    )
    generator = VLLMGenerator(engine)

    prm = None

    # --- CONFIGURATION SELECTION ---

    # OPTION A: Prompted PRM (Uses the same vLLM engine)
    print("\n[Configuration A] Selected: Self-Correction via Prompting")
    prm = VLLMCoTPRMWrapper(engine)

    # OPTION B: Finetuned Logits PRM (Comment A and Uncomment B to use)
    # print("\n[Configuration B] Selected: Dedicated Logits PRM")
    # prm = LogitsPRM(
    #     model_name=LOGITS_PRM_MODEL,
    #     device="cuda",
    #     quantization_config=None
    # )

    if prm is None:
        raise ValueError("No PRM selected. Please uncomment Config A or B.")

    # Run Evaluation
    searcher = HybridSearcher(generator, prm)

    print(f"--- Loading MATH-500 (N={NUM_SAMPLES}) ---")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    # Shuffle and select
    random.seed(SEED)
    indices = random.sample(range(len(dataset)), min(NUM_SAMPLES, len(dataset)))
    samples = [dataset[i] for i in indices]

    correct = 0
    for i, sample in enumerate(samples):
        q = sample.get('problem', sample.get('question'))
        truth_str = sample.get('solution', sample.get('answer'))
        truth = extract_math_answer(truth_str)

        if VERBOSE:
            print(f"\n{'='*10} SAMPLE {i+1}/{len(samples)} {'='*10}")

        # Execute Search
        prediction_text = searcher.run(q)
        prediction = extract_math_answer(prediction_text)

        is_correct = False
        if prediction is not None and truth is not None:
            is_correct = abs(prediction - truth) < EPSILON
            if is_correct: correct += 1

        if VERBOSE:
            status = "✅ CORRECT" if is_correct else f"❌ WRONG (Expected: {truth})"
            print(f"Result: {status} | Pred: {prediction}")

    print(f"\n=== Final Accuracy: {correct}/{len(samples)} ({correct/len(samples):.2%}) ===")

if __name__ == "__main__":
    main()
