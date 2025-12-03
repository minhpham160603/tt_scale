import random
import re
import copy
import time
import torch
import numpy as np
from typing import Optional, List, Tuple, Union
from datasets import load_dataset
from vllm import LLM, SamplingParams

# Import Unified Configuration
try:
    from .config import (
        MODEL_NAME,
        LOGITS_PRM_MODEL,
        UNIFIED_SYS_PROMPT,
        PRM_SYS_PROMPT,
        STOP_STRING,
        extract_math_answer
    )
    from .experiment_utils import save_experiment_results, calculate_metrics
except ImportError:
    from config import (
        MODEL_NAME,
        LOGITS_PRM_MODEL,
        UNIFIED_SYS_PROMPT,
        PRM_SYS_PROMPT,
        STOP_STRING,
        extract_math_answer
    )
    from experiment_utils import save_experiment_results, calculate_metrics

# Optional: Dedicated PRM import
try:
    from tt_scale.prm.logits_prm import LogitsPRM
except ImportError:
    pass

# ==========================================
# HYPERPARAMETERS & CONFIG
# ==========================================
EXPERIMENT_NAME = "hybrid_tree_search_mitosis"

# Search Parameters
MAX_WIDTH = 16           # Width of the beam/population
EXPANSION_FACTOR = 2     # How many clones to spawn on failure
MAX_STEPS = 8            # Max reasoning depth
TAU = 0.5                # Score threshold (0.0 to 1.0)
TEMPERATURE = 0.7        # High temp for diversity in clones
K_TOKENS = 256           # Max tokens per step

# Mode Selection
# "A" for Self-Correction (Prompted), "B" for External Logits PRM
PRM_MODE = "B"
VERBOSE = True
NUM_SAMPLES = 500

# ==========================================
# Data Structures
# ==========================================

class SearchCandidate:
    def __init__(self, history_text: str, cumulative_score: float = 0.0, depth: int = 0):
        self.history = history_text
        self.cum_score = cumulative_score
        self.depth = depth
        self.is_finished = False
        self.id = f"{depth}_{random.randint(0, 99999)}"

    def copy(self):
        """Creates a deep copy for branching."""
        new_cand = SearchCandidate(self.history, self.cum_score, self.depth)
        return new_cand

    @property
    def avg_score(self):
        return self.cum_score / max(1, self.depth)

    @property
    def priority_score(self):
        # Weight depth slightly to prevent the search from getting stuck
        # optimizing the root node forever.
        return self.avg_score + (0.05 * self.depth)

# ==========================================
# Component Classes
# ==========================================

class VLLMGeneratorWrapper:
    def __init__(self, llm_engine):
        self.llm = llm_engine
        self.tokenizer = self.llm.get_tokenizer()

    def format_prompts(self, questions: List[str], partial_histories: List[str]) -> List[str]:
        formatted = []
        for q, hist in zip(questions, partial_histories):
            messages = [
                {"role": "system", "content": UNIFIED_SYS_PROMPT},
                {"role": "user", "content": q},
            ]
            base_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_text = base_prompt + hist
            formatted.append(full_text)
        return formatted

    def generate_batch_steps(self, full_prompts: List[str]) -> List[str]:
        params = SamplingParams(
            temperature=TEMPERATURE,
            max_tokens=K_TOKENS,
            top_p=0.9,
            stop=[STOP_STRING, self.tokenizer.eos_token]
        )
        outputs = self.llm.generate(full_prompts, params, use_tqdm=False)

        results = []
        for output in outputs:
            text = output.outputs[0].text
            if STOP_STRING in text:
                text = text.split(STOP_STRING)[0]
            results.append(text.strip())
        return results

class VLLMCoTPRMWrapper:
    """Wrapper to use the Generator itself as a Verifier via prompting."""
    def __init__(self, llm_engine):
        self.llm = llm_engine
        self.tokenizer = self.llm.get_tokenizer()

    def get_scores_batch(self, qa_pairs: List[Tuple[str, str]]) -> List[float]:
        prompts = []
        for q, full_ans in qa_pairs:
            judge_prompt = f"""
Review the math solution below.
---
Question: {q}
Solution So Far: {full_ans}
---
Rate the correctness of the logical flow. Give a score 1-10.
Output ONLY the number.
"""
            msgs = [
                {"role": "system", "content": PRM_SYS_PROMPT},
                {"role": "user", "content": judge_prompt}
            ]
            txt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(txt)

        params = SamplingParams(temperature=0.0, max_tokens=10)
        outputs = self.llm.generate(prompts, params, use_tqdm=False)

        scores = []
        for out in outputs:
            txt = out.outputs[0].text
            match = re.search(r"(\d+(\.\d+)?)", txt)
            val = 0.0
            if match:
                try:
                    val = max(1.0, min(10.0, float(match.group(1)))) / 10.0
                except: pass
            scores.append(val)
        return scores

# ==========================================
# The Parallel Hybrid Search Algorithm
# ==========================================

class ParallelHybridSearcher:
    def __init__(self, generator_wrapper, prm_engine):
        self.gen = generator_wrapper
        self.prm = prm_engine

    def run(self, raw_question: str) -> str:
        if VERBOSE: print(f"\n--- Starting Search: {raw_question[:40]}... ---")

        # 1. Initialize Wavefront
        wavefront = [SearchCandidate(history_text="", cumulative_score=0.0, depth=0)]
        finished_candidates = []

        # 2. Main Evolution Loop
        for depth in range(MAX_STEPS):
            if not wavefront:
                if VERBOSE: print("  -> All branches pruned. Stopping.")
                break

            # --- A. Batch Generation ---
            batch_prompts = self.gen.format_prompts(
                [raw_question] * len(wavefront),
                [cand.history for cand in wavefront]
            )
            new_steps = self.gen.generate_batch_steps(batch_prompts)

            # --- B. Batch Verification Setup ---
            prm_inputs = []
            candidates_for_prm = []

            for i, cand in enumerate(wavefront):
                step_text = new_steps[i]

                # Check for Immediate Finish using \boxed{}
                if "\\boxed{" in step_text:
                    fin_cand = cand.copy()
                    fin_cand.history += "\n" + step_text
                    fin_cand.cum_score += 1.0
                    fin_cand.depth += 1
                    fin_cand.is_finished = True
                    finished_candidates.append(fin_cand)
                    continue

                if not step_text:
                    continue

                full_text_to_score = cand.history + "\n" + step_text
                prm_inputs.append((raw_question, full_text_to_score))
                candidates_for_prm.append((cand, step_text))

            if not candidates_for_prm:
                if not finished_candidates:
                    break
                if not candidates_for_prm:
                    break

            # --- C. Run Batch PRM ---
            scores = self.prm.get_scores_batch(prm_inputs)

            # --- D. Dynamic Selection & Expansion ---
            next_wavefront = []
            passers = []
            failers = []

            for i, (cand, step_text) in enumerate(candidates_for_prm):
                score = scores[i]
                if score >= TAU:
                    passers.append((cand, step_text, score))
                else:
                    failers.append((cand, step_text, score))

            # 1. Process Passers
            for cand, step_text, score in passers:
                child = cand.copy()
                child.history += "\n" + step_text
                child.cum_score += score
                child.depth += 1
                next_wavefront.append(child)

            # 2. Process Failers (Mitosis)
            if failers:
                failers.sort(key=lambda x: x[2], reverse=True)
                for cand, step_text, score in failers:
                    if len(next_wavefront) >= MAX_WIDTH * 2:
                        break
                    # Spawn clones from the PARENT (retrying the step)
                    for _ in range(EXPANSION_FACTOR):
                        clone = cand.copy()
                        next_wavefront.append(clone)

            # 3. Final Truncation
            if next_wavefront:
                next_wavefront.sort(key=lambda c: c.priority_score, reverse=True)
                next_wavefront = next_wavefront[:MAX_WIDTH]

            wavefront = next_wavefront

        # 3. Final Selection
        all_results = finished_candidates
        if not all_results and wavefront:
            all_results = wavefront

        if not all_results:
            return ""

        best_cand = max(all_results, key=lambda c: c.avg_score)
        return best_cand.history

# ==========================================
# Main Execution
# ==========================================

def main():
    # 1. Configuration object for saving
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "model_name": MODEL_NAME,
        "prm_mode": PRM_MODE,
        "num_samples": NUM_SAMPLES,
        "max_width": MAX_WIDTH,
        "expansion_factor": EXPANSION_FACTOR,
        "max_steps": MAX_STEPS,
        "tau": TAU,
        "temperature": TEMPERATURE,
        "description": "Tree search with mitosis (cloning on failure)"
    }

    print(f"--- Starting {EXPERIMENT_NAME} ---")
    print(f"Model: {MODEL_NAME}")
    print(f"Config: Width={MAX_WIDTH}, Tau={TAU}, Expansion={EXPANSION_FACTOR}")

    gpu_util = 0.9 if PRM_MODE == "A" else 0.6

    # 2. Initialize Models
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=4,
        gpu_memory_utilization=gpu_util,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=4096,
        enforce_eager=True
    )

    generator = VLLMGeneratorWrapper(llm)
    prm = None

    if PRM_MODE == "A":
        prm = VLLMCoTPRMWrapper(llm)
    elif PRM_MODE == "B":
        try:
            from tt_scale.prm.logits_prm import LogitsPRM
            prm = LogitsPRM(model_name=LOGITS_PRM_MODEL, device="cuda")
        except ImportError:
            print("Could not import LogitsPRM.")
            return

    searcher = ParallelHybridSearcher(generator, prm)

    # 3. Load Data
    print(f"Loading MATH-500 (N={NUM_SAMPLES})...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    random.seed(42)
    indices = random.sample(range(len(dataset)), min(NUM_SAMPLES, len(dataset)))
    samples = [dataset[i] for i in indices]

    # 4. Run Search Loop
    results = []

    t0_total = time.time()

    for i, sample in enumerate(samples):
        q = sample.get('problem', sample.get('question'))
        truth_str = sample.get('solution', sample.get('answer'))
        truth = extract_math_answer(truth_str)

        print(f"\n[{i+1}/{NUM_SAMPLES}] Q: {q[:50]}...")

        # Run Search
        t0_sample = time.time()
        final_history = searcher.run(q)
        duration_sample = time.time() - t0_sample

        prediction = extract_math_answer(final_history)

        is_correct = False
        if prediction is not None and truth is not None:
            is_correct = abs(prediction - truth) < 1e-3

        status = "✅" if is_correct else "❌"
        print(f"Result: {status} | Pred: {prediction} | Time: {duration_sample:.2f}s")

        # Record Result
        results.append({
            "id": i,
            "problem": q,
            "ground_truth_full": truth_str,
            "ground_truth_val": truth,
            "generated_history": final_history,
            "prediction_val": prediction,
            "is_correct": is_correct,
            "sample_time": duration_sample
        })

    total_duration = time.time() - t0_total

    # 5. Save & Metrics
    metrics = calculate_metrics(results, total_duration)

    save_path = save_experiment_results(
        experiment_name=EXPERIMENT_NAME,
        config=config,
        results=results,
        metrics=metrics
    )

    print(f"\n{'='*40}")
    print(f"HYBRID SEARCH COMPLETED")
    print(f"{'='*40}")
    print(f"Accuracy: {metrics['accuracy']:.2%} ({metrics['correct_count']}/{metrics['total_samples']})")
    print(f"Duration: {total_duration:.2f}s (Avg {metrics['avg_time_per_sample']:.2f}s/prob)")
    print(f"Saved to: {save_path}")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()
