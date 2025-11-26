import random
import re
import copy
import torch
import numpy as np
from typing import Optional, List, Tuple, Union
from datasets import load_dataset
from vllm import LLM, SamplingParams

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
# LOGITS_PRM_MODEL = "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"

# Algorithm Hyperparameters
MAX_WIDTH = 16           # Width of the beam/population
EXPANSION_FACTOR = 2     # How many clones to spawn on failure
MAX_STEPS = 8            # Max reasoning depth
TAU = 0.5                # Score threshold (0.0 to 1.0)
TEMPERATURE = 0.8        # High temp for diversity in generation
K_TOKENS = 256           # Max tokens per step

# Protocol Constants
FINAL_ANSWER_PREFIX = "<FINAL>"
STOP_STRING = "<END_STEP>"
VERBOSE = True

# Mode Selection
# "A" for Self-Correction (Prompted), "B" for External Logits PRM
PRM_MODE = "A"

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
        # Create new instance with same data to avoid reference bugs
        new_cand = SearchCandidate(self.history, self.cum_score, self.depth)
        return new_cand

    @property
    def avg_score(self):
        return self.cum_score / max(1, self.depth)

# ==========================================
# Component Classes
# ==========================================

class VLLMGeneratorWrapper:
    SYS_PROMPT = f"""You are a genius problem solver.
Solve the problem step-by-step to avoid mistakes.
After **EVERY logical step** of reasoning, output the token {STOP_STRING}.
If all steps are completed, return final answer with `{FINAL_ANSWER_PREFIX}` prefix (e.g., `{FINAL_ANSWER_PREFIX} 16`)."""

    def __init__(self, llm_engine):
        self.llm = llm_engine
        self.tokenizer = self.llm.get_tokenizer()

    def format_prompts(self, questions: List[str], partial_histories: List[str]) -> List[str]:
        formatted = []
        for q, hist in zip(questions, partial_histories):
            messages = [
                {"role": "system", "content": self.SYS_PROMPT},
                {"role": "user", "content": q},
            ]
            base_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Ensure separator when appending history to avoid awkward concatenation
            # Check if history is not empty and doesn't start with newline
            full_text = base_prompt + hist
            formatted.append(full_text)
        return formatted

    def generate_batch_steps(self, full_prompts: List[str]) -> List[str]:
        # T=0.8 for diversity, strictly stop at STOP_STRING
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
            # CRITICAL FIX: Split at stop token to discard hallucinated continuations
            # .replace() is not enough if the model keeps babbling after the stop token
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
            # Judge prompt asking for a score on the *entire* flow so far
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
                {"role": "system", "content": "You are a strict math grader. Output only numerical scores."},
                {"role": "user", "content": judge_prompt}
            ]
            txt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(txt)

        # Greedy for deterministic scoring
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
        print(f"\n--- Starting Search: {raw_question[:40]}... ---")

        # 1. Initialize Wavefront (List of active SearchCandidates)
        wavefront = [SearchCandidate(history_text="", cumulative_score=0.0, depth=0)]
        finished_candidates = []

        # 2. Main Evolution Loop
        for depth in range(MAX_STEPS):
            if not wavefront:
                print("  -> All branches pruned. Stopping.")
                break

            print(f"  [Depth {depth}] Processing {len(wavefront)} active branches...")

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

                # Check for Immediate Finish
                if FINAL_ANSWER_PREFIX in step_text:
                    cand.history += "\n" + step_text
                    cand.cum_score += 1.0 # Boost finished score
                    cand.depth += 1
                    cand.is_finished = True
                    finished_candidates.append(cand)
                    print(f"    -> Branch {cand.id} finished.")
                    continue

                if not step_text: # Handle empty generation edge case
                    continue

                # Prepare for scoring
                full_text_to_score = cand.history + "\n" + step_text
                prm_inputs.append((raw_question, full_text_to_score))
                candidates_for_prm.append((cand, step_text))

            if not candidates_for_prm:
                if not finished_candidates:
                    print("  -> No valid steps generated and no finished candidates.")
                    break
                if not candidates_for_prm: break

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

            # 1. Process Passers (Candidates that passed threshold)
            for cand, step_text, score in passers:
                cand.history += "\n" + step_text
                cand.cum_score += score
                cand.depth += 1
                next_wavefront.append(cand)
                if VERBOSE: print(f"    -> Keep (Score {score:.2f})")

            # 2. Process Failers (Expansion / Mitosis)
            # We calculate how much space is left in the next wavefront
            # NOTE: We do not subtract len(next_wavefront) yet, we allow temporary overflow
            # and sort/truncate later to find the global best mix.

            if failers:
                # Sort failers by score (descending) to prioritize better "misses"
                failers.sort(key=lambda x: x[2], reverse=True)

                for cand, step_text, score in failers:
                    # If the wavefront is already huge, maybe stop expanding?
                    # But let's allow expansion and then truncate to be safe.
                    if len(next_wavefront) >= MAX_WIDTH * 2:
                        break

                    if VERBOSE: print(f"    -> Reject (Score {score:.2f}). Spawning {EXPANSION_FACTOR} clones.")

                    for _ in range(EXPANSION_FACTOR):
                        clone = cand.copy()
                        # Note: We do NOT append the bad step.
                        # It will run generation again on the *same history* next loop.
                        next_wavefront.append(clone)

            # 3. Final Truncation (Survival of the Fittest)
            # This is the CRITICAL FIX: Sort ALL candidates (passers + expanded retries)
            # by their quality metric before killing them off.
            if next_wavefront:
                next_wavefront.sort(key=lambda c: c.avg_score, reverse=True)

                if len(next_wavefront) > MAX_WIDTH:
                    if VERBOSE: print(f"    -> Truncating population from {len(next_wavefront)} to {MAX_WIDTH}")
                    next_wavefront = next_wavefront[:MAX_WIDTH]

            wavefront = next_wavefront

        # 3. Final Selection
        all_results = finished_candidates
        if not all_results and wavefront:
            print("  Warning: No branches output <FINAL>. Using active branches.")
            all_results = wavefront

        if not all_results:
            return ""

        # Normalize scores by depth to be fair
        best_cand = max(all_results, key=lambda c: c.avg_score)

        print(f"--- Search Done. Best Candidate Score: {best_cand.avg_score:.2f} ---")
        return best_cand.history

# ==========================================
# Main & Helpers
# ==========================================

def extract_math_answer(text: str) -> Optional[float]:
    # 1. Boxed LaTeX
    boxed = re.search(r"\\boxed{([^}]+)}", text)
    if boxed:
        clean = boxed.group(1).replace('$', '').replace('\\', '').replace(',', '').strip()
        try:
            nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", clean)
            if nums: return float(nums[-1])
        except: pass

    # 2. Final Prefix
    tag = re.search(rf"{FINAL_ANSWER_PREFIX}\s*([^ \n]+)", text)
    if tag:
        try: return float(tag.group(1).replace(',', '').strip())
        except: pass

    # 3. Last number
    try:
        all_nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text.replace(',', ''))
        if all_nums: return float(all_nums[-1])
    except: pass
    return None

def main():
    print(f"Initializing vLLM Generator: {MODEL_NAME}")

    # Adjust GPU util based on PRM choice
    gpu_util = 0.9 if PRM_MODE == "A" else 0.6

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_util,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=4096,
        enforce_eager=True
    )

    generator = VLLMGeneratorWrapper(llm)
    prm = None

    if PRM_MODE == "A":
        print("Configuration: [A] Self-Correction Prompting")
        prm = VLLMCoTPRMWrapper(llm)
    elif PRM_MODE == "B":
        print("Configuration: [B] External Logits PRM")
        try:
            from tt_scale.prm.logits_prm import LogitsPRM
            prm = LogitsPRM(model_name="RLHFlow/Llama3.1-8B-PRM-Deepseek-Data", device="cuda")
        except ImportError:
            print("Could not import LogitsPRM. Ensure file exists.")
            return

    searcher = ParallelHybridSearcher(generator, prm)

    print(f"Loading MATH-500 (N={NUM_SAMPLES})")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    # Deterministic sample for reproducibility
    random.seed(42)
    indices = random.sample(range(len(dataset)), min(NUM_SAMPLES, len(dataset)))
    samples = [dataset[i] for i in indices]

    correct_count = 0

    for i, sample in enumerate(samples):
        q = sample.get('problem', sample.get('question'))
        truth_str = sample.get('solution', sample.get('answer'))
        truth = extract_math_answer(truth_str)

        print(f"\n{'='*15} SAMPLE {i+1}/{len(samples)} {'='*15}")

        final_history = searcher.run(q)
        prediction = extract_math_answer(final_history)

        is_correct = False
        if prediction is not None and truth is not None:
            is_correct = abs(prediction - truth) < 1e-3
            if is_correct: correct_count += 1

        status = "✅ CORRECT" if is_correct else f"❌ WRONG (Exp: {truth})"
        print(f"Result: {status} | Pred: {prediction}")

    print(f"\n=== Final Accuracy: {correct_count}/{len(samples)} ({correct_count/len(samples):.2%}) ===")

if __name__ == "__main__":
    main()
