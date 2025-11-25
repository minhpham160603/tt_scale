import random
from datasets import load_dataset

from .prm.custom_prm import CustomPRM
from .generator.custom_generator import Generator
from .generator.backtrack_generator import BacktrackGenerator
from .prm.logits_prm import LogitPRM
from transformers import BitsAndBytesConfig
import torch
from .base_classes import AbstractGenerator, AbstractPRM

# ==========================================
# Configuration
# ==========================================

# Sequential hybrid search (original)
K_TOKENS = 64          # How many tokens to generate per "step"
TAU = 0.5              # Score threshold to keep a step (0.0 to 1.0)
MAX_BACKTRACKS = 3     # How many times to retry a step if score is low

NUM_SAMPLES = 3        # Number of GSM8K samples to test

# Parallel hybrid search (mirrors BacktrackAlgoParallel.ipynb)
PAR_K_TOKENS = 20
PAR_TAU = 0.4
MAX_RETRIES = 5            # Per-branch retries before pruning
M_EXPANSION = 3            # How many clones to spawn when a branch fails
MAX_TOTAL_BRANCHES = 8     # Hard cap on number of live branches
MAX_STEPS = 30             # Max generation "steps" per run

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)


class HybridSearcher:
    def __init__(self, generator: AbstractGenerator, prm: AbstractPRM, max_output_len: int = 512):
        # Initialize our two custom classes
        self.generator = generator
        self.prm = prm
        self.max_output_len = max_output_len

    # --------------------------------------------------
    #  Sequential backtracking hybrid search (existing)
    # --------------------------------------------------
    def run(self, prompt: str) -> str:
        """
        Original *sequential* hybrid search:
        - Expand a single trajectory step-by-step.
        - Use the PRM score to decide whether to keep or backtrack.
        """
        print(f"\n--- New Run (sequential): {prompt} ---\n")
        
        # Initial State
        initial_ids = self.generator.tokenize(prompt)
        prompt_len = initial_ids.shape[1]
        checkpoints = [(initial_ids, None, 0)]  # (ids, cache, num_backtracks)
        finished = False
        response_part = ""

        while not finished and checkpoints:
            current_ids, current_cache, bt_count = checkpoints[-1]

            # Safety: stop if we exceed a maximum length
            if current_ids.shape[1] > self.max_output_len:
                print("  -> Reached max_output_len, stopping.")
                finished = True
                break

            updated_full_ids, new_cache, finished = self.generator.generate_step(
                current_ids, 
                current_cache, 
                K_TOKENS,
            )

            response_part = self.generator.decode(updated_full_ids[0][prompt_len:])
            score = self.prm.get_score(prompt, response_part)

            if score > TAU:
                print(f"  -> KEEP (Score {score:.3f})")
                checkpoints.append((updated_full_ids, new_cache, 0))
            else:
                print(f"  -> BACKTRACK (Score {score:.3f})")
                if bt_count < MAX_BACKTRACKS:
                    _ids, _cache, _ = checkpoints.pop()
                    checkpoints.append((_ids, _cache, bt_count + 1))
                    print(f"  -> Retrying step... (Attempt {bt_count + 1}/{MAX_BACKTRACKS})")
                else:
                    print("  -> MAX RETRIES. Forced accept.")
                    checkpoints.append((updated_full_ids, new_cache, 0))

        return response_part

    # --------------------------------------------------
    #  Parallel hybrid search (branching version)
    # --------------------------------------------------
    def run_parallel(
        self,
        prompt: str,
        k_tokens: int = PAR_K_TOKENS,
        tau: float = PAR_TAU,
        max_retries: int = MAX_RETRIES,
        m_expansion: int = M_EXPANSION,
        max_total_branches: int = MAX_TOTAL_BRANCHES,
        max_steps: int = MAX_STEPS,
    ) -> str:
        """
        Parallel hybrid search similar to `BacktrackAlgoParallel.ipynb`.

        Multiple branches (partial solutions) are explored in parallel:
        - At each step we expand every active branch by `k_tokens`.
        - We score all partial answers in *batch* via the PRM.
        - Good branches continue; bad ones are either split (cloned) or pruned.
        """
        print(f"\n--- New Run (parallel): {prompt} ---\n")

        initial_ids = self.generator.tokenize(prompt)
        prompt_len = initial_ids.shape[1]

        # Each branch keeps its own ids/cache and retry counter
        active_branches = [
            {
                "ids": initial_ids,
                "cache": None,
                "retries": 0,
                "history": [],
            }
        ]
        finished_branches = []
        step_count = 0

        while active_branches and step_count < max_steps:
            step_count += 1
            print(f"\n=== Step {step_count} | Active branches: {len(active_branches)} ===")

            # 1) Expand all active branches (using generator batch helper)
            ids_batch = [b["ids"] for b in active_branches]
            cache_batch = [b["cache"] for b in active_branches]

            step_results = self.generator.generate_step_batch(
                input_ids_batch=ids_batch,
                past_key_values_batch=cache_batch,
                max_new_tokens=k_tokens,
            )

            # Collect partial answers (text) for PRM scoring
            partial_texts = []
            full_ids_list = []
            new_cache_list = []
            finished_flags = []

            for (full_seq, new_cache, finished_flag), branch in zip(step_results, active_branches):
                full_ids_list.append(full_seq)
                new_cache_list.append(new_cache)
                finished_flags.append(finished_flag)

                # We always decode from the *original prompt* onward
                partial_text = self.generator.decode(full_seq[0][prompt_len:])
                partial_texts.append(partial_text)

            # 2) Score all partial answers in batch
            qa_pairs = [(prompt, ans) for ans in partial_texts]
            scores = self.prm.get_scores_batch(qa_pairs)

            next_active_branches = []

            for i, (branch, score, full_ids, new_cache, is_finished) in enumerate(
                zip(active_branches, scores, full_ids_list, new_cache_list, finished_flags)
            ):
                print(f"Branch {i}: score={score:.3f}, retries={branch['retries']}")

                # Safety: if the sequence is already too long, treat as finished
                over_max_len = full_ids.shape[1] > self.max_output_len

                if score > tau:
                    print(f"  Br {i}: ✅ Pass ({score:.3f})")
                    if is_finished or over_max_len:
                        final_text = self.generator.decode(full_ids[0][prompt_len:])
                        finished_branches.append({"text": final_text, "score": score})
                    else:
                        next_active_branches.append(
                            {
                                "ids": full_ids,
                                "cache": new_cache,
                                "retries": 0,
                                "history": branch["history"] + [score],
                            }
                        )
                else:
                    print(f"  Br {i}: ❌ Fail ({score:.3f})")
                    if branch["retries"] < max_retries:
                        # Clone the *previous* state of the branch (before this step)
                        num_clones = 1
                        if len(next_active_branches) + len(active_branches) < max_total_branches:
                            num_clones = m_expansion

                        print(
                            f"  -> Split into {num_clones} clones "
                            f"(retry {branch['retries'] + 1}/{max_retries})"
                        )

                        for _ in range(num_clones):
                            next_active_branches.append(
                                {
                                    "ids": branch["ids"],
                                    "cache": branch["cache"],
                                    "retries": branch["retries"] + 1,
                                    "history": branch["history"],
                                }
                            )
                    else:
                        print(f"  Br {i}: 💀 Pruned (max retries reached)")

            # Respect global cap on the number of branches
            if len(next_active_branches) > max_total_branches:
                next_active_branches = next_active_branches[:max_total_branches]

            active_branches = next_active_branches

        if not finished_branches:
            return "Failed."

        # Pick the best finished branch by its final PRM score
        best = max(finished_branches, key=lambda x: x["score"])
        print(f"=> Winner: {best['score']:.3f}")
        return best["text"]


def test_gsm8k(searcher: HybridSearcher, use_parallel: bool = False):
    print("--- Loading GSM8K Dataset ---")
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    # Select random samples
    indices = random.sample(range(len(dataset)), NUM_SAMPLES)
    samples = [dataset[i] for i in indices]

    for i, sample in enumerate(samples):
        question = sample["question"]
        ground_truth = sample["answer"]

        print(f"\n\n{'='*20} SAMPLE {i+1}/{NUM_SAMPLES} {'='*20}")
        print(f"QUESTION: {question}")

        formatted_prompt = f"Question: {question}\nAnswer:"

        if use_parallel:
            final_answer = searcher.run_parallel(formatted_prompt)
        else:
            final_answer = searcher.run(formatted_prompt)

        print(f"\n--- RESULT ---")
        print(f"GENERATED: {final_answer.strip()}")
        print(f"TRUTH:     {ground_truth}")


if __name__ == "__main__":
    generator = BacktrackGenerator(
        model_name="Qwen/Qwen3-4B", 
    )
    
    # prm = LogitPRM(
    #     model_name="RLHFlow/Llama3.1-8B-PRM-Deepseek-Data", 
    # )

    prm = CustomPRM(
        model_name="Qwen/Qwen3-4B",
    )

    searcher = HybridSearcher(generator, prm)

    # Sequential:
    # test_gsm8k(searcher, use_parallel=False)

    # Parallel:
    test_gsm8k(searcher, use_parallel=True)
