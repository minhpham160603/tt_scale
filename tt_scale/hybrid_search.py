import random
from datasets import load_dataset

from .prm.custom_prm import CustomPRM
from .generator.custom_generator import Generator
from .generator.backtrack_generator import BacktrackGenerator
from .prm.logits_prm import LogitPRM
from transformers import BitsAndBytesConfig
import torch
from .base_classes import AbstractGenerator, AbstractPRM

# Configuration
K_TOKENS = 64          # How many tokens to generate per "step"
TAU = 0.5              # Score threshold to keep a step (0.0 to 1.0)
MAX_BACKTRACKS = 3     # How many times to retry a step if score is low
NUM_SAMPLES = 3        # Number of GSM8K samples to test

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

class HybridSearcher:
    def __init__(self, generator: AbstractGenerator, prm: AbstractPRM, max_output_len=512):
        # Initialize our two custom classes
        self.generator = generator
        self.prm = prm
        self.max_output_len = max_output_len

    def run(self, prompt):
        print(f"\n--- New Run: {prompt} ---\n")
        
        # Initial State
        initial_ids = self.generator.tokenize(prompt)
        prompt_len = initial_ids.shape[1]
        checkpoints = [(initial_ids, None, 0)]
        finished = False
        response_part = ""
        while not finished and checkpoints:
            current_ids, current_cache, bt_count = checkpoints[-1]
            if current_ids.shape[1] > self.max_output_len:
                finished = True
                break
            updated_full_ids, new_cache, finished = self.generator.generate_step(
                current_ids, 
                current_cache, 
                K_TOKENS
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

def test_gsm8k(searcher: HybridSearcher):
    print("--- Loading GSM8K Dataset ---")
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    # Select random samples
    indices = random.sample(range(len(dataset)), NUM_SAMPLES)
    samples = [dataset[i] for i in indices]

    for i, sample in enumerate(samples):
        question = sample['question']
        ground_truth = sample['answer']

        print(f"\n\n{'='*20} SAMPLE {i+1}/{NUM_SAMPLES} {'='*20}")
        print(f"QUESTION: {question}")

        formatted_prompt = f"Question: {question}\nAnswer:"
        
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
    test_gsm8k(searcher)