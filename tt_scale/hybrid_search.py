import random
from datasets import load_dataset
import torch
import math
import random
from datasets import load_dataset
import re

from .prm.custom_prm import CustomPRM
from .generator.custom_generator import Generator
from .generator.backtrack_generator import BacktrackGenerator, STOP_STRING
from .base_classes import AbstractGenerator, AbstractPRM

# ==========================================
# Configuration
# ==========================================

# Sequential hybrid search
K_TOKENS = 256          
TAU = 0.5              
MAX_BACKTRACKS = 3     

NUM_SAMPLES = 5    
SEED = 45

# Parallel hybrid search
PAR_K_TOKENS = 20
PAR_TAU = 0.4
MAX_RETRIES = 3            
M_EXPANSION = 3            
MAX_TOTAL_BRANCHES = 6  
KEEPING_BRANCHES = 2
MAX_STEPS = 10        
EPSILON = 1e-3     




class HybridSearcher:
    def __init__(self, generator: AbstractGenerator, prm: AbstractPRM, max_output_len: int = 1024):
        self.generator = generator
        self.prm = prm
        self.max_output_len = max_output_len
        if hasattr(generator, 'tokenizer'):
            self.tokenizer = generator.tokenizer
        else:
            raise AttributeError("Generator must have a tokenizer.")

    # --------------------------------------------------
    #  Sequential backtracking
    # --------------------------------------------------
    def run(self, prompt: str) -> str:
        print(f"\n--- New Run (sequential): {prompt[:50]}... ---\n")
        initial_ids = self.generator.tokenize(prompt)
        prompt_len = initial_ids.shape[1]
        checkpoints = [(initial_ids, None, 0)]  
        finished = False
        response_part = ""

        while not finished and checkpoints:
            current_ids, current_cache, bt_count = checkpoints[-1]

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
                else:
                    print("  -> MAX RETRIES. Forced accept.")
                    checkpoints.append((updated_full_ids, new_cache, 0))

        return response_part

    # --------------------------------------------------
    #  Parallel hybrid search
    # --------------------------------------------------
    def run_parallel(self, prompt: str) -> str:
        print(f"\n--- Parallel Run: {prompt[:50]}... ---")
        
        initial_ids = self.generator.tokenize(prompt)
        
        active_branches = [
            {
                "ids": initial_ids, 
                "cache": None,
                "retries": 0,
                "cumulative_score": 1.0,
                "history": [1.0],
                "text": ""
            }
        ]

        finished_branches = []
        step_count = 0
        retries = 0
        jumping = False

        while active_branches:
            
            step_count += 1
            if step_count > MAX_STEPS:
                print("Max steps reached.")
                break
            cloned_branches = []
            clone_times = max(1, math.floor(MAX_TOTAL_BRANCHES/len(active_branches)))
            for branch in active_branches:
                for _ in range(clone_times):
                    cloned_branches.append({
                        "ids": branch["ids"],
                        "cache": branch["cache"],
                        "retries": branch["retries"],
                        "cumulative_score": branch["cumulative_score"],
                        "history": branch["history"][:],
                        "text": branch["text"]
                    })
            active_branches = cloned_branches
            input_tensors = [b["ids"] for b in active_branches]
            caches = [b["cache"] for b in active_branches]
            
            print(f"--- Step {step_count} | Generating for {len(active_branches)} branches... ---")

            try:
                # Calls the updated generate_batch_step with fixed mask dtype
                generated_seqs, new_caches = self.generator.generate_batch_step(
                    input_tensors, 
                    caches, 
                    PAR_K_TOKENS, 
                    temperature=0.7 + 0.2 * retries,
                    # num_return_sequences= max(1, math.floor(MAX_TOTAL_BRANCHES/len(active_branches)))
                )
                # print(f"  -> Generated {len(generated_seqs)} sequences.")
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print("OOM Error! Clearing cache and reducing branches.")
                    torch.cuda.empty_cache()
                    active_branches = active_branches[:1] # Drastic fallback
                    continue
                raise e

            # Prepare for Scoring
            prompts_for_judge = []
            responses_for_judge = []
            generated_texts = [] 

            for i, seq in enumerate(generated_seqs):
                full_text = self.generator.decode(seq[0])
                generated_texts.append(full_text)
                
                prompts_for_judge.append(prompt)
                
                # Extract partial response logic
                decoded_prompt = self.generator.decode(initial_ids[0])
                if full_text.startswith(decoded_prompt):
                    r = full_text[len(decoded_prompt):]
                else:
                    r = full_text 
                
                responses_for_judge.append(r)

            # Scoring
            torch.cuda.empty_cache() # Help prevent OOM during scoring
            scores = self.prm.get_scores_batch(list(zip(prompts_for_judge, responses_for_judge)))

            next_active_branches = []
            retry_branches = []
            passing_branches = []
            passing = 0
            if jumping:
                passing_threshold = 0.0
                jumping = False
            else:
                passing_threshold = PAR_TAU

            for i, branch in enumerate(active_branches):
                score = scores[i]
                out_seq = generated_seqs[i]
                full_text = generated_texts[i]
                
                prev_len = branch["ids"].shape[1]
                new_tokens = out_seq[0, prev_len:]
                
                is_eos = self.tokenizer.eos_token_id in new_tokens
                new_text = self.generator.decode(new_tokens)

                if score > passing_threshold:
                    print(f"  Br {i}: ✅ Pass ({score:.2f})")
                    passing += 1
                    
                    if is_eos or "###" in new_text or STOP_STRING in new_text: 
                        finished_branches.append({
                            "text": full_text, 
                            "score": score
                        })
                    else:
                        # Remove pads
                        non_pad = (out_seq[0] != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                        if len(non_pad) > 0:
                            clean_ids = out_seq[0][non_pad[0]:].unsqueeze(0)
                        else:
                            clean_ids = out_seq
                        
                        passing_branches.append({
                            "ids": clean_ids,
                            "cache": new_caches[i],
                            "retries": 0,
                            "cumulative_score": score,
                            "history": branch["history"] + [score],
                            "text": full_text
                        })
                else:
                    print(f"  Br {i}: ❌ Fail ({score:.2f})")
                    retry_branches.append(branch)
                    

            # if enough passing branches, not backtracking, and prune to KEEPING_BRANCHES
            if passing >= KEEPING_BRANCHES:
                print(f"  -> Sufficient passing branches ({passing}), pruning to {KEEPING_BRANCHES}.")
                passing_branches.sort(key=lambda x: x["cumulative_score"], reverse=True)
                active_branches = passing_branches[:KEEPING_BRANCHES]
                retries = 0  


            else:
                print(f"  -> Insufficient passing branches ({passing}), backtracking.")

                backtrack_candidates = passing_branches + retry_branches[:KEEPING_BRANCHES - passing]
                
                active_branches = []
                for b in backtrack_candidates:
                    active_branches.append({
                        "ids": b["ids"],
                        "cache": b["cache"],
                        "retries": b["retries"] + 1,
                        "cumulative_score": b["cumulative_score"],
                        "history": b["history"],
                        "text": b["text"]
                    })
                retries += 1
                if retries > MAX_RETRIES:
                    print("  -> Max retries reached, forcing jumping")
                    jumping = True

                    

            
            if not active_branches and not finished_branches:
                return "Failed to generate a solution."

        if not finished_branches:
            if active_branches:
                 best = max(active_branches, key=lambda x: x["cumulative_score"])
                 return best["text"]
            return "Failed."

        best = max(finished_branches, key=lambda x: x["score"])
        print(f"=> Winner: {best['score']:.3f}")
        return best["text"]
    


# def extract_answer_number(text):
#     """Extracts the number after ### or the last number in text."""
#     if not text: return None
#     match = re.search(r'###\s*(-?[\d,]+(?:\.\d+)?)', text)
#     if match:
#         try: return float(match.group(1).replace(',', ''))
#         except: pass
#     numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
#     if numbers:
#         try: return float(numbers[-1].replace(',', ''))
#         except: pass
#     return None

# def extract_result(text):
#     # Regex pattern explanation:
#     # r"..." : Raw string literal
#     # <FINAL>\s* : Matches the literal '<FINAL>' followed by zero or more whitespace characters
#     # ([\d\.]+) : Capturing Group 1. Matches one or more digits (\d) or decimal points (\.).
#     # $ : Asserts position at the end of the string (optional, but good for cleanup)
    
#     # We use \s* to handle potential extra spaces/newlines between <FINAL> and the score.
#     pattern = r"<FINAL>\s*([\d\.]+)"
#     match = re.search(pattern, text, re.MULTILINE)
#     if match:
#         try:
#             score_str = match.group(1)
#             return float(score_str)
#         except ValueError:
#             print(f"Warning: Could not convert '{score_str}' to float.")
#             return None
#     return None

# def test_gsm8k(searcher: HybridSearcher, use_parallel: bool = False):
#     print("--- Loading GSM8K Dataset ---")
#     dataset = load_dataset("openai/gsm8k", "main", split="test")
#     random.seed(42)
#     indices = random.sample(range(len(dataset)), NUM_SAMPLES)
    
#     correct = 0
#     for i, idx in enumerate(indices):
#         sample = dataset[idx]
#         question = sample["question"]
#         truth_str = sample["answer"]
        
#         print(f"\n{'='*15} SAMPLE {i+1}/{NUM_SAMPLES} {'='*15}")
#         print(f"Q: {question}")

#         # Apply Chat Template with System Prompt
#         messages = [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": question}
#         ]
#         formatted_prompt = searcher.tokenizer.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )

#         if use_parallel:
#             output = searcher.run_parallel(formatted_prompt)
#         else:
#             output = searcher.run(formatted_prompt)

#         # Evaluation
#         pred = extract_answer_number(output)
#         truth = extract_answer_number(truth_str.split("####")[-1]) if "####" in truth_str else None
        
#         is_correct = (pred is not None and truth is not None and abs(pred - truth) < 1e-6)
#         if is_correct: correct += 1
#         print(f"\n[Generated]: {output.strip()}")
#         print(f"\n[Truth]:     {sample['answer']}")
#         print(f"\nPred: {pred} | Truth: {truth} | {'✅' if is_correct else '❌'}")

#     print(f"\nAccuracy: {correct}/{NUM_SAMPLES} ({correct/NUM_SAMPLES*100:.1f}%)")

# def test_gsm8k(searcher: HybridSearcher, use_parallel: bool = False):
#     print("--- Loading GSM8K Dataset ---")
#     dataset = load_dataset("openai/gsm8k", "main", split="test")

#     indices = random.sample(range(len(dataset)), NUM_SAMPLES)
#     samples = [dataset[i] for i in indices]

#     for i, sample in enumerate(samples):
#         question = sample["question"]
#         ground_truth = sample["answer"]

#         print(f"\n\n{'='*20} SAMPLE {i+1}/{NUM_SAMPLES} {'='*20}")
#         print(f"QUESTION: {question}")

#         formatted_prompt = f"Question: {question}\nAnswer:"

#         if use_parallel:
#             final_answer = searcher.run_parallel(formatted_prompt)
#         else:
#             final_answer = searcher.run(formatted_prompt)

#         print(f"\n--- RESULT ---")
#         print(f"GENERATED: {final_answer.strip()}")
#         print(f"TRUTH:     {ground_truth}")

def extract_result(text):
    # Regex pattern explanation:
    # r"..." : Raw string literal
    # <FINAL>\s* : Matches the literal '<FINAL>' followed by zero or more whitespace characters
    # ([\d\.]+) : Capturing Group 1. Matches one or more digits (\d) or decimal points (\.).
    # $ : Asserts position at the end of the string (optional, but good for cleanup)
    
    # We use \s* to handle potential extra spaces/newlines between <FINAL> and the score.
    pattern = r"<FINAL>\s*([\d\.]+)"
    match = re.search(pattern, text, re.MULTILINE)
    if match:
        try:
            score_str = match.group(1)
            return float(score_str)
        except ValueError:
            print(f"Warning: Could not convert '{score_str}' to float.")
            return None
    return None


def test_gsm8k(searcher, use_parallel: bool = True, VERBOSE: bool = True):
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    random.seed(SEED)
    indices = random.sample(range(len(dataset)), NUM_SAMPLES)
    correct = 0
    for i in indices:
        sample = dataset[i]
        print(f"\n\n=== Question: {sample['question']} ===")
        if use_parallel:
            output_text = searcher.run_parallel(sample['question'])
        else:
            output_text = searcher.run(sample['question'])
        answer = extract_result(output_text)

        idx = sample['answer'].find("####")
        if idx == -1:
            print("Warning: Could not find '####' in ground truth answer.", sample['answer'])
            continue
        # if answer is None:
        #     print("Warning: Could not extract answer from generated output.", output_text)
        #     continue
        truth_answer = float(sample['answer'][idx+4:].strip())
        correct += abs(answer - truth_answer) < EPSILON
        if VERBOSE:
            print(f"\n[Generated]: {output_text.strip()}")
            print(f"[Generated Answer]: {answer} | [Truth Answer]: {truth_answer}")
        else:
            print(f"[Generated Answer]: {answer} | [Truth Answer]: {truth_answer}")

    print(f"\n\n=== GSM8K Results: {correct}/{NUM_SAMPLES} correct ===")


if __name__ == "__main__":
    generator = BacktrackGenerator(
        model_name="Qwen/Qwen3-0.6B", 
    )
    
    prm = CustomPRM(
        model_name="Qwen/Qwen3-0.6B",
    )

    searcher = HybridSearcher(generator, prm)
    test_gsm8k(searcher, use_parallel=True)