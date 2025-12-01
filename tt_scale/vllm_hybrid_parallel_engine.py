import random
import torch
import re
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tt_scale.prm.logits_prm import LogitsPRM
from tt_scale.prm.vllm_cot_prm import VLLMCoTPRM
import math


# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-0.6B" # "Qwen/Qwen3-4B"
K_TOKENS = 100          
TAU = 0.6              
MAX_BACKTRACKS = 3     
NUM_SAMPLES = 50        
FINAL_ANSWER_PREFIX = "<FINAL>"
DEBUG = False
VERBOSE = True 
EPSILON = 1e-3
SEED = 69      


# Parallel hybrid search
PAR_K_TOKENS = 20
PAR_TAU = 0.4
MAX_RETRIES = 3            
M_EXPANSION = 3            
MAX_TOTAL_BRANCHES = 6  
KEEPING_BRANCHES = 2
MAX_STEPS = 10        
EPSILON = 1e-3     
# ==========================================
# 1. VLLM Generator Wrapper
# ==========================================
class VLLMGenerator:
    STOP_STRING = "<END_STEP>"
    SYS_PROMPT = f"""You are a genius problem solver. 
Solve the problem step-by-step to avoid mistakes.
After **EVERY logical step** of reasoning, output the token {STOP_STRING}.
If all steps are completed, return final answer with `{FINAL_ANSWER_PREFIX}` prefix (for example: `{FINAL_ANSWER_PREFIX} 16` or `{FINAL_ANSWER_PREFIX} 90.6`)"""

    def __init__(self, llm_engine):
        self.llm = llm_engine
        self.tokenizer = self.llm.get_tokenizer()
        self.stop_token = self.STOP_STRING

    def build_input_context(self, question, partial_answer=""):
        """
        Constructs the full input string for the LLM.
        It separates the 'fixed' prompt history from the 'dynamic' partial answer.
        """
        # 1. Format the Conversation History (System + User)
        messages = [
            {"role": "system", "content": self.SYS_PROMPT},
            {"role": "user", "content": question},
        ]
        
        # 2. Apply Template (add_generation_prompt=True adds the Assistant header)
        # e.g., "...<|im_start|>user\n{Question}<|im_end|>\n<|im_start|>assistant\n"
        context_prefix = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False
        )

        context_prefix += partial_answer

        # if DEBUG:
        #     print(">>>>>>>>>>>>>>>>>>>>>")
        #     print("INPUT_CONTEXT:", context_prefix)
        #     print("<<<<<<<<<<<<<<<<<<<<<")
        
        # 3. Append the partial answer so far (Prefilling)
        # vLLM will see this as the assistant having already typed this much
        
        return context_prefix

    def generate_step(self, full_context, retry_attempt=0):
        """
        Generates the next step given the full context.
        """
        temp = 1 + (0.5 * retry_attempt)
        
        params = SamplingParams(
            temperature=temp,
            max_tokens=K_TOKENS,
            top_p=0.9,
            top_k=40, 
        )

        # vLLM automatically uses Prefix Caching here.
        # Since 'context_prefix' (System+User) is constant, it is cached.
        # Since 'partial_answer' grows, vLLM caches the shared prefix of the answer.
        outputs = self.llm.generate([full_context], params, use_tqdm=False)
        new_text = outputs[0].outputs[0].text
        finish_reason = outputs[0].outputs[0].finish_reason
        is_eos = (finish_reason == "stop" and self.stop_token not in new_text)

        if DEBUG:
            print(">>>>>>>>>>>>>>>>>>>>")
            print("GEN_STEP: ", new_text)
            print("<<<<<<<<<<<<<<<<<<<<")
            print("Finish Reason:", finish_reason, "| Is EOS:", is_eos)
        
        return new_text, is_eos

# ==========================================
# 2. VLLM PRM Wrapper
# ==========================================
class VLLMPRM:
    def __init__(self, llm_engine):
        self.llm = llm_engine
        self.tokenizer = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=0.0, max_tokens=10, stop=["\n"])

    def get_score(self, question, current_partial_answer):
        # Create a fresh Judge Prompt
        judge_prompt = f"""
Review the following partial solution to a math problem.
---
Question: {question}
Partial Answer So Far: {current_partial_answer}
---
Rate the logical correctness of the LAST step in the Partial Answer on a scale of 1 to 10.
If the logic is sound, give a high score. If there is an error, give a low score.
Output ONLY the number.
"""
        messages = [
            {"role": "system", "content": "You are a strict math grader. Output only numerical scores."},
            {"role": "user", "content": judge_prompt}
        ]
        
        full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        outputs = self.llm.generate([full_prompt], self.params, use_tqdm=False)
        
        # Parse score
        match = re.search(r"(\d+(\.\d+)?)", outputs[0].outputs[0].text)
        if match:
            try:
                return max(1.0, min(10.0, float(match.group(1)))) / 10.0
            except:
                pass
        return 0.0

# ==========================================
# 3. Hybrid Searcher
# ==========================================
class HybridSearcher:
    def __init__(self, generator, prm, max_len=512):
        self.gen = generator
        self.prm = prm
        self.max_len = max_len

    def run(self, raw_prompt):
        print(f"\n--- New Run: {raw_prompt[:40]}... ---")
        
        # State: (current_generated_text, backtrack_count)
        checkpoints = [("", 0)]
        
        final_response = ""
        finished = False

        while not finished and checkpoints:
            current_generated, bt_count = checkpoints[-1]
            
            # 1. Ask Generator to build the full context string
            # This cleanly separates prompt formatting logic from search logic
            full_gen_context = self.gen.build_input_context(raw_prompt, current_generated)

            if len(full_gen_context) > self.max_len * 4: 
                break

            # 2. Generate Step
            new_chunk, is_eos = self.gen.generate_step(full_gen_context, retry_attempt=bt_count)
            
            if not new_chunk:
                if is_eos: finished = True
                checkpoints.pop()
                continue
                
            # 3. Score
            full_answer_candidate = current_generated + new_chunk
            score = self.prm.get_score(raw_prompt, full_answer_candidate)

            if score >= TAU or bt_count >= MAX_BACKTRACKS:
                if score >= TAU:
                    print(f"  -> KEEP (Score {score:.2f})")
                else:
                    print(f"  -> FORCE KEEP (Score {score:.2f})")
                checkpoints.append((full_answer_candidate, 0))
                if is_eos or (FINAL_ANSWER_PREFIX in new_chunk):
                    finished = True
            else:
                print(f"  -> REJECT (Score {score:.2f})")
                checkpoints.pop()
                checkpoints.append((current_generated, bt_count + 1))
                print(f"     Retrying... ({bt_count + 1}/{MAX_BACKTRACKS})")
            final_response = checkpoints[-1][0]
        return final_response

    def run_parallel(self, prompt: str) -> str:
        print(f"\n--- Parallel Run: {prompt[:50]}... ---")
        
        
        active_branches = [
            {
                "cumulative_score": 1.0,
                "checkpoint": [],
                "finished": False,
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
                        "cumulative_score": branch["cumulative_score"],
                        "checkpoint": branch["checkpoint"].copy(),
                        "finished": branch["finished"],
                    })
            active_branches = cloned_branches
            
            print(f"--- Step {step_count} | Generating for {len(active_branches)} branches... ---")
            
            next_active_branches = []
            retry_branches = []
            passing_branches = []
            passing = 0
            if jumping:
                passing_threshold = 0.0
                jumping = False
            else:
                passing_threshold = PAR_TAU


            for i,branch in enumerate(active_branches):
                try:
                    if branch["finished"]:
                        continue
                    # Calls the updated generate_batch_step with fixed mask dtype
                    current_generated = branch["checkpoint"][-1] if branch["checkpoint"] else ""
                    full_gen_context = self.gen.build_input_context(prompt, current_generated)
                    
                    new_chunk, is_eos = self.gen.generate_step(full_gen_context, retry_attempt=retries)
                    
                    # if not new_chunk:
                    #     if is_eos: 
                    #         branch[finished] = True
                    #     checkpoints.pop()
                    #     continue
                    # print(f"  -> Generated {len(generated_seqs)} sequences.")
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print("OOM Error! Clearing cache and reducing branches.")
                        torch.cuda.empty_cache()
                        active_branches = active_branches[:1] # Drastic fallback
                        continue
                    raise e

                full_answer_candidate = current_generated + new_chunk
                score = self.prm.get_score(prompt, full_answer_candidate)
                branch["cumulative_score"] = score

                if branch["cumulative_score"] > passing_threshold:
                    print(f"  Br {i}: ✅ Pass ({score:.2f})")
                    passing += 1
                    
                    if is_eos or (FINAL_ANSWER_PREFIX in new_chunk): 
                        finished_branches.append({
                            "text": full_answer_candidate,
                            "score": branch["cumulative_score"],
                            "finished": True
                        })
                    else:
                        passing_branches.append({
                            "cumulative_score": score,
                            "checkpoint": [full_answer_candidate],
                            "finished": False
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
                        "cumulative_score": b["cumulative_score"],
                        "checkpoint": b["checkpoint"],
                        "finished": b["finished"],
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

# ==========================================
# 4. Execution
# ==========================================
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


def test_gsm8k(searcher):
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    random.seed(SEED)
    indices = random.sample(range(len(dataset)), NUM_SAMPLES)
    correct = 0
    for i in indices:
        sample = dataset[i]
        print(f"\n\n=== Question: {sample['question']} ===")
        output_text = searcher.run_parallel(sample['question'])
        answer = extract_result(output_text)

        idx = sample['answer'].find("####")
        if idx == -1:
            print("Warning: Could not find '####' in ground truth answer.", sample['answer'])
            continue
        if answer is None:
            print("No answer extracted.")
            continue
        truth_answer = float(sample['answer'][idx+4:].strip())
        correct += abs(answer - truth_answer) < EPSILON
        if VERBOSE:
            print(f"\n[Generated]: {output_text.strip()}")
            print(f"[Truth]:     {sample['answer']}")
        else:
            print(f"[Generated Answer]: {answer} | [Truth Answer]: {truth_answer}")

    print(f"\n\n=== GSM8K Results: {correct}/{NUM_SAMPLES} correct ===")


def main():
    print("Initializing vLLM Engine...")
    engine = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=16384,
    )

    gen = VLLMGenerator(engine)
    # prm = VLLMPRM(engine)
    # prm = LogitsPRM()
    prm = VLLMCoTPRM(engine)
    
    searcher = HybridSearcher(gen, prm, 512)
    test_gsm8k(searcher)

if __name__ == "__main__":
    main()