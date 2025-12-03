import random
import torch
import re
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tt_scale.prm.logits_prm import LogitsPRM
from tt_scale.prm.vllm_cot_prm import VLLMCoTPRM
import math
from typing import List, Optional, Tuple


# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-4B-AWQ" # "Qwen/Qwen3-4B"
K_TOKENS = 256        
TAU = 0.7             
MAX_BACKTRACKS = 3     
NUM_SAMPLES = 30        
FINAL_ANSWER_PREFIX = "<FINAL>"
DEBUG = False
VERBOSE = True 
EPSILON = 1e-3
SEED = 79      


# Parallel hybrid search

      
MAX_TOTAL_BRANCHES = 12  
PASSING_MINIMUM = 3   # minimum passing branches to avoid backtracking
KEEPING_BRANCHES = 4
# MAX_STEPS = 100    
MAX_FINISHED_BRANCHES = 4    
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

    def generate_step(self, full_context, retry_attempt=0, M_EXPANSION=1):
        """
        Generates the next step given the full context.
        """
        temp = 0.6 + (0.5 * retry_attempt)
        
        params = SamplingParams(
            temperature=temp,
            max_tokens=K_TOKENS,
            top_p=0.9,
            top_k=40, 
            n = M_EXPANSION,
            stop=[self.stop_token],
        )

        # vLLM automatically uses Prefix Caching here.
        # Since 'context_prefix' (System+User) is constant, it is cached.
        # Since 'partial_answer' grows, vLLM caches the shared prefix of the answer.
        outputs = self.llm.generate([full_context], params, use_tqdm=False)
        result = []
        for out in outputs[0].outputs:
            new_text = out.text
            finish_reason = out.finish_reason
            is_eos = (finish_reason == "stop" and self.stop_token not in new_text)
            result.append((new_text, is_eos))
        # new_text = outputs[0].outputs[0].text
        # finish_reason = outputs[0].outputs[0].finish_reason
        # is_eos = (finish_reason == "stop" and self.stop_token not in new_text)

            if DEBUG:
                print(">>>>>>>>>>>>>>>>>>>>")
                print("GEN_STEP: ", new_text)
                print("<<<<<<<<<<<<<<<<<<<<")
                print("Finish Reason:", finish_reason, "| Is EOS:", is_eos)
        
        return result

    def generate_batch_step(self, full_contexts, retry_attempt=0, M_EXPANSION=1):
        temp = 1 + (0.5 * retry_attempt)
        params = SamplingParams(
            temperature=temp,
            max_tokens=K_TOKENS,
            top_p=0.9,
            top_k=40,
            n=M_EXPANSION,
            stop=[self.stop_token],
        )
        outputs = self.llm.generate(full_contexts, params, use_tqdm=False)
        
        batch_result = []
        for o in outputs:
            seqs = []
            for out in o.outputs:
                new_text = out.text
                is_eos = (out.finish_reason == "stop" and self.stop_token not in new_text)
                seqs.append((new_text, is_eos))
            batch_result.append(seqs)
        return batch_result

# ==========================================
# 2. VLLM PRM Wrapper
# ==========================================
class VLLMPRM:
    def __init__(self, llm_engine):
        self.llm = llm_engine
        self.tokenizer = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=0.0, max_tokens=10, stop=["\n"])

    def build_judge_prompt(self, question: str, partial_answer: str) -> str:
        
        judge_prompt = f"""
Review the following partial solution to a math problem.
---
Question: {question}
Partial Answer So Far: {partial_answer}
---
Rate the logical correctness of the LAST step in the Partial Answer on a scale of 1 to 10.
Output ONLY the number.
"""
        messages = [
            {"role": "system", "content": "You are a strict math grader. Output only numerical scores."},
            {"role": "user", "content": judge_prompt}
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

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
    
    def get_scores_batch(self, question, partial_answers):
        prompts = [self.build_judge_prompt(question, pa) for pa in partial_answers]
        outs = self.llm.generate(prompts, self.params, use_tqdm=False)
        scores = []
        for o in outs:
            txt = o.outputs[0].text
            m = re.search(r"(\d+(\.\d+)?)", txt)
            if m:
                try:
                    s = max(1.0, min(10.0, float(m.group(1)))) / 10.0
                except:
                    s = 0.0
            else:
                s = 0.0
            scores.append(s)
        return scores

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
            outputs = self.gen.generate_step(full_gen_context, retry_attempt=bt_count)
            new_chunk, is_eos = outputs[0]  
            
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
                "score": 0.0,
                "average_score": 0.0, # for sorting pruning when backtracking
                "checkpoint": [],
                "finished": False,
            }
        ]


        finished_branches = []
        protected_branches = [] # protect passed branches when backtracking
        step_count = 0
        retries = 1
        jumping = False
        finished = False

        while active_branches and not finished:
            
            step_count += 1
            
            
            print(f"--- Step {step_count} | Generating for {len(active_branches)} branches... ---")
            
            next_active_branches = []
            passing_branches = []
            passing = 0
            if jumping:
                passing_threshold = 0.0
                jumping = False
                retries = 1
            else:
                passing_threshold = TAU

            contexts = []
            branch_indices = []
            m_expansion = max(1, math.floor(MAX_TOTAL_BRANCHES/len(active_branches)))
            extra = MAX_TOTAL_BRANCHES - (m_expansion * len(active_branches))
            for i, branch in enumerate(active_branches):
                if branch["finished"]:
                    continue
                current_generated = branch["checkpoint"][-1] if branch["checkpoint"] else ""
                ctx = self.gen.build_input_context(prompt, current_generated)
                contexts.append(ctx)
                branch_indices.append(i)
            
            batch_outputs = self.gen.generate_batch_step(contexts, retry_attempt=retries, M_EXPANSION=m_expansion)

            for bi, seqs in zip(branch_indices, batch_outputs):
                branch = active_branches[bi]
                current_generated = branch["checkpoint"][-1] if branch["checkpoint"] else ""
                candidates = [current_generated + new_chunk for (new_chunk, _) in seqs]

                qa_pairs = [(prompt, cand) for cand in candidates]
                scores = self.prm.get_scores_batch(qa_pairs)

                for j, ((new_chunk, is_eos), score) in enumerate(zip(seqs, scores)):
                    full_answer_candidate = current_generated + new_chunk
                    branch["average_score"] += score

                    if score > passing_threshold:
                        print(f"  Br {j}: ✅ Pass ({score:.2f})")
                        passing += 1
                        
                        if is_eos or (FINAL_ANSWER_PREFIX in new_chunk): 
                            finished_branches.append({
                                "text": full_answer_candidate,
                                "score": score,
                                "finished": True
                            })

                            if len(finished_branches) >= MAX_FINISHED_BRANCHES:
                                print("  -> Reached max finished branches, stopping.")
                                finished = True
                        else:
                            passing_branches.append({
                                "score": score,
                                "checkpoint": [full_answer_candidate],
                                "finished": False,
                                "average_score": 0.0,
                                "text": full_answer_candidate,
                            })
                    else:
                        print(f"  Br {j}: ❌ Fail ({score:.2f})")
                        

            # if enough passing branches, not backtracking, and prune to KEEPING_BRANCHES
            if passing + len(protected_branches) >= PASSING_MINIMUM:
                print(f"  -> Sufficient passing branches ({passing}), pruning to {KEEPING_BRANCHES}.")
                passing_branches.extend(protected_branches)
                passing_branches.sort(key=lambda x: x["score"], reverse=True)
                if passing > KEEPING_BRANCHES:
                    active_branches = passing_branches[:KEEPING_BRANCHES]
                else:
                    active_branches = passing_branches
                retries = 1  
                protected_branches = []


            else:
                print(f"  -> Insufficient passing branches ({passing}), backtracking.")

                active_branches.sort(key=lambda x: x["average_score"], reverse=True)
                protected_branches.extend(passing_branches)
                # Eliminate low-score branches
                backtrack_num = KEEPING_BRANCHES - retries/(MAX_BACKTRACKS+1) * (KEEPING_BRANCHES)

                backtrack_candidates = active_branches[:min(math.ceil(backtrack_num), len(active_branches))]
                
                active_branches = []
                for b in backtrack_candidates:
                    active_branches.append({
                        "score": b["score"],
                        "checkpoint": b["checkpoint"],
                        "finished": b["finished"],
                        "average_score": 0.0,
                    })
                    # print("remaining active branches:", len(active_branches))
                
                retries += 1
                if retries > MAX_BACKTRACKS:
                    print("  -> Max retries reached, forcing jumping")
                    jumping = True
                    retries = 1

                    

            
            if not active_branches and not finished_branches:
                return "Failed to generate a solution."

        if not finished_branches:
            if active_branches:
                best = max(active_branches, key=lambda x: x["score"])
                if best["checkpoint"]:
                    return best["checkpoint"][-1]
                elif "text" in best:
                    return best["text"]
                else:
                    return "Failed."
            return "Failed."

        best = max(finished_branches, key=lambda x: x["score"])
        print(f"=> Winner: {best['score']:.3f}")
        return best["text"]

    def run_parallel_backtrack(self, prompt: str) -> str:
        print(f"\n--- Parallel Backtrack Run: {prompt[:50]}... ---")

        # Each branch keeps a partial answer and scoring history
        active_branches = [
            {
                "partial": "",           # partial answer text so far
                "retries": 0,            # how many times we re-sampled this branch
                "cumulative_score": 1.0, # used for pruning
                "history": [1.0],        # list of step scores
            }
        ]
        finished_branches = []

        step_count = 0
        # optional safety cap (local, doesn't affect teammate's code)
        MAX_STEPS_LOCAL = 50

        while active_branches:
            step_count += 1
            if step_count > MAX_STEPS_LOCAL:
                print("Max steps reached in backtrack parallel search.")
                break

            print(f"--- Step {step_count} | Generating for {len(active_branches)} branches... ---")

            # Build full prompts for all active branches
            contexts = [
                self.gen.build_input_context(prompt, b["partial"])
                for b in active_branches
            ]

            # Simple length guard
            if any(len(ctx) > self.max_len * 4 for ctx in contexts):
                print("Context too long, stopping.")
                break

            # One sample per branch; cloning is handled in this method
            try:
                batch_outputs = self.gen.generate_batch_step(
                    contexts,
                    retry_attempt=0,   # could be tuned; kept simple here
                    M_EXPANSION=1,
                )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print("OOM Error! Clearing cache and reducing branches.")
                    torch.cuda.empty_cache()
                    active_branches = active_branches[:1]
                    continue
                raise e

            next_active_branches = []

            for i, (branch, seqs) in enumerate(zip(active_branches, batch_outputs)):
                # We set M_EXPANSION=1 above, so seqs should have length 1
                if not seqs:
                    continue
                new_chunk, is_eos = seqs[0]

                if not new_chunk:
                    # If nothing new but EOS, treat current partial as finished
                    if is_eos and branch["partial"]:
                        finished_branches.append(
                            {"text": branch["partial"], "score": branch["cumulative_score"]}
                        )
                    continue

                candidate_answer = branch["partial"] + new_chunk

                # PRM score for this extended branch
                score = self.prm.get_score(prompt, candidate_answer)

                # Accept branch if score high enough or retries exhausted
                if score >= TAU or branch["retries"] >= MAX_RETRIES:
                    if score >= TAU:
                        print(f"  Br {i}: ✅ Pass ({score:.2f})")
                    else:
                        print(f"  Br {i}: ⚠️ Force-keep ({score:.2f})")

                    new_branch = {
                        "partial": candidate_answer,
                        "retries": 0,
                        "cumulative_score": score,
                        "history": branch["history"] + [score],
                    }

                    # Finish if EOS or final answer detected
                    if is_eos or (FINAL_ANSWER_PREFIX in new_chunk):
                        finished_branches.append(
                            {"text": candidate_answer, "score": score}
                        )
                    else:
                        next_active_branches.append(new_branch)
                else:
                    # Score below threshold → either split (resample) or prune
                    if branch["retries"] < MAX_RETRIES:
                        # Decide how many clones to spawn
                        current_total = (
                            len(next_active_branches)
                            + len(active_branches)
                            - (i + 1)
                        )
                        num_clones = 1
                        if current_total < MAX_TOTAL_BRANCHES:
                            num_clones = M_EXPANSION

                        print(
                            f"  Br {i}: ❌ Fail ({score:.2f}) -> Split {num_clones}"
                        )

                        for _ in range(num_clones):
                            next_active_branches.append(
                                {
                                    "partial": branch["partial"],
                                    "retries": branch["retries"] + 1,
                                    "cumulative_score": branch["cumulative_score"],
                                    "history": branch["history"],
                                }
                            )
                    else:
                        print(f"  Br {i}: 💀 Pruned (max retries reached)")

            # Global pruning: keep at most MAX_TOTAL_BRANCHES
            if len(next_active_branches) > MAX_TOTAL_BRANCHES:
                next_active_branches.sort(
                    key=lambda x: x["cumulative_score"], reverse=True
                )
                next_active_branches = next_active_branches[:MAX_TOTAL_BRANCHES]

            active_branches = next_active_branches

            if not active_branches and not finished_branches:
                return "Failed to generate a solution."

        # Choose best finished, else best active
        if not finished_branches:
            if active_branches:
                best = max(active_branches, key=lambda x: x["cumulative_score"])
                return best["partial"]
            return "Failed."

        best = max(finished_branches, key=lambda x: x["score"])
        print(f"=> Winner (backtrack): {best['score']:.3f}")
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
    # pattern = r"<FINAL>\s*([\d\.]+)"
    pattern1 = r"(?i)(?:<FINAL>|Final\s*answer\s*:)[\s\S]*?([-+]?\d*\.?\d+)"
    m1 = re.search(pattern1, text)
    if m1:
        try:
            return float(m1.group(1))
        except ValueError:
            pass

    boxed_nums = re.findall(r"\\boxed\{\s*([-+]?\d*\.?\d+)\s*\}", text)
    if boxed_nums:
        try:
            return float(boxed_nums[-1])  # 取最后一个 boxed 数字
        except ValueError:
            pass

    nums = re.findall(r"([-+]?\d*\.?\d+)", text)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            pass

    return None

def extract_math_answer(text: str) -> Optional[float]:
    """Robust extraction for MATH dataset answers (\boxed{...}) or simple numbers."""
    # 1. Prefer Boxed LaTeX
    boxed_match = re.search(r"\\boxed{([^}]+)}", text)
    if boxed_match:
        content = boxed_match.group(1).strip()
        content = content.replace('$', '').replace('\\', '').replace(',', '')
        try:
            nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", content)
            if nums: return float(nums[-1])
        except: pass

    # 2. Fallback: Look for the last number in the text
    try:
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text.replace(',', ''))
        if nums: return float(nums[-1])
    except: pass
    return None


def test_gsm8k(searcher):
    # dataset = load_dataset("openai/gsm8k", "main", split="test")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    random.seed(SEED)
    indices = random.sample(range(len(dataset)), NUM_SAMPLES)
    correct = 0
    total = 0
    for i in indices:
        sample = dataset[i]
        question = sample.get("question", sample.get("problem"))
        print(f"\n\n=== Question: {question} ===")
        output_text = searcher.run_parallel(question)
        answer = extract_result(output_text)

        answer_text = sample.get("answer", sample.get("solution"))
        # idx = answer_text.find("####")
        # if idx == -1:
        #     print("Warning: Could not find '####' in ground truth answer.", answer_text)
        #     continue
        # if answer is None:
        #     print("No answer extracted.")
        #     print("Generated Text:", output_text)
        #     # continue
        #     break
        # truth_answer = float(answer_text[idx+4:].strip())
        truth_answer = extract_math_answer(answer_text) 
        if answer is None or truth_answer is None:
            print("No answer extracted.")
            print("Generated Text:", output_text)
            print("Truth Text:", answer_text)
            continue
        total += 1
        correct += abs(answer - truth_answer) < EPSILON

        if VERBOSE:
            print(f"\n[Generated]: {output_text.strip()}")
            print(f"[Truth]:     {answer_text}")
        else:
            print(f"[Generated Answer]: {answer} | [Truth Answer]: {truth_answer}")
        print("now correct:", correct, "/", total)

    print(f"\n\n=== GSM8K Results: {correct}/{NUM_SAMPLES} correct ===")


def main():
    print("Initializing vLLM Engine...")
    engine = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=1536,
        quantization="awq"
    )

    gen = VLLMGenerator(engine)
    # prm = VLLMPRM(engine)
    # prm = LogitsPRM()
    prm = VLLMCoTPRM(engine)
    
    searcher = HybridSearcher(gen, prm, 512)
    test_gsm8k(searcher)

if __name__ == "__main__":
    main()
