import random
import torch
import re
from datasets import load_dataset
import os
import csv
from vllm import LLM, SamplingParams
from tt_scale.prm.logits_prm import LogitsPRM
from tt_scale.prm.vllm_cot_prm import VLLMCoTPRM
from tt_scale.prm.qwen_math_prm import QwenMathPRM
import math
from typing import List, Optional, Tuple
from transformers import BitsAndBytesConfig
from tt_scale.grader import extract_and_grade



# --- Configuration ---
# MODEL_NAME = "Qwen/Qwen3-4B-AWQ" # "Qwen/Qwen3-4B"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct-AWQ"
# MODEL_NAME = "warshanks/Qwen3-4B-Instruct-2507-AWQ"  
   
TAU = 0.6     # threshold score to accept a step  0.6      
MAX_BACKTRACKS = 3     # maximum backtracks before forcing a step forward
NUM_SAMPLES = 30        # number of evaluation samples
FINAL_ANSWER_PREFIX = "<FINAL>"
DEBUG = False
VERBOSE = True 
EPSILON = 1e-3
SEED = 79    

TEMP_ORIGIN = 0.7 # initial temperature
TEMP_STEP = 0.1 # increase per retry


# Parallel hybrid search

      
MAX_TOTAL_BRANCHES = 6  # maximum total branches at each step
PASSING_MINIMUM = 2   # minimum passing branches to avoid backtracking
KEEPING_BRANCHES = 2 # branches to keep when pruning
MAX_MODEL_LEN = 3072 # maximum model context length
MAX_STEPS = 30    
MAX_FINISHED_BRANCHES = 2    # maximum finished branches to stop
EPSILON = 1e-3     

K_TOKENS = 128 # maximum tokens to generate per step


# ==========================================
# 1. VLLM Generator Wrapper
# ==========================================
class VLLMGenerator:
    STOP_STRING = "<END_STEP>"
    SYS_PROMPT = f"""You are a genius problem solver. 
    Solve the problem step-by-step to avoid mistakes.
    After **EVERY logical step** of reasoning, output the token {STOP_STRING}.
    If all steps are completed, return final answer with `{FINAL_ANSWER_PREFIX}` prefix, and Put your final answer within \\boxed{{}}.(for example: `{FINAL_ANSWER_PREFIX} \\boxed{16}` or `{FINAL_ANSWER_PREFIX} \\boxed{90.6}`)"""

    def __init__(self, llm_engine):
        self.llm = llm_engine
        self.tokenizer = self.llm.get_tokenizer()
        self.stop_token = self.STOP_STRING

    def build_input_context(self, question, partial_answer=""):
        """
        Build the chat prompt and left-truncate assistant partial_answer at token level
        to fit within the model length budget. We reserve 1 token for decoding to
        avoid vLLM pre-validation errors (prompt + at least 1 token > max length).
        """
        # 1) Build conversation prefix as tokens (System + User + assistant header)
        messages = [
            {"role": "system", "content": self.SYS_PROMPT},
            {"role": "user", "content": question},
        ]
        prefix_tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # 2) Encode assistant partial tokens
        pa_tokens = self.tokenizer.encode(partial_answer or "")

        # 3) Enforce budget: allow at most MAX_MODEL_LEN - 1 prompt tokens
        max_prompt_tokens = max(1, MAX_MODEL_LEN - 1)
        total = len(prefix_tokens) + len(pa_tokens)
        if total > max_prompt_tokens:
            # keep_pa = max(0, max_prompt_tokens - len(prefix_tokens))
            # pa_tokens = pa_tokens[-keep_pa:] if keep_pa > 0 else []
            return None  # exceed model length
        merged_tokens = prefix_tokens + pa_tokens
        return self.tokenizer.decode(merged_tokens)

    def generate_step(self, full_context, retry_attempt=0, M_EXPANSION=1):
        """
        Generates the next step given the full context.
        """
        temp = TEMP_ORIGIN + (TEMP_STEP * retry_attempt)
        
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
        temp = TEMP_ORIGIN + (TEMP_STEP * retry_attempt)
        params = SamplingParams(
            temperature=temp,
            max_tokens=K_TOKENS,
            top_p=0.9,
            top_k=40,
            n=M_EXPANSION, # number of expansions per prompt
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

    # def run(self, raw_prompt):
    #     print(f"\n--- New Run: {raw_prompt[:40]}... ---")
        
    #     # State: (current_generated_text, backtrack_count)
    #     checkpoints = [("", 0)]
        
    #     final_response = ""
    #     finished = False

    #     while not finished and checkpoints:
    #         current_generated, bt_count = checkpoints[-1]
            
    #         # 1. Ask Generator to build the full context string
    #         # This cleanly separates prompt formatting logic from search logic
    #         full_gen_context = self.gen.build_input_context(raw_prompt, current_generated)

    #         if len(full_gen_context) > self.max_len * 4: 
    #             break

    #         # 2. Generate Step
    #         outputs = self.gen.generate_step(full_gen_context, retry_attempt=bt_count)
    #         new_chunk, is_eos = outputs[0]  
            
    #         if not new_chunk:
    #             if is_eos: finished = True
    #             checkpoints.pop()
    #             continue
                
    #         # 3. Score
    #         full_answer_candidate = current_generated + new_chunk
    #         score = self.prm.get_score(raw_prompt, full_answer_candidate)

    #         if score >= TAU or bt_count >= MAX_BACKTRACKS:
    #             if score >= TAU:
    #                 print(f"  -> KEEP (Score {score:.2f})")
    #             else:
    #                 print(f"  -> FORCE KEEP (Score {score:.2f})")
    #             checkpoints.append((full_answer_candidate, 0))
    #             if is_eos or (FINAL_ANSWER_PREFIX in new_chunk):
    #                 finished = True
    #         else:
    #             print(f"  -> REJECT (Score {score:.2f})")
    #             checkpoints.pop()
    #             checkpoints.append((current_generated, bt_count + 1))
    #             print(f"     Retrying... ({bt_count + 1}/{MAX_BACKTRACKS})")
    #         final_response = checkpoints[-1][0]
    #     return final_response

    def run_parallel(self, prompt: str, backtrack=True, passing_minimum: int = PASSING_MINIMUM, tau: float = TAU, agg="last") -> str:
        
        if VERBOSE:
            print(f"\n--- Parallel Run: {prompt[:50]}... ---")
        
        # Initialize active branches
        active_branches = [
            {
                "score": 0.0,
                "average_sub_score": 0.0, # for sorting pruning when backtracking
                "checkpoint": [],
                "finished": False,
                "branch_steps": 0,
            }
        ]

        
        finished_branches = [] # collected finished branches
        protected_branches = [] # protect passed branches when backtracking
        step_count = 0 # number of steps taken
        retries_count = 0 # total retries across all steps
        jump_count = 0 # total jumps across all steps
        backtrack_count = 0 # total backtracks(one backtrack can have multiple retries) across all steps
        retries = 1 # number of retries for current step
        jumping = False # whether to force a jump forward
        finished = False # whether overall generation is finished

        while active_branches : # and not finished
            
            step_count += 1
            if step_count > MAX_STEPS:
                if VERBOSE:
                    print("  -> Reached maximum steps, stopping.")
                break
            
            if VERBOSE:
                print(f"--- Step {step_count} | Generating for {len(active_branches)} branches... ---")
            
            next_active_branches = []
            passing_branches = [] # to store branches that pass this step, and will be keeped while backtracking
            passing = 0

            # if jumping, temporarily set passing threshold to 0
            if jumping:
                passing_threshold = 0.0
                jumping = False
                retries = 1
            else:
                passing_threshold = tau

            contexts = []
            branch_indices = []
            # Determine how many branches to expand based on the number of active branches
            if backtrack:
                m_expansion = max(1, math.floor(MAX_TOTAL_BRANCHES/len(active_branches)))
                extra = MAX_TOTAL_BRANCHES - (m_expansion * len(active_branches))
            else:
                m_expansion = int(MAX_TOTAL_BRANCHES/KEEPING_BRANCHES)
                extra = 0


            for i, branch in enumerate(active_branches):
                if branch["finished"]:
                    continue
                current_generated = branch["checkpoint"][-1] if branch["checkpoint"] else ""
                ctx = self.gen.build_input_context(prompt, current_generated)
                if ctx is None:
                    if VERBOSE:
                        print(f"  Br {i}: Dropped (prompt budget exceeded)")
                    continue
                contexts.append(ctx)
                branch_indices.append(i)
            
            batch_outputs = self.gen.generate_batch_step(contexts, retry_attempt=retries, M_EXPANSION=m_expansion)

            for bi, seqs in zip(branch_indices, batch_outputs):
                branch = active_branches[bi]
                current_generated = branch["checkpoint"][-1] if branch["checkpoint"] else ""
                candidates = [current_generated + new_chunk + "\n\n" for (new_chunk, _) in seqs]

                # qa_pairs = [(prompt, cand) for cand in candidates]
                questions = [prompt] 
                partial_answers = [candidates]
                scores = self.prm.get_scores_batch(questions, partial_answers)

                for j, ((new_chunk, is_eos), score_all) in enumerate(zip(seqs, scores[0])):
                    full_answer_candidate = current_generated + new_chunk
                    if agg == "last":
                        score = score_all[-1]
                    elif agg == "mean":
                        # score = score_all[-1]
                        score = sum(score_all)/len(score_all)
                    elif agg == "ema":
                        score = 0.5 * branch["score"] + 0.5 * score_all[-1]
                    branch["average_sub_score"] += score
                    
                    if score > passing_threshold:
                        if VERBOSE:
                            print(f"  Br {j}: ✅ Pass ({score:.2f})")
                        passing += 1
                        
                        if is_eos or (FINAL_ANSWER_PREFIX in new_chunk): 
                            if agg == "mean":
                                score = sum(score_all)/len(score_all)
                            finished_branches.append({
                                "text": full_answer_candidate,
                                "score": score,
                                "finished": True
                            })

                            # if len(finished_branches) >= MAX_FINISHED_BRANCHES:
                            #     print("  -> Reached max finished branches, stopping.")
                            #     finished = True
                        else:
                            passing_branches.append({
                                "score": score,
                                "checkpoint": [full_answer_candidate],
                                "finished": False,
                                "average_sub_score": 0.0,
                                "text": full_answer_candidate,
                            })
                    else:
                        if VERBOSE:
                            print(f"  Br {j}: ❌ Fail ({score:.2f})")
                        

            # if enough passing branches, not backtracking, and prune to KEEPING_BRANCHES
            if passing + len(protected_branches) >= passing_minimum:
                if VERBOSE:
                    print(f"  -> Sufficient passing branches ({passing}), pruning to {KEEPING_BRANCHES}.")
                passing_branches.extend(protected_branches)
                passing_branches.sort(key=lambda x: x["score"], reverse=True)
                if passing > KEEPING_BRANCHES:
                    active_branches = passing_branches[:KEEPING_BRANCHES]
                else:
                    active_branches = passing_branches
                retries = 1  
                protected_branches = [] # to store passing branches for multiple backtracks

            # Otherwise, backtrack
            else:
                if VERBOSE:
                    print(f"  -> Insufficient passing branches ({passing}), backtracking.")
                # sort the parent branches based on the average score of children branches
                active_branches.sort(key=lambda x: x["average_sub_score"], reverse=True)
                # when backtracking, add passing_branches to protected_branches
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
                        "average_sub_score": 0.0,
                    })
                    # print("remaining active branches:", len(active_branches))
                if retries == 1:
                    backtrack_count += 1
                retries += 1
                retries_count += 1
                if retries > MAX_BACKTRACKS:
                    if VERBOSE:
                        print("  -> Reached maximum backtracks, forcing jump forward.")
                    jumping = True
                    retries = 1
                    jump_count += 1

                    
        stats = [step_count, retries_count, backtrack_count, jump_count]
            

        if not finished_branches:
            if active_branches:
                best = max(active_branches, key=lambda x: x["score"])
                if best["checkpoint"]:
                    return best["checkpoint"][-1],stats
                elif "text" in best:
                    return best["text"],stats
                else:
                    return "Failed.", stats
            return "Failed.", stats

        best = max(finished_branches, key=lambda x: x["score"])
        if VERBOSE:
            print(f"=> Winner: {best['score']:.3f}")
        return best["text"], stats


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
    # Try to parse LaTeX-style numeric expressions after the final marker first
    try:
        mfa = re.search(r"(?i)(?:<FINAL>|Final\s*answer\s*:)[\s\S]*$", str(text))
        if mfa:
            tail = mfa.group(0)
            val = parse_latex_numeric(tail)
            if val is not None:
                return val
    except Exception:
        pass

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
            return float(boxed_nums[-1])  
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
    
    if text is None:
        return None
    try:
        s = str(text)
    except Exception:
        return None

    # Try LaTeX numeric parsing (supports \boxed, \frac, \sqrt)
    try:
        val = parse_latex_numeric(s)
        if val is not None:
            return val
    except Exception:
        pass

    # 1. Prefer Boxed LaTeX
    boxed_match = re.search(r"\\boxed{([^}]+)}", s)
    if boxed_match:
        content = boxed_match.group(1).strip()
        content = content.replace('$', '').replace('\\', '').replace(',', '')
        try:
            nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", content)
            if nums: return float(nums[-1])
        except:
            pass

    # 2. Fallback: Look for the last number in the text
    try:
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s.replace(',', ''))
        if nums: return float(nums[-1])
    except:
        pass
    return None









def test_MathArena(searcher, backtrack=True,ds_name="MathArena/hmmt_nov_2025", agg="last"):
    # Decide effective thresholds per run to avoid local/global confusion
    if not backtrack:
        tau_local = -1.0  # disable backtracking threshold
        passing_minimum_local = 0
    else:
        tau_local = TAU
        passing_minimum_local = PASSING_MINIMUM
    dataset = load_dataset(ds_name, split="train")

    random.seed(SEED)
    # Prepare log file at project root to ensure discoverability
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"matharena_{MODEL_NAME.replace('/', '_')}_tau{tau_local}_M{MAX_TOTAL_BRANCHES}_K{KEEPING_BRANCHES}_{ds_name.replace('/', '_')}.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index",  "model_answer", "gold_answer", "is_correct"]) 
    print(f"[Log] Writing to: {os.path.abspath(log_file)}")
    # indices = random.sample(range(len(dataset)), NUM_SAMPLES)
    correct = 0
    total = 0
    step_count = 0
    retries_count = 0
    backtrack_count = 0
    jump_count = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        question = sample.get("question", sample.get("problem"))
        if VERBOSE:
            print(f"\n\n=== Question: {question} ===")
        output_text, stats = searcher.run_parallel(
            question,
            backtrack=backtrack,
            passing_minimum=passing_minimum_local,
            tau=tau_local,
            agg = agg
        )
        answer_text = sample.get("answer", sample.get("solution"))

        # Build proper messages: user then assistant (last must be assistant)
        messages = [
            {"role": "user", "content": str(question)},
            {"role": "assistant", "content": str(output_text)},
        ]
        # Approximate output token count via tokenizer
        tokenizer = searcher.gen.tokenizer
        output_tokens = len(tokenizer.encode(str(output_text)))
        competition_config = {"final_answer": True, "strict_parsing": False}

        model_answer = None
        warning_value = None
        try:
            model_answer, is_correct, warning_value = extract_and_grade(
                messages, output_tokens, str(answer_text), competition_config, debug_info="MathArena"
            )
        except Exception as e:
            print("extract_and_grade error:", e)
            # Fallback to numeric/symbolic comparison
            ok, pv, tv = compare_answers(output_text, answer_text, EPSILON)
            is_correct = ok
            model_answer = output_text

        total += 1
        correct += 1 if is_correct else 0
        step_count += stats[0]
        retries_count += stats[1]
        backtrack_count += stats[2]
        jump_count += stats[3]

        # Append structured log record
        try:
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    i,
                    str(model_answer),
                    str(answer_text),
                    bool(is_correct),
                ])
        except Exception as le:
            print("log write error:", le)

        if VERBOSE: 
            print(f"[Generated Answer]: {model_answer} | [Truth Answer]: {sample.get('answer', sample.get('solution'))}")
            print("now correct:", correct, "/", total)

    print(f"\n\n=== {ds_name} Results: {correct}/{total} correct ===")
    return correct, total, (step_count, retries_count, backtrack_count, jump_count)

def main():
    print("Initializing vLLM Engine...")
    engine = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=MAX_MODEL_LEN,
        # quantization="awq",
        # quantization="bitsandbytes"
    )

    gen = VLLMGenerator(engine)
    # prm = VLLMPRM(engine)
    # prm = LogitsPRM()
    prm = QwenMathPRM()
    # prm = VLLMCoTPRM(engine)
    
    searcher = HybridSearcher(gen, prm)

    datasets = ["MathArena/hmmt_nov_2025","MathArena/aime_2025","MathArena/cmimc_2025","MathArena/brumo_2025","MathArena/apex_2025","MathArena/hmmt_nov_2025"]
    # test_gsm8k(searcher)
    corrects, totals = 0, 0
    step_counts, retries_counts, backtrack_counts, jump_counts = 0,0,0,0
    for ds in datasets:
        print(f"\n\n\n==== Testing on dataset: {ds} ====")
        correct, total, stats = test_MathArena(searcher,backtrack=True,ds_name=ds,agg="mean")
        corrects += correct
        totals += total
        step_counts += stats[0]
        retries_counts += stats[1]
        backtrack_counts += stats[2]
        jump_counts += stats[3]
        # print(f"==== Results for {ds}: {correct}/{total} correct ====")
    print(f"\n\n\n==== Overall Results: {corrects}/{totals} correct ====")
    print(f"Steps: {step_counts}, Retries: {retries_counts}, Backtracks: {backtrack_counts}, Jumps: {jump_counts}")
if __name__ == "__main__":
    main()
