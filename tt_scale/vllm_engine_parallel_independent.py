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
   
TAU = 0.8     # threshold score to accept a step  0.8      
MAX_BACKTRACKS = 3     # maximum backtracks before forcing a step forward
NUM_SAMPLES = 30        # number of evaluation samples
FINAL_ANSWER_PREFIX = "<FINAL>"
DEBUG = False
VERBOSE = True 
EPSILON = 1e-3
SEED = 79    

TEMP_ORIGIN = 0.7 # initial temperature
TEMP_STEP = 0.1 # increase per retry

      
MAX_TOTAL_BRANCHES = 6  # maximum total branches at each step
PASSING_MINIMUM = 2   # minimum passing branches to avoid backtracking
KEEPING_BRANCHES = 2 # branches to keep when pruning
MAX_MODEL_LEN = 3072 # maximum model context length
MAX_STEPS = 30    
MAX_FINISHED_BRANCHES = 2    # maximum finished branches to stop
EPSILON = 1e-3     

K_TOKENS = 128 # maximum tokens to generate per step


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

        pa_tokens = self.tokenizer.encode(partial_answer or "")

        max_prompt_tokens = max(1, MAX_MODEL_LEN - 1)
        total = len(prefix_tokens) + len(pa_tokens)
        if total > max_prompt_tokens:
           
            return None  
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

        
        outputs = self.llm.generate([full_context], params, use_tqdm=False)
        result = []
        for out in outputs[0].outputs:
            new_text = out.text
            finish_reason = out.finish_reason
            is_eos = (finish_reason == "stop" and self.stop_token not in new_text)
            result.append((new_text, is_eos))
        
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


class HybridSearcher:
    def __init__(self, generator, prm, max_len=512):
        self.gen = generator
        self.prm = prm
        self.max_len = max_len

    def run_parallel(
        self,
        prompt: str,
        backtrack: bool = True,
        passing_minimum: int = PASSING_MINIMUM,
        tau: float = TAU,
        agg: str = "last",
        max_retries: int = MAX_BACKTRACKS,
        max_branches: int = MAX_TOTAL_BRANCHES,
        expansion_factor: int = 2,
    ) -> str:

        if VERBOSE:
            print(f"\n--- Parallel Run: {prompt[:50]}... ---")

        active_branches = [{"answer": "", "retries": 0, "score": 0.0, "finished": False}]
        completed_branches = []

        step_count = 0
        retries_count = 0
        backtrack_count = 0
        jump_count = 0 

        while active_branches:
            step_count += 1
            if step_count > MAX_STEPS:
                if VERBOSE:
                    print(" -> Reached maximum steps, stopping.")
                break

            if VERBOSE:
                print(f"\n=== Step {step_count} | Active: {len(active_branches)} ===")

            gen_results = {}

            
            groups = {}
            for i, br in enumerate(active_branches):
                groups.setdefault(br["retries"], []).append(i)

            for retry_level, idxs in groups.items():
                contexts = []
                kept_idxs = []

                for i in idxs:
                    br = active_branches[i]
                    ctx = self.gen.build_input_context(prompt, br["answer"])
                    if ctx is None:
                        continue
                    contexts.append(ctx)
                    kept_idxs.append(i)

                if not contexts:
                    continue

                batch_outs = self.gen.generate_batch_step(
                    contexts, retry_attempt=retry_level, M_EXPANSION=1
                )

                for i, seqs in zip(kept_idxs, batch_outs):
                    new_chunk, is_eos = seqs[0]  
                    new_chunk = new_chunk or ""
                    gen_results[i] = (new_chunk, bool(is_eos))

            if not gen_results:
                break

            candidates = []          
            cand_branch_idx = []     
            cand_meta = {}           

            for i, br in enumerate(active_branches):
                if i not in gen_results:
                    continue
                new_chunk, is_step_finished = gen_results[i]

                if (not new_chunk.strip()) and is_step_finished:
                    full_text = br["answer"]
                    cand_meta[i] = (full_text, True)
                    continue

                if not new_chunk.strip() and not is_step_finished:
                    full_text = br["answer"]
                    cand_meta[i] = (full_text, False)
                    continue

                full_text = (br["answer"] + new_chunk) if br["answer"] else new_chunk
                is_finished = is_step_finished or (FINAL_ANSWER_PREFIX in new_chunk)
                cand_meta[i] = (full_text, is_finished)

                candidates.append(full_text + "\n\n")
                cand_branch_idx.append(i)

            score_lookup = {}
            if candidates:
                questions = [prompt]
                partial_answers = [candidates]
                scores_nested = self.prm.get_scores_batch(questions, partial_answers)

                for i, score_seq in zip(cand_branch_idx, scores_nested[0]):
                    if not score_seq:
                        score = 0.0
                    elif agg == "mean":
                        score = float(sum(score_seq) / len(score_seq))
                    elif agg == "mean_only_final":
                        is_finished = cand_meta.get(i, ("", False))[1]
                        score = float(sum(score_seq) / len(score_seq)) if is_finished else float(score_seq[-1])
                    else:  # "last"
                        score = float(score_seq[-1])
                    
                    score_lookup[i] = score


            next_active = []
            rejected_candidates = [] 
            any_retried_this_step = False

            for i, br in enumerate(active_branches):
                full_text, is_finished = cand_meta.get(i, (br["answer"], br["finished"]))
                score = score_lookup.get(i, -1.0)

                if is_finished and (full_text.strip() or br["answer"].strip()):
                    completed_branches.append(
                        {"answer": full_text, "retries": br["retries"], "score": max(br["score"], score), "finished": True}
                    )
                    continue

                threshold = tau - (0.05 * br["retries"])
                passed = score >= threshold

                if VERBOSE:
                    status_icon = "✅" if passed else "❌"
                    print(f" Br {i}: {status_icon} ({score:.2f}/{threshold:.2f})")

                if passed:
                    next_active.append(
                        {"answer": full_text, "retries": 0, "score": score, "finished": False}
                    )
                else:
                    if backtrack and br["retries"] < max_retries:
                        any_retried_this_step = True
                        retries_count += 1
                        clones = expansion_factor if len(next_active) < max_branches else 1
                        for _ in range(clones):
                            next_active.append(
                                {
                                    "answer": br["answer"], 
                                    "retries": br["retries"] + 1,
                                    "score": br["score"],
                                    "finished": False,
                                }
                            )
                    else:
                        rejected_candidates.append({
                            "original_idx": i,
                            "current_score": score,
                            "retries": br["retries"]
                        })

            if any_retried_this_step:
                backtrack_count += 1

            if len(next_active) == 0 and len(rejected_candidates) > 0:
                if VERBOSE:
                    print(" -> ⚠️ All branches failed. Attempting rescue of best rejected candidate.")
                
                best_reject = max(rejected_candidates, key=lambda x: x['current_score'])
                
                
                parent_branch = active_branches[best_reject['original_idx']]
                
                next_active.append({
                    "answer": parent_branch["answer"],
                    "retries": parent_branch["retries"] + 1, 
                    "score": parent_branch["score"],
                    "finished": False
                })
                
                retries_count += 1
                backtrack_count += 1


            # --- 5. Pruning ---
            if len(next_active) > max_branches:
                next_active.sort(key=lambda x: (x["score"], -x["retries"]), reverse=True)
                next_active = next_active[:max_branches]
                if VERBOSE:
                    print(f" -> Pruned to top {max_branches}")

            active_branches = next_active

            if len(completed_branches) >= MAX_FINISHED_BRANCHES:
                break

        stats = [step_count, retries_count, backtrack_count, jump_count]

        if completed_branches:
            best = max(completed_branches, key=lambda x: x["score"])
            if VERBOSE:
                print(f"=> Winner: {best['score']:.3f}")
            return best["answer"], stats

        if active_branches:
            best = max(active_branches, key=lambda x: x["score"])
            return best["answer"], stats

        return "Failed.", stats


# ==========================================
# 4. Execution
# ==========================================
def extract_result(text):
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
    try:
        val = parse_latex_numeric(s)
        if val is not None:
            return val
    except Exception:
        pass

    boxed_match = re.search(r"\\boxed{([^}]+)}", s)
    if boxed_match:
        content = boxed_match.group(1).strip()
        content = content.replace('$', '').replace('\\', '').replace(',', '')
        try:
            nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", content)
            if nums: return float(nums[-1])
        except:
            pass

    try:
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s.replace(',', ''))
        if nums: return float(nums[-1])
    except:
        pass
    return None


def test_MathArena(searcher, backtrack=True,ds_name="MathArena/hmmt_nov_2025", agg="last"):

    if not backtrack:
        tau_local = -1.0  # disable backtracking threshold
        passing_minimum_local = 0
    else:
        tau_local = TAU
        passing_minimum_local = PASSING_MINIMUM
    dataset = load_dataset(ds_name, split="train")

    random.seed(SEED)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"matharena_{MODEL_NAME.replace('/', '_')}_tau{tau_local}_M{MAX_TOTAL_BRANCHES}_K{KEEPING_BRANCHES}_{ds_name.replace('/', '_')}.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index",  "model_answer", "gold_answer", "is_correct"]) 
    print(f"[Log] Writing to: {os.path.abspath(log_file)}")

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

        messages = [
            {"role": "user", "content": str(question)},
            {"role": "assistant", "content": str(output_text)},
        ]
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
        correct, total, stats = test_MathArena(searcher,backtrack=True,ds_name=ds,agg="mean_only_final")
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
