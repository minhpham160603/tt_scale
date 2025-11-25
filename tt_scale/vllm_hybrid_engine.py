import random
import torch
import re
from datasets import load_dataset
from vllm import LLM, SamplingParams

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-4B" # "Qwen/Qwen3-4B"
K_TOKENS = 64          
TAU = 0.5              
MAX_BACKTRACKS = 3     
NUM_SAMPLES = 3        
FINAL_ANSWER_PREFIX = "<FINAL>"

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
            {"role": "assistant", "content": partial_answer}
        ]
        
        # 2. Apply Template (add_generation_prompt=True adds the Assistant header)
        # e.g., "...<|im_start|>user\n{Question}<|im_end|>\n<|im_start|>assistant\n"
        context_prefix = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        # 3. Append the partial answer so far (Prefilling)
        # vLLM will see this as the assistant having already typed this much
        
        return context_prefix

    def generate_step(self, full_context, retry_attempt=0):
        """
        Generates the next step given the full context.
        """
        temp = 0.9 + (0.1 * retry_attempt)
        
        params = SamplingParams(
            temperature=temp,
            max_tokens=K_TOKENS,
            stop=[self.stop_token], 
        )

        # vLLM automatically uses Prefix Caching here.
        # Since 'context_prefix' (System+User) is constant, it is cached.
        # Since 'partial_answer' grows, vLLM caches the shared prefix of the answer.
        outputs = self.llm.generate([full_context], params, use_tqdm=False)
        new_text = outputs[0].outputs[0].text
        
        # Check stop reason
        finish_reason = outputs[0].outputs[0].finish_reason
        cleaned_text = new_text.replace(self.stop_token, "").strip()
        
        # It is finished if it stopped naturally (EOS) or hit our stop token
        is_eos = (finish_reason == "stop" and self.stop_token not in new_text)
        print("Finish Reason:", finish_reason, "| Is EOS:", is_eos)
        
        return cleaned_text, is_eos

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
                
            if FINAL_ANSWER_PREFIX in new_chunk:
                finished = True

            # 3. Score
            full_answer_candidate = current_generated + " " + new_chunk
            score = self.prm.get_score(raw_prompt, full_answer_candidate)
            
            print_chunk = new_chunk.replace('\n', ' ')

            if score >= TAU:
                print(f"  -> KEEP (Score {score:.2f}) | {print_chunk}...")
                checkpoints.append((current_generated + " " + new_chunk, 0))
                if is_eos: finished = True
            else:
                print(f"  -> REJECT (Score {score:.2f}) | {print_chunk}...")
                
                if bt_count < MAX_BACKTRACKS:
                    _, old_bt = checkpoints.pop()
                    checkpoints.append((current_generated, old_bt + 1))
                    print(f"     Retrying... ({old_bt + 1}/{MAX_BACKTRACKS})")
                else:
                    print("     MAX RETRIES. Forced accept.")
                    checkpoints.append((current_generated + " " + new_chunk, 0))

            final_response = checkpoints[-1][0]

        return final_response

# ==========================================
# 4. Execution
# ==========================================
def test_gsm8k(searcher):
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    indices = random.sample(range(len(dataset)), NUM_SAMPLES)
    
    for i in indices:
        sample = dataset[i]
        print(f"\n\n=== Question: {sample['question']} ===")
        result = searcher.run(sample['question'])
        print(f"\n[Generated]: {result.strip()}")
        print(f"[Truth]:     {sample['answer']}")

if __name__ == "__main__":
    print("Initializing vLLM Engine...")
    engine = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=8096,
    )

    gen = VLLMGenerator(engine)
    prm = VLLMPRM(engine)
    
    searcher = HybridSearcher(gen, prm, 512)
    test_gsm8k(searcher)