import random
import torch
import re
from vllm import LLM, SamplingParams
from tt_scale.prm.logits_prm import LogitsPRM
from tt_scale.utils import get_datetime_string
from .constants import *

class VLLMGenerator:
    SYS_PROMPT = f"""You are a genius problem solver. 
Solve the problem step-by-step to avoid mistakes.
After **EVERY logical step** of reasoning, output two newlines (a blank line).
Return final answer with `{FINAL_ANSWER_PREFIX}` prefix (for example: `{FINAL_ANSWER_PREFIX} 16` or `{FINAL_ANSWER_PREFIX} 90.6`)"""

    def __init__(self, llm_engine):
        self.llm = llm_engine
        self.tokenizer = self.llm.get_tokenizer()

    def build_input_context(self, question, partial_answer=""):
        messages = [
            {"role": "system", "content": self.SYS_PROMPT},
            {"role": "user", "content": question},
        ]

        if len(partial_answer) > 0:
            messages.append({"role": "assistant", "content": partial_answer})
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                continue_final_message=True,
                enable_thinking=False
            )
        else:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )

    def handle_output(self, output_obj):
        raw_text = output_obj.text
        finish_reason = output_obj.finish_reason

        last_separator_index = raw_text.rfind(STEP_SEPARATOR)
        step_finished = False

        if finish_reason == "eos_token":
            step_finished = True
            return raw_text.replace(STEP_SEPARATOR, "\n"), step_finished

        if last_separator_index > 50: # Avoid getting stuck
            new_text = raw_text[:last_separator_index]
            new_text = new_text.replace(STEP_SEPARATOR, "\n")
        else:
            new_text = raw_text.replace(STEP_SEPARATOR, "\n")
            if finish_reason == "length" and VERBOSE:
                print(f"  Warning: Hit max_tokens without finding step separator. "
                      f"Generated {len(new_text)} chars.")
        
        return new_text, step_finished

    def generate_step(self, question, partial_answer="",
                      retry_attempt=0, max_tokens=K_TOKENS, num_sequences=1):
        """Generate next step, finding last STEP_SEPARATOR to handle \n\n in code blocks."""
        full_context = self.build_input_context(question, partial_answer)
        temp = 1 + (0.5 * retry_attempt)
        
        params = SamplingParams(
            temperature=temp,
            max_tokens=max_tokens,
            top_p=0.9,
            top_k=40,
            n=num_sequences,
        )

        outputs = self.llm.generate([full_context], params, use_tqdm=False)
        
        new_texts = []
        finishes = []
        for answers in outputs[0].outputs:
            new_text, step_finished = self.handle_output(answers)
            new_texts.append(new_text)
            finishes.append(step_finished)
        return new_texts, finishes

class LogEntry:
    def __init__(self, generated_text, score, backtrack_count):
        self.generated_text = generated_text
        self.score = score
        self.backtrack_count = backtrack_count
    
    def __str__(self):
        return f"\tGenerated Text: {self.generated_text}\n\tScore: {self.score}; Backtrack Count: {self.backtrack_count}"

class HybridSearcher:
    def __init__(self, generator, prm, max_len=1000):
        self.gen = generator
        self.prm = prm
        self.max_len = max_len

    def run(self, raw_prompt):
        print(f"\n--- New Run: {raw_prompt[:40]}... ---")
        log = []
        
        checkpoints = [("", 0)]
        final_response = ""
        finished = False

        while not finished and checkpoints:
            current_generated, bt_count = checkpoints[-1]
            
            if len(current_generated) > self.max_len * 4:
                break

            new_chunks, finishes = self.gen.generate_step(
                raw_prompt,
                current_generated,
                retry_attempt=bt_count,
                num_sequences= 1 if bt_count == 0 else 8
            )
            # for now we just want to pick the best answers
            best_score = -1
            best_chunk = ""
            best_full_answer_candidate = ""
            for new_chunk, step_finished in zip(new_chunks, finishes, strict=True):
                if not new_chunk:
                    if step_finished:
                        finished = True
                        break
                    continue
                if current_generated and not current_generated.endswith(STEP_SEPARATOR):
                    full_answer_candidate = current_generated + STEP_SEPARATOR + new_chunk
                else:
                    full_answer_candidate = current_generated + new_chunk
                score = self.prm.get_score(raw_prompt, full_answer_candidate)[-1]
                if score > best_score:
                    best_score = score
                    best_chunk = new_chunk
                    best_full_answer_candidate = full_answer_candidate
                    finished = step_finished
            
            if best_score == -1: # all answers are empty
                checkpoints.pop()
                continue

            score = best_score
            new_chunk = best_chunk
            full_answer_candidate = best_full_answer_candidate
                
            log.append(LogEntry(new_chunk, score, bt_count))

            threshold = TAU - 0.05*bt_count
            if score >= threshold or bt_count >= MAX_BACKTRACKS:
                if score >= threshold:
                    print(f"  -> KEEP (Score {score:.2f})")
                else:
                    print(f"  -> FORCE KEEP (Score {score:.2f})")
                checkpoints.append((full_answer_candidate, 0))
                if (FINAL_ANSWER_PREFIX in new_chunk):
                    finished = True
            else:
                print(f"  -> REJECT (Score {score:.2f})")
                checkpoints.pop()
                checkpoints.append((current_generated, bt_count + 1))
                print(f"     Retrying... ({bt_count + 1}/{MAX_BACKTRACKS})")
            final_response = checkpoints[-1][0]

        return final_response, log

def main():
    from .utils import get_datasets
    sample = get_datasets("openai/gsm8k", 12)[11]
    question = sample['question']
    print(sample)

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
    hybrid_searcher = HybridSearcher(gen, LogitsPRM())
    final_response = hybrid_searcher.run(question)
    print(final_response)


if __name__ == "__main__":
    main()