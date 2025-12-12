import random
import torch
import re
from collections import defaultdict
from vllm import LLM, SamplingParams
from tt_scale.prm.logits_prm import LogitsPRM
from tt_scale.utils import get_datetime_string
from .constants import *
from datasets import load_dataset


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

    # def handle_output(self, output_obj):
    #     raw_text = output_obj.text
    #     finish_reason = output_obj.finish_reason

    #     last_separator_index = raw_text.rfind(STEP_SEPARATOR)
    #     step_finished = False

    #     if finish_reason == "eos_token":
    #         step_finished = True
    #         return raw_text.replace(STEP_SEPARATOR, "\n"), step_finished

    #     if last_separator_index > 50: # Avoid getting stuck
    #         new_text = raw_text[:last_separator_index]
    #         new_text = new_text.replace(STEP_SEPARATOR, "\n")
    #     else:
    #         new_text = raw_text.replace(STEP_SEPARATOR, "\n")
    #         if finish_reason == "length" and VERBOSE:
    #             print(f"  Warning: Hit max_tokens without finding step separator. "
    #                   f"Generated {len(new_text)} chars.")
        
    #     return new_text, step_finished
    
    def handle_output(self, output_obj):
        """Extracts the new step from the generated text."""
        # vLLM returns the full prompt + generation or just generation depending on config.
        # usually output_obj.outputs[0].text is just the new generated tokens.
        raw_text = output_obj.outputs[0].text
        finish_reason = output_obj.outputs[0].finish_reason

        # Find the step separator to stop early if multiple steps were generated
        last_separator_index = raw_text.find(STEP_SEPARATOR)
        step_finished = False

        if finish_reason == "eos_token":
            step_finished = True
        
        # If we found a separator, cut off everything after it
        if last_separator_index != -1:
            raw_text = raw_text[:last_separator_index]
            step_finished = True
        
        # Clean up formatting
        clean_text = raw_text.replace(STEP_SEPARATOR, "\n")
        return clean_text, step_finished

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
        for output in outputs[0].outputs: 
            # Mock output object structure for handle_output compatibility
            class MockOutput:
                def __init__(self, o): self.outputs = [o]
            
            new_text, step_finished = self.handle_output(MockOutput(output))
            new_texts.append(new_text)
            finishes.append(step_finished)
            
        return new_texts, finishes
    
    def generate_batch_steps(self, requests, max_tokens=K_TOKENS):
        """
        Args:
            requests: List of dicts {'question': str, 'answer': str, 'retries': int, 'id': int}
        Returns:
            Dict mapping request 'id' -> (new_text, is_finished)
        """
        grouped_requests = defaultdict(list)
        for req in requests:
            grouped_requests[req['retries']].append(req)

        results_map = {}

        for retry_count, batch_reqs in grouped_requests.items():
            prompts = [self.build_input_context(r['question'], r['answer']) for r in batch_reqs]
            
            temp = min(1.0 + (0.5 * retry_count), 2.0)
            
            params = SamplingParams(
                temperature=temp,
                max_tokens=max_tokens,
                top_p=0.9,
                stop=[STEP_SEPARATOR] # Stop generation exactly at separator
            )

            outputs = self.llm.generate(prompts, params, use_tqdm=False)

            for req, output in zip(batch_reqs, outputs):
                new_text, finished = self.handle_output(output)
                results_map[req['id']] = (new_text, finished)

        return results_map
    
    

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
    
    def parallel_run(self, 
            prompt: str, 
            tau: float = TAU, 
            max_retries: int = MAX_BACKTRACKS, 
            max_branches: int = 8,
            expansion_factor: int = 2):
        
        print(f"\n--- New Parallel Run: {prompt[:40]}... ---")
        
        active_branches = [{
            'answer': "", 
            'retries': 0, 
            'score': 0.0,
            'finished': False
        }]
        
        completed_branches = []
        step_idx = 0

        while active_branches:
            step_idx += 1
            print(f"\n=== Step {step_idx} | Active: {len(active_branches)} ===")
            
            requests = []
            for i, branch in enumerate(active_branches):
                requests.append({
                    'id': i,
                    'question': prompt,
                    'answer': branch['answer'],
                    'retries': branch['retries']
                })

            step_results = self.gen.generate_batch_steps(requests)

            candidates_to_score = [] 
            branch_indices_map = []  
            
            # Temporary storage for the new text chunks to avoid decoding twice
            new_chunks_map = {} 

            for i, branch in enumerate(active_branches):
                new_chunk, is_step_finished = step_results[i]
                new_chunks_map[i] = (new_chunk, is_step_finished)
                                
                if not new_chunk.strip() and not is_step_finished:
                    full_candidate = branch['answer'] 

                else:
                    full_candidate = (branch['answer'] + "\n\n" + new_chunk) if branch['answer'] else new_chunk
                
                candidates_to_score.append((prompt, full_candidate))
                branch_indices_map.append(i)

            if candidates_to_score:
                batch_scores = self.prm.get_scores_batch(candidates_to_score)
            else:
                batch_scores = []

            # Create a lookup for easy access: branch_index -> score
            score_lookup = {idx: sc for idx, sc in zip(branch_indices_map, batch_scores)}

            next_active_branches = []
            
            for i, branch in enumerate(active_branches):
                new_chunk, is_step_finished = new_chunks_map[i]
                
                score = score_lookup.get(i, -1.0)
                
                # Construct full text again for the branch object
                full_text = (branch['answer'] + "\n\n" + new_chunk) if branch['answer'] else new_chunk
                is_final_answer = FINAL_ANSWER_PREFIX in new_chunk

                # Dynamic Threshold Calculation
                current_threshold = tau - (0.05 * branch['retries'])

                # Log outcome
                status_icon = "✅" if score >= current_threshold else "❌"
                print(f" Br {i}: {status_icon} (Score: {score:.3f} / {current_threshold:.3f})")

                if score >= current_threshold:
                    new_branch = {
                        'answer': full_text,
                        'retries': 0,
                        'score': score,
                        'finished': branch['finished'] or is_step_finished or is_final_answer
                    }
                    
                    if new_branch['finished']:
                        completed_branches.append(new_branch)
                    else:
                        next_active_branches.append(new_branch)
                else:
                    if branch['retries'] < max_retries:
                        clones_needed = expansion_factor if len(next_active_branches) < max_branches else 1
                        
                        for _ in range(clones_needed):
                            next_active_branches.append({
                                'answer': branch['answer'], 
                                'retries': branch['retries'] + 1,
                                'score': branch['score'],  
                                'finished': False
                            })

            # If we have too many active branches, keep the best ones
            if len(next_active_branches) > max_branches:
                # Sort primarily by Score (High is good), secondarily by Retries (Low is good)
                next_active_branches.sort(key=lambda x: (x['score'], -x['retries']), reverse=True)
                next_active_branches = next_active_branches[:max_branches]
                print(f" -> Pruned to top {max_branches} active branches.")

            active_branches = next_active_branches
            
            if step_idx >= 80: 
                print("\n[!] Max steps reached (30). Stopping search.")
                break

        # =========================================================
        # FINAL SELECTION LOGIC (Partial Fallback)
        # =========================================================
        
        # 1. If we have finished solutions, pick the best one.
        if completed_branches:
            best_sol = max(completed_branches, key=lambda x: x['score'])
            print(f"\n=> 🏆 Winner (Finished): {best_sol['score']:.3f}")
            return best_sol['answer']

        # 2. If NO finished solutions, pick the best active partial branch.
        if active_branches:
            best_partial = max(active_branches, key=lambda x: x['score'])
            print(f"\n=> ⚠️ Best PARTIAL Solution (Score: {best_partial['score']:.3f})")
            return best_partial['answer']
        
        print("\n=> 💀 All branches died.")
        return "FAILED TO GENERATE SOLUTION"
    
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
        
        final_answer = searcher.parallel_run(formatted_prompt)

        print(f"\n--- RESULT ---")
        print(f"GENERATED: {final_answer.strip()}")
        print(f"TRUTH:     {ground_truth}")

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
    
    # test_gsm8k(hybrid_searcher)
    final_response = hybrid_searcher.parallel_run(question)
    print(final_response)


if __name__ == "__main__":
    main()