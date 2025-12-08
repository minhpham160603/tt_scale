import torch
import re
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from ..base_classes import AbstractPRM

class CustomPRM(AbstractPRM):
    def __init__(self, model_name: str):
        print(f"Loading Verifier [{model_name}] via Hugging Face...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    def _create_prompt(self, question, answer):
        return f"""
        Give score to the reasoning steps in the answers below on a scale of 1 to 10, given the question.
        If the reasoning is sound and correct, give a high score. Else, give a low score.
        Output ONLY the number, floating point numbers are fine.
        
        Q: {question}
        A: {answer}
        """

    def _extract_score(self, text):
        match = re.search(r"(\d+(\.\d+)?)", text)
        if match:
            try:
                score = float(match.group(1))
                return max(1.0, min(10.0, score)) / 10.
            except ValueError:
                pass
        return 0.0

    def get_score(self, question: str, answer: str) -> float:
        """
        Helper function to score a single Q/A pair.
        """
        scores = self.get_scores_batch([(question, answer)])
        return scores[0]

    def get_scores_batch(self, qa_pairs):
        """
        Efficiently processes a list of (question, answer) tuples using HF batching.
        """
        batch_prompts = []
        for q, a in qa_pairs:
            raw_prompt = self._create_prompt(q, a)
            chat = [{"role": "user", "content": raw_prompt}]
            full_prompt = self.tokenizer.apply_chat_template(
                chat, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False
            )
            batch_prompts.append(full_prompt)
        inputs = self.tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False, # Greedy for deterministic scoring
                pad_token_id=self.tokenizer.pad_token_id
            )
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        scores = [self._extract_score(text) for text in decoded_texts]
        return scores

class VLLMPRM:
    def __init__(self, llm_engine):
        from vllm import SamplingParams
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

# --- Usage Example ---
def main():
    model_name = "Qwen/Qwen3-4B" 
    
    prm = CustomPRM(model_name)
    
    # 1. Test Batch
    print("\n--- Testing Batch ---")
    data_batch = [
        ("What is 2+2?", "The answer is 4."),
        ("What is the capital of France?", "Paris is the capital."),
        ("Solve x^2 = 9", "x is 3 or -3.")
    ]
    batch_scores = prm.get_scores_batch(data_batch)
    for (q, a), score in zip(data_batch, batch_scores):
        print(f"Q: {q[:15]}... | Score: {score}")

    # 2. Test Single
    print("\n--- Testing Single Function ---")
    q = "Write a python function to add two numbers."
    a = "def add(a, b): return a - b" # Intentionally wrong
    single_score = prm.get_score(q, a)
    print(f"Q: {q}\nA: {a}\nScore: {single_score}")

if __name__ == "__main__":
    main()