import torch
import re
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

USE_VLLM = False
VLLM_AVAILABLE = False
if USE_VLLM:
    try:
        from vllm import LLM, SamplingParams
        VLLM_AVAILABLE = True
        print(">> vLLM detected. Using high-performance inference.")
    except ImportError:
        print(">> vLLM not found. Falling back to HuggingFace Transformers.")


default_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

class CoTPRM:
    def __init__(self, model_name: str, device_map="auto", quantization_config=default_bnb_config):
        print(f"Loading CoT Verifier [{model_name}]...")
        self.backend = "vllm" if VLLM_AVAILABLE else "hf"
        self.model_name = model_name
        
        if self.backend == "vllm":
            # Initialize vLLM Engine
            self.llm = LLM(
                model=model_name, 
                tensor_parallel_size=1, 
                trust_remote_code=True,
                dtype="float16",
                max_model_len=1592,
                quantization="awq" # Uncomment if loading an AWQ model
            )
            self.sampling_params = SamplingParams(
                temperature=0.0,    # Greedy for deterministic reasoning
                max_tokens=512,     # Allow space for "Thinking"
            )
            self.tokenizer = self.llm.get_tokenizer()
            
        else:
            # Initialize Hugging Face
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=True,
            ).eval()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _create_cot_prompt(self, question: str, answer: str) -> str:
        """
        Asks the model to think step-by-step before scoring.
        """
        return f"""
        Review the question and the provided answer step below.
        
        Task:
        1. Briefly analyze the logical correctness of the answer. Identify any errors.
        2. After your analysis, provide a score from 1 (worst) to 10 (best).
        3. End your response strictly with the format: "Score: <number>"
        
        Q: {question}
        A: {answer}
        """

    def _extract_score_from_cot(self, text: str) -> float:
        """
        Parses the LAST number appearing after 'Score:' to handle the chain of thought.
        """
        # Regex to find "Score: 8", "Score: 8.5", "Score: 10", etc.
        match = re.findall(r"Score:\s*(\d+(\.\d+)?)", text, re.IGNORECASE)
        
        if match:
            try:
                # Take the last match found (the final verdict)
                score = float(match[-1][0])
                return max(1.0, min(10.0, score))
            except ValueError:
                pass
        
        # print(f"Warning: Could not parse CoT score from: '{text[-50:]}...'") 
        return 0.0

    def get_scores_batch(self, qa_pairs: list) -> list:
        """
        Efficiently processes a list of (question, answer) tuples.
        """
        # 1. Prepare Prompts
        prompts = []
        for q, a in qa_pairs:
            raw_prompt = self._create_cot_prompt(q, a)
            chat = [{"role": "user", "content": raw_prompt}]
            
            # Apply chat template
            full_prompt = self.tokenizer.apply_chat_template(
                chat, 
                tokenize=False, 
                add_generation_prompt=True
            )
            prompts.append(full_prompt)

        generated_texts = []

        # 2. Inference (Dual Backend)
        if self.backend == "vllm":
            # vLLM handles batching natively
            outputs = self.llm.generate(prompts, self.sampling_params)
            for output in outputs:
                generated_texts.append(output.outputs[0].text)
        else:
            # Hugging Face Loop (Fallback)
            print("Running HF sequential inference (slow)...")
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                input_len = inputs.input_ids.shape[1]
                
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=512, # Match vLLM max tokens
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                generated_texts.append(self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True))

        # 3. Extract Scores from Reasoning
        scores = [self._extract_score_from_cot(text) for text in generated_texts]
        
        if generated_texts:
            print(f"\n[DEBUG Sample Reasoning]:\n{generated_texts[0][:200]}...\n")

        return scores

# Usage
def main():
    # Example setup
    verifier = CoTPRM("Qwen/Qwen3-0.6B")
    
    # Batch Data
    batch_data = [
        ("Solve for x: 2x + 5 = 15", "First, I subtract 5: 2x = 10. Then divide by 2: x = 5."),
        ("What is the capital of France?", "The capital is Berlin."),
        ("Write a python function to add two numbers.", "def add(a, b): return a * b") # Intentionally wrong code
    ]
    
    scores = verifier.get_scores_batch(batch_data)
    
    for (q, a), score in zip(batch_data, scores):
        print(f"Q: {q} | Score: {score}")

if __name__ == "__main__":
    main()