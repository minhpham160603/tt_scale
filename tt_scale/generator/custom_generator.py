import torch
import json
from typing import List, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# 1. Robust Import for vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
    print(">> vLLM detected. Using high-performance inference.")
except ImportError:
    VLLM_AVAILABLE = False
    print(">> vLLM not found. Falling back to HuggingFace Transformers.")

# Default config for HF fallback
default_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

class Generator:
    def __init__(self, 
                 model_name: str, 
                 device_map="auto", 
                 quantization_config=default_bnb_config,
                 gpu_memory_utilization: float = 0.5):
        """
        Initializes the Generator Model.
        
        Args:
            gpu_memory_utilization (float): Fraction of GPU memory to use for vLLM. 
                                            Set to < 1.0 (e.g. 0.6) if you plan to load 
                                            a Verifier model on the same GPU.
        """
        self.model_name = model_name
        self.backend = "vllm" if VLLM_AVAILABLE else "hf"
        
        print(f"Loading Generator [{model_name}] via [{self.backend}]...")

        if self.backend == "vllm":
            # Initialize vLLM Engine
            # CRITICAL: gpu_memory_utilization allows running Generator + Verifier on one GPU
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=1,
                trust_remote_code=True,
                dtype="float16",
                gpu_memory_utilization=gpu_memory_utilization, 
                max_model_len=8192,
                enforce_eager=True
                # quantization="awq" # Uncomment if using AWQ models
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

    def generate_batch(self, 
                       prompts: List[str], 
                       temperature: float = 0.7, 
                       max_new_tokens: int = 1024,
                       n: int = 1) -> List[List[str]]:
        """
        Generates responses for a batch of prompts.
        
        Args:
            n (int): Number of sequences to return per prompt (useful for Beam Search / DVTS).
        
        Returns:
            List[List[str]]: A list where each element is a list of 'n' responses for that prompt.
        """
        
        # 1. Format Prompts with Chat Template
        formatted_prompts = []
        for prompt in prompts:
            chat = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                chat, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False
            )
            formatted_prompts.append(formatted_prompt)

        results = []

        # 2. Inference
        if self.backend == "vllm":
            # Set sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_new_tokens,
                n=n,  # Number of output sequences to return for the given prompt
                stop_token_ids=[self.tokenizer.eos_token_id]
            )
            
            # vLLM handles batching automatically
            outputs = self.llm.generate(formatted_prompts, sampling_params)
            
            # Extract text
            for output in outputs:
                # vLLM returns 'n' outputs per prompt
                batch_responses = [o.text for o in output.outputs]
                results.append(batch_responses)

        else:
            # Hugging Face Fallback (Sequential Loop to prevent padding complexity)
            print("Running HF sequential inference (slower)...")
            for prompt_text in formatted_prompts:
                inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
                input_len = inputs.input_ids.shape[1]
                
                # Handling 'n' outputs in HF requires num_return_sequences
                # and usually do_sample=True if n > 1
                do_sample = temperature > 0
                
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else None,
                        num_return_sequences=n,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode all 'n' sequences for this prompt
                batch_responses = []
                for i in range(n):
                    generated_text = self.tokenizer.decode(
                        output_ids[i][input_len:], 
                        skip_special_tokens=True
                    )
                    batch_responses.append(generated_text)
                
                results.append(batch_responses)

        return results

# --- Usage Example ---
def main():
    # Example: Loading Qwen-1.5B 
    # NOTE: Set gpu_memory_utilization to 0.4 or 0.5 if you plan to load the PRM later!
    generator = Generator(
        model_name="Qwen/Qwen3-0.6B", 
        gpu_memory_utilization=0.6
    )

    questions = [
        "Explain quantum entanglement like I'm 5.",
        "Write a python function to merge two sorted lists.",
    ]

    # Generate 1 response per question
    print("--- Generating Single Responses ---")
    responses = generator.generate_batch(questions, temperature=0.7, n=1)
    
    for q, r_list in zip(questions, responses):
        print(f"\nQ: {q}")
        print(f"A: {r_list[0][:200]}...") # Print first 200 chars

    # Generate 3 diverse responses per question (Useful for DVTS/Beam Search)
    print("\n--- Generating Diverse Candidates (n=3) ---")
    responses_diverse = generator.generate_batch(questions, temperature=0.9, n=3)
    
    for q, r_list in zip(questions, responses_diverse):
        print(f"\nQ: {q}")
        for i, r in enumerate(r_list):
            print(f"  Candidate {i+1}: {r[:50]}...")

if __name__ == "__main__":
    main()