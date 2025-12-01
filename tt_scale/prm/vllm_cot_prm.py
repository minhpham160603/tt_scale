from vllm import SamplingParams
import re
from ..base_classes import AbstractPRM

class VLLMCoTPRM(AbstractPRM):
    def __init__(self, llm_engine):
        self.llm = llm_engine
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
            

    def _create_cot_prompt(self, question: str, answer: str) -> str:
        """
        Asks the model to think step-by-step before scoring.
        """
        return f"""
        Review the question and the provided answer step below.
        
        Task:
        1. Briefly analyze the logical correctness of the answer.
        2. After your analysis, provide a score between 1 - 10. 
        Give high score to the answer if the direction is correct, even if it is partially completed.
        3. End your response strictly with the format: "Score: <number>". Float score is acceptable.
        
        Q: {question}
        A: {answer}
        """

    def _extract_score_from_cot(self, text: str) -> float:
        """
        Parses the LAST number appearing after 'Score:' to handle the chain of thought.
        """
        # Regex to find "Score: 8", "Score: 8.5", "Score: 10", etc.
        # print(">>>>>>>>>")
        # print("DEBUG CoTPRM:", text)
        # print("<<<<<<<<<")
        match = re.findall(r"Score:\s*(\d+(\.\d+)?)", text, re.IGNORECASE)
        
        if match:
            try:
                # Take the last match found (the final verdict)
                score = float(match[-1][0])
                return max(1.0, min(10.0, score)) / 10.
            except ValueError:
                pass
        
        # print(f"Warning: Could not parse CoT score from: '{text[-50:]}...'") 
        return 0.0
    
    def get_score(self, question, partial_answer):
        return self.get_scores_batch([(question, partial_answer)])[0]

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
                add_generation_prompt=True,
                enable_thinking=False
            )
            prompts.append(full_prompt)

        scores = []
        outputs = self.llm.generate(prompts, self.sampling_params)
        for output in outputs:
            scores.append(self._extract_score_from_cot(output.outputs[0].text))
        return scores

# Usage
def main():
    from vllm import LLM
    MODEL_NAME = "Qwen/Qwen3-4B"
    # Example setup
    engine = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=16384,
    )
    verifier = VLLMCoTPRM(engine)

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