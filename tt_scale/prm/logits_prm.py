import torch
from ..base_classes import AbstractPRM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# --- Configuration ---
DEFAULT_MODEL = "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

class LogitsPRM(AbstractPRM):
    """
    Wraps a Process Reward Model that scores based on the log-probability 
    of specific tokens (e.g., "+" vs "-").
    """
    def __init__(self, model_name=DEFAULT_MODEL, device="cuda", quantization_config=bnb_config):
        print(f"Loading Verifier [{model_name}]...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            dtype=torch.float16,
            trust_remote_code=True,
        ).eval()
        self.device = device
        
        self.plus_tag_id = self.tokenizer.encode("+")[-1]
        self.minus_tag_id = self.tokenizer.encode("-")[-1]
        self.candidate_tokens = [self.plus_tag_id, self.minus_tag_id]

    def get_score(self, prompt_text: str, partial_response_text: str, step_separator: str = "\n\n") -> float:
        """
        Calculates the score (probability of '+') for the given context.
        """
        chat = []

        ans = partial_response_text.split(step_separator)
        step_scores = []
        for i, a in enumerate(ans):
            if i == 0:
                text = prompt_text + a
            else:
                text = a
            chat.append({"role": "user", "content": text})
            chat.append({"role": "assistant", "content": "+"})
        
            formatted_input = self.tokenizer.apply_chat_template(
                chat, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            inputs = self.tokenizer.encode(
                formatted_input,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=inputs)
                logits = outputs.logits[:, -3, self.candidate_tokens]
                prob = logits.softmax(dim=-1)[:, 0] 
                score = prob.cpu().tolist()[0]
                step_scores.append(score)

        # Debug print
        display_text = partial_response_text.replace("\n", " ")[-40:]
        print(f"  [Verifier] Raw Score: {step_scores[-1]} | ...{display_text}")
        
        return step_scores
    
    def get_scores_batch(self, questions: list[str], answers: list[list[str]]) -> list[list[list[float]]]:
            """
            Implementation of the abstract method.
            Currently uses a loop for safety because the specific logit index (-3)
            can be fragile with padding in batch mode.
            One question can map to multiple answers.
            dim: num_questions x num_answers x num_steps
            """
            scores = []
            for q, ans in zip(questions, answers, strict=True):
                sub_scores = []
                for a in ans:
                    sub_scores.append(self.get_score(q, a))
                scores.append(sub_scores)
            return scores

if __name__ == "__main__":
    prm = LogitsPRM()
    math_qa = {
        "question": "Cecilia feeds her puppy 1 cup of dog food per day for the first 180 days and 2 cups per day for the rest of the first year. Each bag contains 110 cups. How many bags of food does she use in the first year?",
        "answer": [
            "A year has 365 days.",
            "The number of days after the first 180 days is 365 - 180 = 185.",
            "For the first 180 days, she uses 180 × 1 = 160 cups.",
            "For the next 185 days, she uses 185 × 2 = 320 cups.",
            "Total cups in the first year is 160 + 370 = 530.",
            "Each bag has 110 cups, so 530 ÷ 110 = 4.",
            "Final answer: 4 bags.",
        ]
    }

    score = prm.get_score(math_qa["question"], "\n\n".join(math_qa["answer"]))
    print(score)