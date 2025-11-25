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

    def get_score(self, prompt_text: str, partial_response_text: str) -> float:
        """
        Calculates the score (probability of '+') for the given context.
        """
        chat = [
            {"role": "user", "content": prompt_text + partial_response_text},
            {"role": "assistant", "content": "+"}
        ]
        
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

        # Debug print
        display_text = partial_response_text.replace("\n", " ")[-40:]
        print(f"  [Verifier] Raw Score: {score:.3f} | ...{display_text}")
        
        return score
    
    def get_scores_batch(self, qa_pairs: list[tuple[str, str]]) -> list[float]:
            """
            Implementation of the abstract method.
            Currently uses a loop for safety because the specific logit index (-3)
            can be fragile with padding in batch mode.
            """
            scores = []
            for q, a in qa_pairs:
                scores.append(self.get_score(q, a))
            return scores