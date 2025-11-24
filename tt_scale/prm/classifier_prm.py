import torch
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from tt_scale.base_classes import AbstractPRM

class ClassifierPRM(AbstractPRM):
    SYS_PROMPT = f"""
    You are a professional judge to give score to the reasoning steps in the answers below on a scale of 1 to 10, given the question.
    If the reasoning is sound and correct, give a high score. Else, give a low score.
    """

    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = "right"  # Classification usually prefers right padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            num_labels=1 
        ).eval()

        
        self.model.config.pad_token_id = self.model.config.eos_token_id

    def get_score(self, question: str, answer: str) -> float:
        """
        Helper function to score a single Q/A pair.
        """
        scores = self.get_scores_batch([(question, answer)])
        return scores[0]

    def get_scores_batch(self, qa_pairs):
        """
        Efficiently processes a list of (question, answer) tuples using a single forward pass.
        """
        # 1. Prepare Prompts using Chat Template
        # This ensures the model sees the exact format it was trained on (User/Assistant)
        batch_texts = []
        for q, a in qa_pairs:
            chat = [
                {"role": "system", "content": self.SYS_PROMPT},
                {"role": "user", "content": f"Q: {q}\n\nA: {a}"},
            ]
            formatted_input = self.tokenizer.apply_chat_template(
                chat, 
                tokenize=False, 
                add_generation_prompt=False
            )
            batch_texts.append(formatted_input)

        # 2. Tokenize
        inputs = self.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        # 3. Forward Pass (No Generation!)
        with torch.no_grad():
            outputs = self.model(**inputs)
            raw_scores = outputs.logits.squeeze(-1) 
            scores = torch.sigmoid(raw_scores) 
            print(f"Raw Scores: {raw_scores}")

        return scores.float().cpu().tolist()

# --- Usage Example ---
def main():
    model_name = "Qwen/Qwen3-0.6B" 
    
    prm = ClassifierPRM(model_name)
    
    # 1. Test Batch
    print("\n--- Testing Batch ---")
    data_batch = [
        ("What is 2+2?", "The answer is 4."),
        ("What is 2+2?", "The answer is 5."),
    ]
    batch_scores = prm.get_scores_batch(data_batch)
    
    for (q, a), score in zip(data_batch, batch_scores):
        print(f"Q: {q}\nA: {a}\nScore: {score:.4f}\n")

if __name__ == "__main__":
    main()