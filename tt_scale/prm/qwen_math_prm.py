import torch
from ..base_classes import AbstractPRM
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch.nn.functional as F

# --- Configuration ---
DEFAULT_MODEL = "Qwen/Qwen2.5-Math-PRM-7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for Qwen models
    bnb_4bit_quant_type="nf4"
)


def make_step_rewards(logits, token_masks):
    """
    Extract step-wise rewards from model logits based on separator token positions.

    Args:
        logits: Model output logits [batch_size, seq_len, 2]
        token_masks: Boolean mask indicating positions of step separator tokens

    Returns:
        List of lists containing reward scores for each step
    """
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)

    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]
        # Extract positive probabilities (index 1) from non-zero positions
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


class QwenMathPRM(AbstractPRM):
    """
    Process Reward Model wrapper for Qwen2.5-Math-PRM-7B.

    This PRM uses a specialized Qwen model architecture that outputs step-wise
    rewards based on special separator tokens (<extra_0>), unlike logit-based
    models that check specific '+' or '-' tokens.
    """

    def __init__(self, model_name=DEFAULT_MODEL, device="cuda", quantization_config=bnb_config):
        print(f"Loading Qwen Math PRM [{model_name}]...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Use AutoModel instead of AutoModelForCausalLM for Qwen PRM
        self.model = AutoModel.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,  # Qwen models prefer bfloat16
            trust_remote_code=True,
        ).eval()

        self.device = device

        # Get the step separator token ID
        # Qwen Math PRM uses <extra_0> as the step separator
        self.step_sep_id = self.tokenizer.encode("<extra_0>")[0]

    def get_score(self, prompt_text: str, partial_response_text: str, step_separator: str = "\n\n") -> list[float]:
        """
        Calculates step-wise scores for the given reasoning chain.

        Args:
            prompt_text: The problem/question text
            partial_response_text: The step-by-step solution
            step_separator: Delimiter between steps (default: "\n\n")

        Returns:
            List of scores for each step (range: 0.0 to 1.0)
        """
        # Split response into steps
        steps = partial_response_text.split(step_separator)

        # Build messages with step separators
        # Format: [system] [user: prompt] [assistant: step1<extra_0>step2<extra_0>...]
        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
        ]

        # Apply chat template
        conversation_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize
        input_ids = self.tokenizer.encode(
            conversation_str,
            return_tensors="pt"
        ).to(self.device)

        # Run model inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, use_cache=False)
            logits = outputs[0]  # Shape: [batch_size, seq_len, 2]

            # Create mask for step separator tokens
            token_masks = (input_ids == self.step_sep_id)

            # Extract step rewards
            step_rewards = make_step_rewards(logits, token_masks)

            # Return the first (and only) batch element
            scores = step_rewards[0] if step_rewards else []

        # Debug print
        if scores:
            display_text = partial_response_text.replace("\n", " ")[-40:]
            # print(f"  [Qwen Math PRM] Scores: {scores} | ...{display_text}")

        return scores

    def get_scores_batch(self, questions: list[str], answers: list[list[str]]) -> list[list[list[float]]]:
        """
        Implementation of the abstract method for batch scoring.

        One question can map to multiple answers.

        Args:
            questions: List of problem statements
            answers: List of answer lists, where each answer is a list of solution strings

        Returns:
            Scores with dimension: num_questions x num_answers x num_steps
        """
        scores = []
        for q, ans in zip(questions, answers, strict=True):
            sub_scores = []
            for a in ans:
                sub_scores.append(self.get_score(q, a))
            scores.append(sub_scores)
        return scores


