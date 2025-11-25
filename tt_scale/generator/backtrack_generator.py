from ..base_classes import AbstractGenerator
import torch 
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer
)
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

STOP_STRING = "<END_STEP>"

SYS_PROMPT = f"""
You are a genius math solver. 
Solve the problem step-by-step to avoid mistake. 
After every logical step of reasoning, output the token {STOP_STRING}.
DO NOT output multiple steps at once.
Give the final answer, with ### prefix, only all steps are completed.
"""

class BacktrackGenerator(AbstractGenerator):
    def __init__(self, model_name, device=DEVICE, quantization_config=bnb_config):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config=quantization_config, 
            device_map="auto"
        )
        self.device = device
    
    def tokenize(self, text):
        chat = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": text}
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(
            chat, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False
        )
        return self.tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(self.device)
    
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def generate_step(
        self,
        input_ids,
        past_key_values,
        max_new_tokens,
        temperature: float = 0.7,
        stop_strings=None,
    ):
        # 1. Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                past_key_values=past_key_values,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                return_dict_in_generate=True,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                # If no stop_strings are passed, default to the internal STOP_STRING
                stop_strings=stop_strings or [STOP_STRING],
                tokenizer=self.tokenizer,
            )
        
        full_sequence = outputs.sequences
        past_key_values = outputs.past_key_values
        
        # 3. Extract new text to check for stop strings/EOS
        input_len = input_ids.shape[1]
        new_tokens = full_sequence[0][input_len:]

        # 4. Determine if finished
        finished = False
        if self.tokenizer.eos_token_id in new_tokens:
            finished = True

        print(
            "DEBUG Generated Tokens:",
            self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
        )

        return full_sequence, self._trim_cache(past_key_values), finished

    def _trim_cache(self, past_key_values):
        """
        Trims the KV cache to match the sequence length required for the next step.
        Creates a NEW cache object to ensure backtracking states remain immutable.
        """
        if past_key_values is None:
            return None
        
        new_kv = []
        for layer in past_key_values:
            key_state, value_state = layer
            trimmed_key = key_state[:, :, :-1, :]
            trimmed_value = value_state[:, :, :-1, :]
            new_kv.append((trimmed_key, trimmed_value))
        return tuple(new_kv)  # DynamicCache(new_kv)
