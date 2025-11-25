from ..base_classes import AbstractGenerator
import torch 
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache
)
import torch.nn.functional as F

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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config=quantization_config, 
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
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
        )
        return self.tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(self.device)
    
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def generate_step(self, input_ids, past_key_values, max_new_tokens, temperature=0.7, stop_strings=None):
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                past_key_values=past_key_values,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                return_dict_in_generate=True,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                stop_strings=stop_strings or [STOP_STRING],
                tokenizer=self.tokenizer
            )
        
        full_sequence = outputs.sequences
        past_key_values = outputs.past_key_values
        
        input_len = input_ids.shape[1]
        new_tokens = full_sequence[0][input_len:]
        finished = self.tokenizer.eos_token_id in new_tokens

        return full_sequence, self._trim_cache(past_key_values), finished

    def _trim_cache(self, past_key_values):
        """
        Trims the KV cache to match the sequence length required for the next step.
        RETURNS A DYNAMICCACHE object to be consistent with notebook logic.
        """
        if past_key_values is None:
            return None
        
        # Handle the case where past_key_values is already a DynamicCache or a tuple
        # We iterate to extract layers safely
        new_keys = []
        new_vals = []

        # Generic iteration works for both Tuple of tuples AND DynamicCache
        for layer in past_key_values:
            key_state, value_state = layer
            
            # Safety check for dimensions
            if key_state.shape[2] <= 1:
                return None

            # Trim last token
            new_keys.append(key_state[:, :, :-1, :])
            new_vals.append(value_state[:, :, :-1, :])

        new_cache = DynamicCache()
        new_cache.key_cache = new_keys
        new_cache.value_cache = new_vals
        return new_cache

    def _pad_past_key_values(self, caches):
        """
        Pads a list of KV-caches to the same sequence length so they can be batched.
        Matches the notebook logic exactly.
        """
        if not caches or all(c is None for c in caches):
            return None

        valid_caches = [c for c in caches if c is not None]
        if not valid_caches:
            return None
        
        # Access internals of the first valid cache
        # If it's DynamicCache, use key_cache. If tuple, len is layers.
        first_valid = valid_caches[0]
        if hasattr(first_valid, "key_cache"):
            num_layers = len(first_valid.key_cache)
        else:
            num_layers = len(first_valid)

        batched_keys = []
        batched_vals = []

        for layer_idx in range(num_layers):
            layer_keys_list = []
            layer_vals_list = []
            lengths = []

            for c in caches:
                if c is None: 
                    continue
                
                if hasattr(c, "key_cache"):
                    k, v = c.key_cache[layer_idx], c.value_cache[layer_idx]
                else:
                    k, v = c[layer_idx]

                k_squeezed = k.squeeze(0) 
                v_squeezed = v.squeeze(0)
                layer_keys_list.append(k_squeezed)
                layer_vals_list.append(v_squeezed)
                lengths.append(k_squeezed.shape[1])

            if not lengths:
                continue

            max_len = max(lengths)
            padded_layer_keys = []
            padded_layer_vals = []

            for k_s, v_s in zip(layer_keys_list, layer_vals_list):
                cur_len = k_s.shape[1]
                pad_amt = max_len - cur_len
                if pad_amt > 0:
                    k_s = F.pad(k_s, (0, 0, pad_amt, 0)) 
                    v_s = F.pad(v_s, (0, 0, pad_amt, 0))
                padded_layer_keys.append(k_s)
                padded_layer_vals.append(v_s)

            batched_keys.append(torch.stack(padded_layer_keys, dim=0))
            batched_vals.append(torch.stack(padded_layer_vals, dim=0))

        new_cache = DynamicCache()
        new_cache.key_cache = batched_keys
        new_cache.value_cache = batched_vals
        return new_cache

    def generate_batch_step(self, input_ids_list, past_key_values_list, max_new_tokens, temperature=0.7):
        # 1. Reset if inconsistent
        if any(c is None for c in past_key_values_list):
             past_key_values_list = [None] * len(input_ids_list)

        # 2. Pad Inputs (Left Padding)
        max_len = max(t.size(1) for t in input_ids_list)
        padded_input_ids = []
        attention_masks = []

        for t in input_ids_list:
            seq_len = t.size(1)
            pad_len = max_len - seq_len
            
            pads = torch.full((1, pad_len), self.tokenizer.pad_token_id, device=self.device, dtype=torch.long)
            padded_t = torch.cat([pads, t], dim=1)
            
            # Mask: 0 for pad, 1 for real
            mask_t = torch.cat([
                torch.zeros((1, pad_len), device=self.device, dtype=torch.long), 
                torch.ones((1, seq_len), device=self.device, dtype=torch.long)
            ], dim=1)
            
            padded_input_ids.append(padded_t)
            attention_masks.append(mask_t)

        batch_input_ids = torch.cat(padded_input_ids, dim=0)
        batch_masks = torch.cat(attention_masks, dim=0)

        # 3. Pad Cache
        batch_cache = self._pad_past_key_values(past_key_values_list)

        # 4. Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=batch_input_ids,
                past_key_values=batch_cache,
                attention_mask=batch_masks,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_seqs = outputs.sequences
        new_batched_cache = outputs.past_key_values

        # 5. Debatch results and trim
        final_sequences = []
        final_caches = []
        
        if hasattr(new_batched_cache, "key_cache"):
            key_cache = new_batched_cache.key_cache
            value_cache = new_batched_cache.value_cache
        else:
            key_cache = [layer[0] for layer in new_batched_cache]
            value_cache = [layer[1] for layer in new_batched_cache]

        for i in range(len(input_ids_list)):
            final_sequences.append(generated_seqs[i:i+1])
            
            new_keys = []
            new_vals = []
            
            valid_branch = True
            for k, v in zip(key_cache, value_cache):
                # k is [batch, heads, seq, dim]
                if k.shape[2] <= 1:
                    valid_branch = False
                    break
                
                k_trimmed = k[i:i+1, :, :-1, :]
                v_trimmed = v[i:i+1, :, :-1, :]
                new_keys.append(k_trimmed)
                new_vals.append(v_trimmed)
            
            if valid_branch and new_keys:
                branch_cache = DynamicCache()
                branch_cache.key_cache = new_keys
                branch_cache.value_cache = new_vals
                final_caches.append(branch_cache)
            else:
                final_caches.append(None)

        return final_sequences, final_caches