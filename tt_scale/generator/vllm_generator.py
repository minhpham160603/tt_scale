from tt_scale.config import Config, default_config
from vllm import SamplingParams

class VLLMGenerator:
    STOP_STRING = "<END_STEP>"
    
    def __init__(self, llm_engine, config: Config = None):
        self.llm = llm_engine
        self.tokenizer = self.llm.get_tokenizer()
        self.stop_token = self.STOP_STRING
        self.config = config if config is not None else default_config
        self.SYS_PROMPT = f"""You are a genius problem solver. 
    Solve the problem step-by-step to avoid mistakes.
    After **EVERY logical step** of reasoning, output the token {self.STOP_STRING}.
    If all steps are completed, return final answer with `{self.config.final_answer_prefix}` prefix, and Put your final answer within \\boxed{{}}.(for example: `{self.config.final_answer_prefix} \\boxed{{16}}` or `{self.config.final_answer_prefix} \\boxed{{90.6}}`)"""

    def build_input_context(self, question, partial_answer=""):
        """
        Build the chat prompt and left-truncate assistant partial_answer at token level
        to fit within the model length budget. We reserve 1 token for decoding to
        avoid vLLM pre-validation errors (prompt + at least 1 token > max length).
        """
        # 1) Build conversation prefix as tokens (System + User + assistant header)
        messages = [
            {"role": "system", "content": self.SYS_PROMPT},
            {"role": "user", "content": question},
        ]
        prefix_tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # 2) Encode assistant partial tokens
        pa_tokens = self.tokenizer.encode(partial_answer or "")

        # 3) Enforce budget: allow at most MAX_MODEL_LEN - 1 prompt tokens
        max_prompt_tokens = max(1, self.config.max_model_len - 1)
        total = len(prefix_tokens) + len(pa_tokens)
        if total > max_prompt_tokens:
            # keep_pa = max(0, max_prompt_tokens - len(prefix_tokens))
            # pa_tokens = pa_tokens[-keep_pa:] if keep_pa > 0 else []
            return None  # exceed model length
        merged_tokens = prefix_tokens + pa_tokens
        return self.tokenizer.decode(merged_tokens)

    def generate_step(self, full_context, retry_attempt=0, M_EXPANSION=1):
        """
        Generates the next step given the full context.
        """
        temp = self.config.temp_origin + (self.config.temp_step * retry_attempt)
        
        params = SamplingParams(
            temperature=temp,
            max_tokens=self.config.k_tokens,
            top_p=0.9,
            top_k=40, 
            n = M_EXPANSION,
            stop=[self.stop_token],
        )

        # vLLM automatically uses Prefix Caching here.
        # Since 'context_prefix' (System+User) is constant, it is cached.
        # Since 'partial_answer' grows, vLLM caches the shared prefix of the answer.
        outputs = self.llm.generate([full_context], params, use_tqdm=False)
        result = []
        for out in outputs[0].outputs:
            new_text = out.text
            finish_reason = out.finish_reason
            is_eos = (finish_reason == "stop" and self.stop_token not in new_text)
            # Return token count along with text and is_eos
            token_count = len(out.token_ids) if hasattr(out, 'token_ids') else 0
            result.append((new_text, is_eos, token_count))
        # new_text = outputs[0].outputs[0].text
        # finish_reason = outputs[0].outputs[0].finish_reason
        # is_eos = (finish_reason == "stop" and self.stop_token not in new_text)

            if self.config.debug:
                print(">>>>>>>>>>>>>>>>>>>>")
                print("GEN_STEP: ", new_text)
                print("<<<<<<<<<<<<<<<<<<<<")
                print("Finish Reason:", finish_reason, "| Is EOS:", is_eos)
        
        return result

    def generate_batch_step(self, full_contexts, retry_attempt=0, M_EXPANSION=1, k_tokens=None):
        if k_tokens is None:
            k_tokens = self.config.k_tokens
        temp = self.config.temp_origin + (self.config.temp_step * retry_attempt)
        params = SamplingParams(
            temperature=temp,
            max_tokens=k_tokens,
            top_p=0.9,
            top_k=40,
            n=M_EXPANSION, # number of expansions per prompt
            stop=[self.stop_token],
        )
        outputs = self.llm.generate(full_contexts, params, use_tqdm=False)
        
        batch_result = []
        for o in outputs:
            seqs = []
            for out in o.outputs:
                new_text = out.text
                is_eos = (out.finish_reason == "stop" and self.stop_token not in new_text)
                # Return token count along with text and is_eos
                token_count = len(out.token_ids) if hasattr(out, 'token_ids') else 0
                seqs.append((new_text, is_eos, token_count))
            batch_result.append(seqs)
        return batch_result
