from tt_scale.base_classes import Searcher
from tt_scale.config import Config
from vllm import SamplingParams
from typing import Tuple
import numpy as np


class BestOfN(Searcher):
    """
    Best-of-N searcher that generates N candidates and selects the best one based on PRM scores.
    """
    
    def __init__(self, generator, prm, config: Config):
        super().__init__(generator, prm, config)
        # Access the underlying vLLM engine from VLLMGenerator
        self.llm = generator.llm
        self.tokenizer = generator.tokenizer
    
    def run(self, prompt: str, *args, **kwargs) -> Tuple[str, "Stats"]:
        """
        Generates N candidates for the prompt, scores them, and returns the best one.
        
        Args:
            prompt: The problem/question to solve
            
        Returns:
            Tuple[str, Stats]: (best_candidate, stats) where stats is a Stats object
        """
        if self.config.verbose:
            print(f"\n--- Best-of-{self.config.N}: {prompt[:50]}... ---")
        
        from tt_scale.utils import Stats
        total_tokens = 0  # Track total tokens generated
        # Build input context (question only, no partial answer for best_of_n)
        full_context = self.generator.build_input_context(prompt, partial_answer="")
        if full_context is None:
            if self.config.verbose:
                print("  -> Context too long, skipping.")
            stats = Stats(step_count=0, retries_count=0, backtrack_count=0, jump_count=0, total_tokens=0)
            return "Failed: Context too long.", stats
        
        # Phase 1: Generate N candidates
        if self.config.verbose:
            print(f"  -> Generating {self.config.N} candidates...")
        
        gen_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            n=self.config.N,  # Generate N sequences per prompt
            top_p=0.9,
            stop=[self.generator.stop_token] if hasattr(self.generator, 'stop_token') else None,
        )
        
        outputs = self.llm.generate([full_context], gen_params, use_tqdm=False)
        
        # Extract candidate texts and track tokens
        candidates = []
        total_tokens = 0
        for out in outputs[0].outputs:
            candidate_text = out.text
            candidates.append(candidate_text)
            # Track token count
            if hasattr(out, 'token_ids'):
                total_tokens += len(out.token_ids)
        
        if not candidates:
            if self.config.verbose:
                print("  -> No candidates generated.")
            stats = Stats(step_count=1, retries_count=0, backtrack_count=0, jump_count=0, total_tokens=total_tokens)
            return "Failed: No candidates generated.", stats
        
        if self.config.verbose:
            print(f"  -> Generated {len(candidates)} candidates.")
        
        # Phase 2: Score all candidates using PRM
        if self.config.verbose:
            print(f"  -> Scoring {len(candidates)} candidates...")
        
        # Prepare for PRM scoring: questions and answers format
        questions = [prompt]
        partial_answers = [candidates]  # List of lists: one question -> multiple answers
        
        scores_nested = self.prm.get_scores_batch(questions, partial_answers)
        
        # Extract scores for each candidate
        # scores_nested is list[list[list[float]]]: [question][candidate][step_scores]
        candidate_scores = []
        if scores_nested and len(scores_nested) > 0:
            for candidate_idx in range(len(candidates)):
                score_seq = scores_nested[0][candidate_idx] if candidate_idx < len(scores_nested[0]) else []
                
                if not score_seq:
                    score = 0.0
                elif self.config.agg == "mean":
                    score = float(sum(score_seq) / len(score_seq))
                elif self.config.agg == "mean_only_final":
                    # For best_of_n, we assume the answer is complete, so use mean
                    score = float(sum(score_seq) / len(score_seq))
                else:  # "last" (default)
                    score = float(score_seq[-1])
                
                candidate_scores.append(score)
        else:
            # Fallback: assign zero scores if PRM fails
            candidate_scores = [0.0] * len(candidates)
        
        # Phase 3: Select best candidate
        best_idx = int(np.argmax(candidate_scores))
        best_candidate = candidates[best_idx]
        best_score = candidate_scores[best_idx]
        
        if self.config.verbose:
            print(f"  -> Best candidate (idx {best_idx}): score={best_score:.3f}")
            if self.config.debug:
                print(f"  -> All scores: {[f'{s:.3f}' for s in candidate_scores]}")
        
        # Return stats: For best_of_n: 1 step (one generation phase), no retries/backtracks/jumps
        stats = Stats(step_count=1, retries_count=0, backtrack_count=0, jump_count=0, total_tokens=total_tokens)
        return best_candidate, stats
