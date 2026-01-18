from tt_scale.base_classes import Searcher
from tt_scale.config import Config
from vllm import SamplingParams
from typing import Tuple, List
from dataclasses import dataclass
import copy
import logging
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class Beam:
    """Represents a single beam in the beam search."""
    prompt: str
    index: int
    current_text: str
    next_text: str
    stop_reason: str
    all_scores: List[float]  # PRM scores for each step
    pruned: bool = False
    completed: bool = False
    completion_tokens: int = 0
    history: List[str] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []


class BeamSearch(Searcher):
    """
    Simplified beam search searcher that processes one question at a time.
    Based on the search-and-learn beam search implementation.
    """
    
    def __init__(self, generator, prm, config: Config):
        super().__init__(generator, prm, config)
        # Access the underlying vLLM engine from VLLMGenerator
        self.llm = generator.llm
        self.tokenizer = generator.tokenizer
        
        # Beam search specific config (use existing config fields where possible)
        self.num_iterations = getattr(config, 'num_iterations', config.max_steps)
        self.beam_width = getattr(config, 'beam_width', 1)  # M in the paper
        self.n_beams = getattr(config, 'n_beams', config.keeping_branches)  # Total beams to maintain
        self.lookahead = getattr(config, 'lookahead', 0)  # Lookahead steps
        self.filter_duplicates = getattr(config, 'filter_duplicates', False)
        self.sort_completed = getattr(config, 'sort_completed', False)
        
        # Ensure n_beams is a multiple of beam_width
        if self.n_beams % self.beam_width != 0:
            self.n_beams = (self.n_beams // self.beam_width) * self.beam_width
            if self.n_beams == 0:
                self.n_beams = self.beam_width
    
    def _generate_step(self, full_context: str, lookahead_steps: int = 0) -> Tuple[str, str, int]:
        """
        Generate one step with optional lookahead.
        
        Returns:
            (next_text, stop_reason, token_count)
        """
        # For the first step, use temperature; for lookahead, use greedy (temp=0)
        temp = self.config.temperature if lookahead_steps == 0 else 0.0
        
        params = SamplingParams(
            temperature=temp,
            max_tokens=self.config.k_tokens,
            top_p=0.9,
            n=1,
            stop=[self.generator.stop_token],
        )
        
        outputs = self.llm.generate([full_context], params, use_tqdm=False)
        output = outputs[0].outputs[0]
        
        next_text = output.text
        stop_reason = output.finish_reason or "length"
        token_count = len(output.token_ids) if hasattr(output, 'token_ids') else 0
        
        # If we have lookahead, continue generating
        lookahead_text = next_text
        for _ in range(lookahead_steps):
            if stop_reason == "stop" and self.generator.stop_token not in lookahead_text:
                break  # EOS reached
            
            # Continue from where we left off
            continued_context = full_context + lookahead_text
            lookahead_params = SamplingParams(
                temperature=0.0,  # Greedy for lookahead
                max_tokens=self.config.k_tokens,
                top_p=0.9,
                n=1,
                stop=[self.generator.stop_token],
            )
            lookahead_outputs = self.llm.generate([continued_context], lookahead_params, use_tqdm=False)
            lookahead_output = lookahead_outputs[0].outputs[0]
            lookahead_text += lookahead_output.text
            token_count += len(lookahead_output.token_ids) if hasattr(lookahead_output, 'token_ids') else 0
            
            if lookahead_output.finish_reason == "stop" and self.generator.stop_token not in lookahead_output.text:
                stop_reason = "EOS"
                break
        
        return next_text, stop_reason, token_count
    
    def _aggregate_score(self, scores: List[float]) -> float:
        """Aggregate PRM scores using the configured strategy."""
        if not scores:
            return 0.0
        
        if self.config.agg == "mean":
            return float(np.mean(scores))
        elif self.config.agg == "mean_only_final":
            return float(np.mean(scores))
        else:  # "last" (default)
            return float(scores[-1]) if scores else 0.0
    
    def run(self, prompt: str, *args, **kwargs) -> Tuple[str, "Stats"]:
        """
        Run beam search for a single prompt.
        
        Args:
            prompt: The problem/question to solve
            
        Returns:
            Tuple[str, Stats]: (best_answer, stats)
        """
        from tt_scale.utils import Stats
        
        if self.config.verbose:
            logger.info(f"\n--- Beam Search: {prompt[:50]}... ---")
        
        total_tokens = 0
        step_count = 0
        
        # Initialize beams
        beams: List[Beam] = []
        for i in range(self.n_beams):
            beams.append(
                Beam(
                    prompt=prompt,
                    index=i,
                    current_text="",
                    next_text="",
                    stop_reason="",
                    all_scores=[],
                    pruned=False,
                    completed=False,
                    completion_tokens=0,
                    history=[],
                )
            )
        
        completed_beams: List[Beam] = []
        
        # Main beam search loop
        for iteration in range(self.num_iterations):
            # Get active (non-pruned, non-completed) beams
            active_beams = [b for b in beams if not b.pruned and not b.completed]
            
            if not active_beams:
                break  # All beams are done
            
            # Ensure we have enough beams (duplicate if needed)
            if len(active_beams) < self.n_beams:
                repeats = (self.n_beams // len(active_beams)) + 1
                extended_beams = [copy.deepcopy(b) for b in (active_beams * repeats)[:self.n_beams]]
                active_beams = extended_beams
            
            # Generate next step for all active beams
            lookahead = 0 if iteration == self.num_iterations - 1 else self.lookahead
            
            for beam in active_beams:
                # Build context for this beam
                full_context = self.generator.build_input_context(
                    prompt, 
                    partial_answer=beam.current_text
                )
                
                if full_context is None:
                    beam.completed = True
                    completed_beams.append(beam)
                    continue
                
                # Generate next step
                next_text, stop_reason, token_count = self._generate_step(full_context, lookahead)
                beam.next_text = next_text
                beam.stop_reason = stop_reason
                beam.completion_tokens += token_count
                total_tokens += token_count
                step_count += 1
                
                # Update beam state
                beam.current_text += next_text
                beam.history.append(next_text)
                
                # Check if beam is completed
                if stop_reason == "stop" and self.generator.stop_token not in next_text:
                    beam.completed = True
                    completed_beams.append(beam)
                elif next_text == "":
                    beam.completed = True
                    completed_beams.append(beam)
            
            # Score all active beams using PRM
            prompts_for_scoring = [beam.prompt for beam in active_beams]
            completions_for_scoring = [[beam.current_text] for beam in active_beams]
            
            # Get PRM scores: returns list[list[list[float]]] = num_questions x num_answers x num_steps
            scores_nested = self.prm.get_scores_batch(prompts_for_scoring, completions_for_scoring)
            
            # Update beam scores
            for idx, beam in enumerate(active_beams):
                if scores_nested and idx < len(scores_nested):
                    question_scores = scores_nested[idx]  # List of answer scores for this question
                    if question_scores and len(question_scores) > 0:
                        beam.all_scores = question_scores[0]  # Extract step scores for this beam (first answer)
                    else:
                        beam.all_scores = []
                else:
                    beam.all_scores = []
            
            # Filter out completed beams
            active_beams = [b for b in active_beams if not b.completed]
            
            if not active_beams:
                break  # All beams completed
            
            # Filter duplicates if enabled
            if self.filter_duplicates:
                unique_beams = {}
                for beam in active_beams:
                    if beam.current_text not in unique_beams:
                        unique_beams[beam.current_text] = beam
                active_beams = list(unique_beams.values())
            
            # Prune to top-k beams based on aggregated scores
            top_k = self.n_beams // self.beam_width
            if len(active_beams) > top_k:
                agg_scores = [self._aggregate_score(b.all_scores) for b in active_beams]
                top_indices = np.argsort(agg_scores)[-top_k:]
                
                # Mark non-top beams as pruned
                for idx, beam in enumerate(active_beams):
                    if idx not in top_indices:
                        beam.pruned = True
                
                # Keep only top beams
                active_beams = [active_beams[i] for i in top_indices]
            
            # Update beams list
            beams = active_beams + [b for b in beams if b.completed or b.pruned]
        
        # Select best beam from completed beams, or from active if none completed
        if completed_beams:
            if self.sort_completed:
                completed_beams = sorted(
                    completed_beams,
                    key=lambda b: self._aggregate_score(b.all_scores),
                    reverse=True
                )
            best_beam = completed_beams[0]
        else:
            # No completed beams, use best active beam
            active_beams = [b for b in beams if not b.pruned and not b.completed]
            if active_beams:
                active_beams = sorted(
                    active_beams,
                    key=lambda b: self._aggregate_score(b.all_scores),
                    reverse=True
                )
                best_beam = active_beams[0]
            else:
                # Fallback: return empty result
                stats = Stats(
                    step_count=step_count,
                    retries_count=0,
                    backtrack_count=0,
                    jump_count=0,
                    total_tokens=total_tokens
                )
                return "Failed: No valid beams.", stats
        
        if self.config.verbose:
            best_score = self._aggregate_score(best_beam.all_scores)
            logger.info(f"  -> Best beam (idx {best_beam.index}): score={best_score:.3f}, tokens={best_beam.completion_tokens}")
        
        # Return stats
        stats = Stats(
            step_count=step_count,
            retries_count=0,
            backtrack_count=0,
            jump_count=0,
            total_tokens=total_tokens
        )
        
        return best_beam.current_text, stats
