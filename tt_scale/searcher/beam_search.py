"""
Adapt from https://github.com/huggingface/search-and-learn/blob/main/src/sal/search/beam_search.py
"""


from tt_scale.base_classes import Searcher
from tt_scale.config import Config
from typing import Tuple, List
from dataclasses import dataclass
import copy
import logging
import numpy as np
from tt_scale.utils import Stats
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
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
        # Beam search specific config (use existing config fields where possible)
        self.num_iterations = config.max_steps
        self.beam_width = config.keeping_branches
        self.n_beams = config.max_total_branches
        self.filter_duplicates = False
        self.sort_completed = False
    
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
    
    def run(self, prompt: str, *args, **kwargs) -> Tuple[str, Stats]:
        """
        Run beam search for a single prompt.
        
        Args:
            prompt: The problem/question to solve
            
        Returns:
            Tuple[str, Stats]: (best_answer, stats)
        """
        if self.config.verbose:
            logger.info(f"\n--- Beam Search: {prompt[:50]}... ---")
        
        total_tokens = 0
        step_count = 0
        
        # Initialize beams (start with a single empty beam)
        beams: List[Beam] = [
            Beam(
                prompt=prompt,
                index=0,
                current_text="",
                next_text="",
                stop_reason="",
                all_scores=[],
                pruned=False,
                completed=False,
                completion_tokens=0,
                history=[],
            )
            for _ in range(self.n_beams)
        ]
        
        completed_beams: List[Beam] = []
        
        # Main beam search loop
        for iteration in range(self.num_iterations):
            # Get active (non-pruned, non-completed) beams
            active_beams = [b for b in beams if not b.pruned and not b.completed]
            
            if not active_beams:
                break  # All beams are done

            # Build contexts for all active beams (like CollectiveBacktrack)
            contexts: List[str] = []
            live_beams: List[Beam] = []
            for beam in active_beams:
                ctx = self.generator.build_input_context(prompt, beam.current_text)
                if ctx is None:
                    beam.completed = True
                    completed_beams.append(beam)
                    continue
                contexts.append(ctx)
                live_beams.append(beam)

            if not live_beams:
                break

            # Expand each active beam into `beam_width` candidates using the generator directly.
            # This mirrors the pattern in CollectiveBacktrack: generate, then score expansions.
            batch_outputs = self.generator.generate_batch_step(
                contexts,
                retry_attempt=0,
                M_EXPANSION=(self.config.N // len(active_beams))+1,
                k_tokens=self.config.k_tokens,
            )

            expanded_beams: List[Beam] = []
            for parent_idx, (parent_beam, seqs) in enumerate(zip(live_beams, batch_outputs)):
                for j, (new_chunk, is_eos, token_count) in enumerate(seqs):
                    child = copy.deepcopy(parent_beam)
                    child.index = parent_beam.index * self.beam_width + j
                    child.next_text = new_chunk
                    child.stop_reason = "eos" if is_eos else ""
                    child.current_text = (parent_beam.current_text or "") + (new_chunk or "")
                    child.history = (parent_beam.history or []) + [new_chunk or ""]
                    child.completion_tokens = parent_beam.completion_tokens + token_count

                    total_tokens += token_count
                    step_count += 1

                    if is_eos or (new_chunk and self.config.final_answer_prefix in new_chunk) or (new_chunk == ""):
                        child.completed = True
                        completed_beams.append(child)
                    expanded_beams.append(child)

            if not expanded_beams:
                break

            # Score expanded beams using PRM
            questions = [prompt]
            partial_answers = [[b.current_text for b in expanded_beams]]
            scores_nested = self.prm.get_scores_batch(questions, partial_answers)

            # scores_nested: [ [ [step_scores...], [step_scores...], ... ] ]
            step_scores_list = scores_nested[0] if scores_nested and len(scores_nested) > 0 else []
            for b, step_scores in zip(expanded_beams, step_scores_list):
                b.all_scores = step_scores or []

            # Only keep non-completed for next iteration
            next_active = [b for b in expanded_beams if not b.completed]

            logger.debug(f"Active beams: {len(active_beams)}, Next active beams: {len(next_active)}, pruned to {self.beam_width}")
            # Prune to top-n_beams based on aggregated score
            if len(next_active) > self.beam_width:
                agg_scores = [self._aggregate_score(b.all_scores) for b in next_active]
                top_indices = np.argsort(agg_scores)[-self.beam_width:]
                next_active = [next_active[i] for i in top_indices]


            beams = next_active
        
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
            active_beams = [b for b in beams if not b.completed]
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
