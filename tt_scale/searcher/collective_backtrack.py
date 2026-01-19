import random
import torch
import re
from datasets import load_dataset
import os
import csv
import time
import logging
from vllm import LLM, SamplingParams
from tt_scale.prm.logits_prm import LogitsPRM
from tt_scale.prm.vllm_cot_prm import VLLMCoTPRM
from tt_scale.prm.qwen_math_prm import QwenMathPRM
import math
from typing import List, Optional, Tuple
from tt_scale.generator.vllm_generator import VLLMGenerator
from tt_scale.grader import extract_and_grade
import sys
from tt_scale.config import Config, default_config
from tt_scale.arg_parse import parse_args
from tt_scale.utils import _append_summary_csv
from tt_scale.base_classes import *

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class CollectiveBacktrack(Searcher):
    def __init__(self, generator, prm, config: Config):
        super().__init__(generator, prm, config)

    def run(
        self,
        prompt: str,
        *args, **kwargs
    ) -> str:
        logger.debug(f"\n--- Parallel Run: {prompt[:50]}... ---")
        
        # Initialize active branches
        active_branches = [
            {
                "score": 0.0,
                "average_sub_score": 0.0, # for sorting pruning when backtracking
                "checkpoint": [],
                "finished": False,
                "branch_steps": 0,
            } for _ in range(self.config.passing_minimum)
        ]

        
        finished_branches = [] # collected finished branches
        protected_branches = [] # protect passed branches when backtracking
        step_count = 0 # number of steps taken
        retries_count = 0 # total retries across all steps
        jump_count = 0 # total jumps across all steps
        backtrack_count = 0 # total backtracks(one backtrack can have multiple retries) across all steps
        retries = 1 # number of retries for current step
        jumping = False # whether to force a jump forward
        finished = False # whether overall generation is finished
        total_tokens = 0  # Track total tokens generated

        while active_branches : # and not finished
            
            step_count += 1
            if step_count > self.config.max_steps:
                logger.debug("  -> Reached maximum steps, stopping.")
                break
            
            logger.debug(f"--- Step {step_count} | Generating for {len(active_branches)} branches... ---")
            
            next_active_branches = []
            passing_branches = [] # to store branches that pass this step, and will be keeped while backtracking
            passing = 0

            # if jumping, temporarily set passing threshold to 0
            if jumping:
                passing_threshold = 0.0
                jumping = False
                retries = 1
            else:
                passing_threshold = self.config.tau

            contexts = []
            branch_indices = []
            # Determine how many branches to expand based on the number of active branches
            if self.config.backtrack:
                m_expansion = max(1, math.floor(self.config.max_total_branches/len(active_branches)))
                extra = self.config.max_total_branches - (m_expansion * len(active_branches))
            else:
                m_expansion = int(self.config.max_total_branches/self.config.keeping_branches)
                extra = 0

            if not self.config.always_expand and retries == 1:
                m_expansion = 1

            for i, branch in enumerate(active_branches):
                if branch["finished"]:
                    continue
                current_generated = branch["checkpoint"][-1] if branch["checkpoint"] else ""
                ctx = self.generator.build_input_context(prompt, current_generated)
                if ctx is None:
                    logger.debug(f"  Br {i}: Dropped (prompt budget exceeded)")
                    continue
                contexts.append(ctx)
                branch_indices.append(i)
            # Optional warmup: allocate a larger token budget on the first step to
            # help the model establish the solution trajectory.
            if self.config.warmup:
                if step_count == 1:
                    batch_outputs = self.generator.generate_batch_step(contexts, retry_attempt=retries, M_EXPANSION=m_expansion, k_tokens=2 * self.config.k_tokens)
                else:
                    batch_outputs = self.generator.generate_batch_step(contexts, retry_attempt=retries, M_EXPANSION=m_expansion, k_tokens=self.config.k_tokens)
            else:
                batch_outputs = self.generator.generate_batch_step(contexts, retry_attempt=retries, M_EXPANSION=m_expansion, k_tokens=self.config.k_tokens)
            for bi, seqs in zip(branch_indices, batch_outputs):
                branch = active_branches[bi]
                current_generated = branch["checkpoint"][-1] if branch["checkpoint"] else ""
                # Extract token counts and accumulate
                for new_chunk, _, token_count in seqs:
                    total_tokens += token_count
                candidates = [current_generated + new_chunk + "\n\n" for (new_chunk, _, _) in seqs]

                # qa_pairs = [(prompt, cand) for cand in candidates]
                questions = [prompt] 
                partial_answers = [candidates]
                scores = self.prm.get_scores_batch(questions, partial_answers)

                for j, ((new_chunk, is_eos, _), score_all) in enumerate(zip(seqs, scores[0])):
                    full_answer_candidate = current_generated + new_chunk
                    if self.config.agg == "last":
                        score = score_all[-1]
                    elif self.config.agg == "mean_step":
                        score = sum(score_all)/len(score_all)
                    elif self.config.agg == "mean_last":
                        score = score_all[-1]
                    elif self.config.agg == "ema":
                        score = 0.5 * branch["score"] + 0.5 * score_all[-1]
                    else: 
                        raise ValueError(f"Invalid aggregation method: {self.config.agg}")
                    branch["average_sub_score"] += score
                    
                    if score > passing_threshold:
                        logger.debug(f"  Br {j}: \033[92mPass ({score:.2f})\033[0m")
                        passing += 1
                        
                        if is_eos or (self.config.final_answer_prefix in new_chunk): 
                            if self.config.agg == "mean_step" or self.config.agg =="mean_last":
                                score = sum(score_all)/len(score_all)
                            finished_branches.append({
                                "text": full_answer_candidate,
                                "score": score,
                                "finished": True
                            })

                            # if len(finished_branches) >= MAX_FINISHED_BRANCHES:
                            #     print("  -> Reached max finished branches, stopping.")
                            #     finished = True
                        else:
                            passing_branches.append({
                                "score": score,
                                "checkpoint": [full_answer_candidate],
                                "finished": False,
                                "average_sub_score": 0.0,
                                "text": full_answer_candidate,
                            })
                    else:
                        logger.debug(f"  Br {j}: \033[91mFail ({score:.2f})\033[0m")
                        
            if len(finished_branches) >= self.config.N:
                logger.debug(f"  -> Reached max finished branches, stopping.")
                break

            # if enough passing branches, not backtracking, and prune to KEEPING_BRANCHES
            if passing + len(protected_branches) >= self.config.passing_minimum:
                logger.debug(f"  -> Sufficient passing branches ({passing}), pruning to {self.config.keeping_branches}.")
                passing_branches.extend(protected_branches)
                passing_branches.sort(key=lambda x: x["score"], reverse=True)
                active_branches = passing_branches[:self.config.keeping_branches]

                retries = 1  
                protected_branches = [] # to store passing branches for multiple backtracks

            # Otherwise, backtrack
            else:
                logger.debug(f"  -> Insufficient passing branches {passing} <= {self.config.passing_minimum}, backtracking.")
                # sort the parent branches based on the average score of children branches
                active_branches.sort(key=lambda x: x["average_sub_score"], reverse=True)
                # when backtracking, add passing_branches to protected_branches
                protected_branches.extend(passing_branches)
                # Eliminate low-score branches
                backtrack_num = self.config.keeping_branches - retries/(self.config.max_backtracks+1) * (self.config.keeping_branches)

                backtrack_candidates = active_branches[:min(math.ceil(backtrack_num), len(active_branches))]
                
                active_branches = []
                for b in backtrack_candidates:
                    active_branches.append({
                        "score": b["score"],
                        "checkpoint": b["checkpoint"],
                        "finished": b["finished"],
                        "average_sub_score": 0.0,
                    })
                    # print("remaining active branches:", len(active_branches))
                if retries == 1:
                    backtrack_count += 1
                retries += 1
                retries_count += 1
                if retries > self.config.max_backtracks:
                    logger.debug("  -> Reached maximum backtracks, forcing jump forward.")
                    jumping = True
                    retries = 1
                    jump_count += 1
            logger.debug(f"  -> Active branches: {len(active_branches)}, Passing branches: {len(passing_branches)}, Protected branches: {len(protected_branches)}")

                    
        from tt_scale.utils import Stats
        stats = Stats(
            step_count=step_count,
            retries_count=retries_count,
            backtrack_count=backtrack_count,
            jump_count=jump_count,
            total_tokens=total_tokens,
        )
            

        if not finished_branches:
            if active_branches:
                best = max(active_branches, key=lambda x: x["score"])
                if best["checkpoint"]:
                    return best["checkpoint"][-1], stats
                elif "text" in best:
                    return best["text"], stats
                else:
                    return "Failed.", stats
            return "Failed.", stats

        best = max(finished_branches, key=lambda x: x["score"])
        logger.debug(f"=> Winner: {best['score']:.3f}")
        return best["text"], stats

