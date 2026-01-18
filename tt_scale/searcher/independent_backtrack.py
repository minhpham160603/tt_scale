import logging
from tt_scale.config import Config
from tt_scale.base_classes import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class IndependentBacktrack(Searcher):
    def __init__(self, generator: AbstractGenerator, prm: AbstractPRM, config: Config):
        self.gen = generator
        self.prm = prm
        self.config = config

    def run(
        self,
        prompt: str,
        *args, **kwargs
    ) -> str:
        logger.debug(f"\n--- Parallel Run: {prompt[:50]}... ---")

        active_branches = [{"answer": "", "retries": 0, "score": 0.0, "finished": False}]
        completed_branches = []

        step_count = 0
        retries_count = 0
        backtrack_count = 0
        jump_count = 0
        total_tokens = 0  # Track total tokens generated

        while active_branches:
            step_count += 1
            if step_count > self.config.max_steps:
                logger.debug(" -> Reached maximum steps, stopping.")
                break

            logger.debug(f"\n=== Step {step_count} | Active: {len(active_branches)} ===")

            gen_results = {}

            
            groups = {}
            for idx, br in enumerate(active_branches):
                groups.setdefault(br["retries"], []).append(idx)

            for retry_level, idxs in groups.items():
                contexts = []
                kept_idxs = []

                for i in idxs:
                    br = active_branches[i]
                    ctx = self.gen.build_input_context(prompt, br["answer"])
                    if ctx is None:
                        continue
                    contexts.append(ctx)
                    kept_idxs.append(i)

                if not contexts:
                    continue

                batch_outs = self.gen.generate_batch_step(
                    contexts, retry_attempt=retry_level, M_EXPANSION=1
                )

                for i, seqs in zip(kept_idxs, batch_outs):
                    new_chunk, is_eos, token_count = seqs[0]  
                    new_chunk = new_chunk or ""
                    total_tokens += token_count  # Accumulate token count
                    gen_results[i] = (new_chunk, bool(is_eos))
                    logger.debug(f" Br {i}: {new_chunk}... | Is EOS: {is_eos} | Token Count: {token_count}")

            if not gen_results:
                break

            candidates = []          
            cand_branch_idx = []     
            cand_meta = {}           

            for i, br in enumerate(active_branches):
                if i not in gen_results:
                    continue
                new_chunk, is_step_finished = gen_results[i]

                if (not new_chunk.strip()) and is_step_finished:
                    full_text = br["answer"]
                    cand_meta[i] = (full_text, True)
                    continue

                if not new_chunk.strip() and not is_step_finished:
                    full_text = br["answer"]
                    cand_meta[i] = (full_text, False)
                    continue

                full_text = (br["answer"] + new_chunk) if br["answer"] else new_chunk
                is_finished = is_step_finished or (self.config.final_answer_prefix in new_chunk)
                cand_meta[i] = (full_text, is_finished)

                candidates.append(full_text + "\n\n")
                cand_branch_idx.append(i)

            score_lookup = {}
            if candidates:
                questions = [prompt]
                partial_answers = [candidates]
                scores_nested = self.prm.get_scores_batch(questions, partial_answers)

                for i, score_seq in zip(cand_branch_idx, scores_nested[0]):
                    if not score_seq:
                        score = 0.0
                    elif self.config.agg == "mean":
                        score = float(sum(score_seq) / len(score_seq))
                    elif self.config.agg == "mean_only_final":
                        is_finished = cand_meta.get(i, ("", False))[1]
                        score = float(sum(score_seq) / len(score_seq)) if is_finished else float(score_seq[-1])
                    else:  # "last"
                        score = float(score_seq[-1])
                    
                    score_lookup[i] = score


            next_active = []
            rejected_candidates = [] 
            any_retried_this_step = False

            for i, br in enumerate(active_branches):
                full_text, is_finished = cand_meta.get(i, (br["answer"], br["finished"]))
                score = score_lookup.get(i, -1.0)

                if is_finished and (full_text.strip() or br["answer"].strip()):
                    completed_branches.append(
                        {"answer": full_text, "retries": br["retries"], "score": max(br["score"], score), "finished": True}
                    )
                    continue

                threshold = self.config.tau - (0.05 * br["retries"])
                passed = score >= threshold

                status_icon = "✅" if passed else "❌"
                logger.debug(f" Br {i}: {status_icon} ({score:.2f}/{threshold:.2f})")

                if passed:
                    next_active.append(
                        {"answer": full_text, "retries": 0, "score": score, "finished": False}
                    )
                else:
                    if self.config.backtrack and br["retries"] < self.config.max_backtracks:
                        any_retried_this_step = True
                        retries_count += 1
                        clones = self.config.expansion_factor if len(next_active) < self.config.max_branches else 1
                        for _ in range(clones):
                            next_active.append(
                                {
                                    "answer": br["answer"], 
                                    "retries": br["retries"] + 1,
                                    "score": br["score"],
                                    "finished": False,
                                }
                            )
                    else:
                        rejected_candidates.append({
                            "original_idx": i,
                            "current_score": score,
                            "retries": br["retries"]
                        })

            if any_retried_this_step:
                backtrack_count += 1

            if len(next_active) == 0 and len(rejected_candidates) > 0:
                logger.debug(" -> ⚠️ All branches failed. Attempting rescue of best rejected candidate.")
                
                best_reject = max(rejected_candidates, key=lambda x: x['current_score'])
                
                
                parent_branch = active_branches[best_reject['original_idx']]
                
                next_active.append({
                    "answer": parent_branch["answer"],
                    "retries": parent_branch["retries"] + 1, 
                    "score": parent_branch["score"],
                    "finished": False
                })
                
                retries_count += 1
                backtrack_count += 1


            # --- 5. Pruning ---
            if len(next_active) > self.config.max_branches:
                next_active.sort(key=lambda x: (x["score"], -x["retries"]), reverse=True)
                next_active = next_active[:self.config.max_branches]
                logger.debug(f" -> Pruned to top {self.config.max_branches}")

            active_branches = next_active

            if len(completed_branches) >= self.config.max_finished_branches:
                break

        from tt_scale.utils import Stats
        stats = Stats(
            step_count=step_count,
            retries_count=retries_count,
            backtrack_count=backtrack_count,
            jump_count=jump_count,
            total_tokens=total_tokens,
        )

        if completed_branches:
            best = max(completed_branches, key=lambda x: x["score"])
            logger.debug(f"=> Winner: {best['score']:.3f}")
            return best["answer"], stats

        if active_branches:
            best = max(active_branches, key=lambda x: x["score"])
            return best["answer"], stats

        return "Failed.", stats
