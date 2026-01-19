from tt_scale.arg_parse import parse_args
from tt_scale.config import Config
from tt_scale.generator.vllm_generator import VLLMGenerator
from tt_scale.prm.qwen_math_prm import QwenMathPRM
from tt_scale.base_classes import *
from tt_scale.searcher.best_of_n import BestOfN
from tt_scale.searcher.collective_backtrack import CollectiveBacktrack
from tt_scale.searcher.independent_backtrack import IndependentBacktrack
from tt_scale.searcher.beam_search import BeamSearch
import time
import logging
from vllm import LLM
from tt_scale.utils import _append_summary_csv, test, Stats
from tqdm import tqdm

logger = logging.getLogger(__name__)

method_to_searcher = {
    "collective": CollectiveBacktrack,
    "independent": IndependentBacktrack,
    "beam_search": BeamSearch,
    "best_of_n": BestOfN,
}

def main():
    args, config = parse_args()

    logger.info("Initializing vLLM Engine...")
    start_ts = time.perf_counter()
    engine = LLM(
        model=config.model_name,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_mem_util,
        trust_remote_code=True,
        dtype=config.dtype,
        max_model_len=config.max_model_len,
        quantization=config.quantization,
    )

    

    datasets = config.datasets if config.datasets else [
        "MathArena/hmmt_nov_2025",
        "MathArena/aime_2025",
        "MathArena/cmimc_2025",
        "MathArena/brumo_2025",
        "MathArena/apex_2025",
        "MathArena/hmmt_feb_2025",
    ]
    gen = VLLMGenerator(engine, config=config)
    prm = QwenMathPRM()
    searcher = method_to_searcher[config.method](gen, prm, config=config)
    
    corrects, totals = 0, 0
    cumulative_stats = Stats()
    all_per_question_stats = []  # Store all per-question stats across all datasets
    overall_start_ts = time.perf_counter()
    for ds in tqdm(datasets, desc="Testing datasets"):
        logger.info(f"\n\n\n==== Testing on dataset: {ds} ====")
        ds_start_ts = time.perf_counter()
        correct, total, stats, per_question_stats = test(
            searcher,
            ds
        )
        ds_elapsed = time.perf_counter() - ds_start_ts
        corrects += correct
        totals += total
        # Handle both Stats object and list format for backward compatibility
        if isinstance(stats, Stats):
            cumulative_stats += stats
        else:
            cumulative_stats += Stats.from_list(stats)
        # Store per-question stats for this dataset
        all_per_question_stats.extend(per_question_stats)
        # print(f"==== Results for {ds}: {correct}/{total} correct ====")
    logger.info(f"\n\n\n==== Overall Results: {corrects}/{totals} correct ====")
    logger.info(f"Total Steps: {cumulative_stats.step_count}, Total Retries: {cumulative_stats.retries_count}, "
          f"Total Backtracks: {cumulative_stats.backtrack_count}, Total Jumps: {cumulative_stats.jump_count}")
    
    # Calculate and print average tokens per question
    avg_tokens = 0.0
    if all_per_question_stats:
        avg_tokens = sum(s.total_tokens for s in all_per_question_stats) / len(all_per_question_stats)
        total_tokens_all = sum(s.total_tokens for s in all_per_question_stats)
        logger.info(f"Total Tokens (all questions): {total_tokens_all}")
        logger.info(f"Average Tokens per Question: {avg_tokens:.2f}")
    else:
        logger.info(f"Total Tokens (all questions): {cumulative_stats.total_tokens}")
        avg_tokens = cumulative_stats.total_tokens / totals if totals > 0 else 0.0
    
    overall_elapsed = time.perf_counter() - overall_start_ts
    _append_summary_csv(
        config.summary_csv,
        {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_tag": config.run_tag or "",
            "model": config.model_name,
            "dataset": "__overall__",
            "tau": config.tau,
            "max_total_branches": config.max_total_branches,
            "keeping_branches": config.keeping_branches,
            "passing_minimum": config.passing_minimum,
            "k_tokens": config.k_tokens,
            "max_steps": config.max_steps,
            "max_model_len": config.max_model_len,
            "agg": config.agg,
            "backtrack": bool(config.backtrack),
            "num_samples": (config.num_samples if config.num_samples is not None else ""),
            "seed": config.seed,
            "correct": corrects,
            "total": totals,
            "steps": cumulative_stats.step_count,
            "retries": cumulative_stats.retries_count,
            "backtracks": cumulative_stats.backtrack_count,
            "jumps": cumulative_stats.jump_count,
            "total_tokens": cumulative_stats.total_tokens,
            "avg_tokens_per_question": round(avg_tokens, 2),
            "elapsed_s": round(overall_elapsed, 3),
        },
    )
    elapsed = time.perf_counter() - start_ts
    s_int = int(elapsed)
    h, rem = divmod(s_int, 3600)
    m, sec = divmod(rem, 60)
    ms = int((elapsed - s_int) * 1000)
    logger.info(f"total time: ({h:02d}:{m:02d}:{sec:02d}.{ms:03d})")
if __name__ == "__main__":
    main()