import re
import os
import csv
from typing import Optional
import random
from dataclasses import dataclass
from datasets import load_dataset
from tt_scale.grader import extract_and_grade
from tt_scale.config import Config
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@dataclass
class Stats:
    """Statistics object for tracking search performance."""
    step_count: int = 0
    retries_count: int = 0
    backtrack_count: int = 0
    jump_count: int = 0
    total_tokens: int = 0
    
    def __add__(self, other: "Stats") -> "Stats":
        """Add two Stats objects together."""
        return Stats(
            step_count=self.step_count + other.step_count,
            retries_count=self.retries_count + other.retries_count,
            backtrack_count=self.backtrack_count + other.backtrack_count,
            jump_count=self.jump_count + other.jump_count,
            total_tokens=self.total_tokens + other.total_tokens,
        )
    
    def to_list(self) -> list:
        """Convert to list format for backward compatibility."""
        return [self.step_count, self.retries_count, self.backtrack_count, self.jump_count, self.total_tokens]
    
    @classmethod
    def from_list(cls, stats_list: list) -> "Stats":
        """Create Stats from list format."""
        if len(stats_list) >= 5:
            return cls(
                step_count=stats_list[0],
                retries_count=stats_list[1],
                backtrack_count=stats_list[2],
                jump_count=stats_list[3],
                total_tokens=stats_list[4],
            )
        elif len(stats_list) == 4:
            # Backward compatibility: old format without total_tokens
            return cls(
                step_count=stats_list[0],
                retries_count=stats_list[1],
                backtrack_count=stats_list[2],
                jump_count=stats_list[3],
                total_tokens=0,
            )
        else:
            raise ValueError(f"Invalid stats list length: {len(stats_list)}")


def _append_summary_csv(path: str, row: dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = os.path.exists(path)
    fieldnames = list(row.keys())
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def test(searcher, ds_name):
    # Decide effective thresholds per run to avoid local/global confusion
    config = searcher.config
    if not config.backtrack:
        tau_local = -1.0  # disable backtracking threshold
        passing_minimum_local = 0
    else:
        tau_local = config.tau
        passing_minimum_local = config.passing_minimum
    dataset = load_dataset(ds_name, split=config.split)
    if config.num_samples is not None:
        dataset = dataset.select(range(config.num_samples))
    random.seed(config.seed)
    log_file = None
    if config.detail_log:
        # Prepare log file at project root to ensure discoverability
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(base_dir, "logs")
        ds_name_clean = ds_name.replace('/', '_')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir,
            f"{ds_name_clean}_{config.model_name.replace('/', '_')}_tau{tau_local}_M{config.max_total_branches}_K{config.keeping_branches}_{ds_name_clean}.csv",
        )
        if not os.path.exists(log_file):
            with open(log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "model_answer", "gold_answer", "is_correct", "total_tokens"])
        logger.debug(f"[Log] Writing to: {os.path.abspath(log_file)}")
    # Optionally sub-sample for quick sweeps
    indices = list(range(len(dataset)))
    if config.num_samples is not None and isinstance(config.num_samples, int) and config.num_samples > 0:
        if config.num_samples < len(indices):
            indices = random.sample(indices, config.num_samples)
    correct = 0
    total = 0
    cumulative_stats = Stats()
    per_question_stats = []  # Store stats per question for analysis
    for i, ds_i in enumerate(indices):
        sample = dataset[ds_i]
        question = sample.get("question", sample.get("problem"))
        logger.debug(f"\n\n=== Question: {question} ===")
        output_text, stats = searcher.run(
            question,
            backtrack=config.backtrack,
            passing_minimum=config.passing_minimum,
            tau=tau_local,
            agg=config.agg,
            warmup=config.warmup,
        )
        answer_text = sample.get("answer", sample.get("solution"))

        # Build proper messages: user then assistant (last must be assistant)
        messages = [
            {"role": "user", "content": str(question)},
            {"role": "assistant", "content": str(output_text)},
        ]
        # Approximate output token count via tokenizer
        tokenizer = searcher.gen.tokenizer
        output_tokens = len(tokenizer.encode(str(output_text)))
        competition_config = {"final_answer": True, "strict_parsing": False}
        logger.debug(f"messages: {messages[-1]['content']}")
        logger.debug(f"answer_text: {answer_text}")
        model_answer = None
        warning_value = None
        try:
            model_answer, is_correct, warning_value = extract_and_grade(
                messages, output_tokens, str(answer_text), competition_config, debug_info="MathArena"
            )
        except Exception as e:
            logger.debug(f"extract_and_grade error: {e}")
            # Fallback to numeric/symbolic comparison
            model_answer = output_text

        total += 1
        correct += 1 if is_correct else 0
        # Handle both Stats object and list format for backward compatibility
        if isinstance(stats, Stats):
            question_stats = stats
        else:
            question_stats = Stats.from_list(stats)
        
        cumulative_stats += question_stats
        per_question_stats.append(question_stats)  # Store per-question stats
        
        # Print stats per question
        logger.info(f"  Question {i+1}/{len(indices)}: Correct={is_correct}, Steps={question_stats.step_count}, "
              f"Retries={question_stats.retries_count}, Backtracks={question_stats.backtrack_count}, "
              f"Jumps={question_stats.jump_count}, Tokens={question_stats.total_tokens}")

        # Append structured log record
        if config.detail_log and log_file:
            try:
                with open(log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            int(ds_i),
                            str(model_answer),
                            str(answer_text),
                            bool(is_correct),
                            question_stats.total_tokens,  # Add token count per question
                        ]
                    )
            except Exception as le:
                logger.debug(f"log write error: {le}")

        if config.verbose: 
            logger.debug(f"[Generated Answer]: {model_answer} | [Truth Answer]: {sample.get('answer', sample.get('solution'))}")
            logger.debug(f"now correct: {correct} / {total}")

    logger.debug(f"\n\n=== {ds_name} Results: {correct}/{total} correct ===")
    # Calculate average tokens
    if per_question_stats:
        avg_tokens = sum(s.total_tokens for s in per_question_stats) / len(per_question_stats)
        logger.debug(f"Average tokens per question: {avg_tokens:.2f}")
    
    return correct, total, cumulative_stats, per_question_stats
