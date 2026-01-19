"""
Analyze PRM reliability by computing Pearson correlations between partial and final scores.

This script loads Best-of-N samples, scores them with different PRMs at various token
thresholds, and computes correlation coefficients to assess PRM reliability.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
import torch
from transformers import BitsAndBytesConfig
from typing import List, Dict, Any
import sys

# Add parent directory to path to import tt_scale modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tt_scale.prm.qwen_math_prm import QwenMathPRM
from tt_scale.grader import extract_and_grade


def truncate_text_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens."""
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)


def aggregate_scores(scores: List[float], method: str) -> float:
    """Aggregate step-wise scores using the specified method."""
    if not scores:
        return 0.0
    
    if method == "last":
        return float(scores[-1])
    elif method == "mean":
        return float(np.mean(scores))
    elif method == "min":
        return float(np.min(scores))
    elif method == "whole":
        # For 'whole', we treat all scores as a single aggregate
        # This is handled differently in the scoring function
        return float(np.mean(scores))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def score_with_prm(
    prm: QwenMathPRM,
    question: str,
    solution: str,
    token_threshold: int = None,
    aggregation: str = "mean",
) -> float:
    """
    Score a solution using PRM, optionally truncating to token_threshold.
    
    Args:
        prm: The PRM model
        question: The problem statement
        solution: The solution text
        token_threshold: Maximum tokens to consider (None for full solution)
        aggregation: How to aggregate scores ("mean", "min", "last", "whole")
    
    Returns:
        Aggregated score
    """
    # Truncate solution if needed
    if token_threshold is not None:
        solution = truncate_text_to_tokens(solution, prm.tokenizer, token_threshold)
    
    # Get step-wise scores
    try:
        scores = prm.get_score(question, solution)
    except Exception:
        print("Error scoring with PRM")
        return 0.0
    
    # Aggregate
    return aggregate_scores(scores, aggregation)


def evaluate_correctness(
    question: str,
    solution: str,
    gold_answer: str,
) -> bool:
    """
    Evaluate if a solution is correct using the MathArena grader.
    
    Args:
        question: The problem statement
        solution: The solution text
        gold_answer: The gold answer
    
    Returns:
        True if correct, False otherwise
    """
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": solution},
    ]
    
    competition_config = {
        "final_answer": True,
        "strict_parsing": False,
    }
    
    try:
        model_answer, is_correct, warning = extract_and_grade(
            messages,
            0,  # output_tokens not used for correctness check
            gold_answer,
            competition_config,
            debug_info="PRM_Analysis"
        )
        return bool(is_correct)
    except Exception:
        # If grading fails, assume incorrect
        return False


def analyze_prm_reliability(
    data_file: Path,
    prm_model_name: str,
    token_thresholds: List[int],
    aggregation_methods: List[str],
    quantization_config: Any = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Analyze PRM reliability across different token thresholds and aggregation methods.
    
    Returns a dictionary with correlation results.
    """
    # Load data
    print(f"Loading data from: {data_file}")
    with open(data_file, "r") as f:
        data = json.load(f)
    
    results_list = data["results"]
    print(f"Loaded {len(results_list)} problems")
    
    # Initialize PRM
    print(f"Loading PRM: {prm_model_name}")
    prm = QwenMathPRM(
        model_name=prm_model_name,
        device=device,
        quantization_config=quantization_config,
    )
    
    # Results storage
    correlation_results = {}
    
    # For each aggregation method
    for agg_method in aggregation_methods:
        print(f"\n=== Aggregation: {agg_method} ===")
        correlation_results[agg_method] = {}
        
        # Collect all solutions across all problems
        all_questions = []
        all_solutions = []
        all_gold_answers = []
        
        for problem in results_list:
            question = problem["question"]
            gold_answer = problem["gold_answer"]
            
            for sol in problem["solutions"]:
                all_questions.append(question)
                all_solutions.append(sol["text"])
                all_gold_answers.append(gold_answer)
        
        print(f"Total solutions to analyze: {len(all_solutions)}")
        
        # Evaluate correctness for all solutions
        print("Evaluating correctness...")
        correctness = []
        for q, s, g in tqdm(zip(all_questions, all_solutions, all_gold_answers), 
                            total=len(all_solutions), 
                            desc="Checking correctness"):
            correctness.append(evaluate_correctness(q, s, g))
        
        correctness_binary = np.array([1 if c else 0 for c in correctness])
        print(f"Correct solutions: {sum(correctness)} / {len(correctness)} ({100*sum(correctness)/len(correctness):.2f}%)")
        
        # Score at final (full solution)
        print("Scoring final (full) solutions...")
        final_scores = []
        for q, s in tqdm(zip(all_questions, all_solutions), 
                        total=len(all_solutions), 
                        desc="Scoring final"):
            score = score_with_prm(prm, q, s, token_threshold=None, aggregation=agg_method)
            final_scores.append(score)
        
        final_scores = np.array(final_scores)
        
        # For each token threshold, score and compute correlation
        for tau in token_thresholds:
            print(f"Analyzing threshold τ={tau}...")
            partial_scores = []
            
            for q, s in tqdm(zip(all_questions, all_solutions), 
                            total=len(all_solutions), 
                            desc=f"Scoring τ={tau}"):
                score = score_with_prm(prm, q, s, token_threshold=tau, aggregation=agg_method)
                partial_scores.append(score)
            
            partial_scores = np.array(partial_scores)
            
            # Compute correlations
            # 1. Partial vs Final score correlation
            try:
                corr_partial_final, p_val_pf = pearsonr(partial_scores, final_scores)
            except Exception:
                print("Error computing partial-final correlation")
                corr_partial_final, p_val_pf = 0.0, 1.0
            
            # 2. Partial score vs Correctness correlation
            try:
                corr_partial_correctness, p_val_pc = pearsonr(partial_scores, correctness_binary)
            except Exception:
                print("Error computing partial-correctness correlation")
                corr_partial_correctness, p_val_pc = 0.0, 1.0
            
            # 3. Final score vs Correctness correlation (for reference)
            try:
                corr_final_correctness, p_val_fc = pearsonr(final_scores, correctness_binary)
            except Exception:
                print("Error computing final-correctness correlation")
                corr_final_correctness, p_val_fc = 0.0, 1.0
            
            correlation_results[agg_method][tau] = {
                "partial_vs_final": {
                    "r": float(corr_partial_final),
                    "p": float(p_val_pf),
                },
                "partial_vs_correctness": {
                    "r": float(corr_partial_correctness),
                    "p": float(p_val_pc),
                },
                "final_vs_correctness": {
                    "r": float(corr_final_correctness),
                    "p": float(p_val_fc),
                },
            }
            
            print(f"  τ={tau}: Partial→Final r={corr_partial_final:.3f}, "
                  f"Partial→Correctness r={corr_partial_correctness:.3f}, "
                  f"Final→Correctness r={corr_final_correctness:.3f}")
    
    return correlation_results


def main():
    parser = argparse.ArgumentParser(description="Analyze PRM reliability via correlation analysis")
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to JSON file with Best-of-N samples"
    )
    parser.add_argument(
        "--prm_model",
        type=str,
        default="Qwen/Qwen2.5-Math-PRM-7B",
        help="PRM model name"
    )
    parser.add_argument(
        "--token_thresholds",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 1536, 2048],
        help="Token thresholds to analyze"
    )
    parser.add_argument(
        "--aggregation_methods",
        type=str,
        nargs="+",
        default=["mean", "min", "last", "whole"],
        help="Aggregation methods to test"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSON file for results (default: auto-generated)"
    )
    parser.add_argument(
        "--no_quantization",
        action="store_true",
        help="Disable 4-bit quantization"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Setup quantization
    if args.no_quantization:
        quantization_config = None
        quant_suffix = "no_quant"
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        quant_suffix = "4bit"
    
    # Run analysis
    correlation_results = analyze_prm_reliability(
        data_file=Path(args.data_file),
        prm_model_name=args.prm_model,
        token_thresholds=args.token_thresholds,
        aggregation_methods=args.aggregation_methods,
        quantization_config=quantization_config,
        device=args.device,
    )
    
    # Determine output file
    if args.output_file is None:
        data_file_stem = Path(args.data_file).stem
        prm_name = args.prm_model.replace("/", "_")
        output_file = Path(args.data_file).parent / f"{data_file_stem}_correlation_{prm_name}_{quant_suffix}.json"
    else:
        output_file = Path(args.output_file)
    
    # Save results
    print(f"\nSaving results to: {output_file}")
    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "data_file": str(args.data_file),
                "prm_model": args.prm_model,
                "token_thresholds": args.token_thresholds,
                "aggregation_methods": args.aggregation_methods,
                "quantization": quant_suffix,
            },
            "correlations": correlation_results,
        }, f, indent=2)
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY: Pearson Correlation (Partial → Correctness)")
    print("="*80)
    print(f"{'Threshold':<12}", end="")
    for agg in args.aggregation_methods:
        print(f"{agg:<15}", end="")
    print()
    print("-"*80)
    
    for tau in args.token_thresholds:
        print(f"τ={tau:<10}", end="")
        for agg in args.aggregation_methods:
            r = correlation_results[agg][tau]["partial_vs_correctness"]["r"]
            print(f"{r:>6.3f}         ", end="")
        print()
    
    print("="*80)
    print("\nDone!")


if __name__ == "__main__":
    main()
