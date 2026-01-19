"""
Generate Best-of-N samples for PRM reliability analysis.

This script generates N samples for each problem in a dataset using a specified
generator model and saves the results as JSON for later analysis.
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Generate Best-of-N samples for PRM analysis")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Math-Instruct",
        help="Generator model name"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MathArena/aime_2025",
        help="Dataset name from HuggingFace"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of problems to sample (None for all)"
    )
    parser.add_argument(
        "--N",
        type=int,
        default=64,
        help="Number of solutions to generate per problem"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens per generation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="prm_analysis_data",
        help="Output directory for JSON files"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Data type for model"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Quantization method (e.g., 'awq', None for no quantization)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize vLLM engine
    print(f"Loading model: {args.model_name}")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype=args.dtype,
        quantization=args.quantization,
    )
    tokenizer = llm.get_tokenizer()
    
    # System prompt matching paper experiments
    system_prompt = (
        "Solve the following math problem efficiently and clearly:\n\n"
        "- For simple problems (2 steps or fewer):\n"
        "Provide a concise solution with minimal explanation.\n\n"
        "- For complex problems (3 steps or more):\n"
        "Use this step-by-step format:\n\n"
        "## Step 1: [Concise description]\n"
        "[Brief explanation and calculations]\n\n"
        "## Step 2: [Concise description]\n"
        "[Brief explanation and calculations]\n\n"
        "...\n\n"
        "Regardless of the approach, always conclude with:\n\n"
        "Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\n"
        "Where [answer] is just the final number or expression that solves the problem."
    )
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split=args.split)
    
    if args.num_samples is not None:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    
    print(f"Generating {args.N} solutions for {len(dataset)} problems...")
    
    # Prepare output
    results = []
    
    # Generate for each problem
    for idx, sample in enumerate(tqdm(dataset, desc="Generating solutions")):
        question = sample.get("question", sample.get("problem"))
        answer = sample.get("answer", sample.get("solution"))
        
        # Build chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            [messages],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )[0]
        
        # Generate N solutions
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            n=args.N,
            top_p=0.9,
        )
        
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        
        # Extract solutions
        solutions = []
        for output in outputs[0].outputs:
            solutions.append({
                "text": output.text,
                "num_tokens": len(output.token_ids) if hasattr(output, 'token_ids') else 0,
            })
        
        # Store result
        problem_result = {
            "index": idx,
            "question": question,
            "gold_answer": answer,
            "solutions": solutions,
        }
        results.append(problem_result)
    
    # Save to JSON
    dataset_name = args.dataset.replace("/", "_")
    model_name = args.model_name.replace("/", "_")
    output_file = output_dir / f"{dataset_name}_{model_name}_N{args.N}.json"
    
    print(f"Saving results to: {output_file}")
    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "model_name": args.model_name,
                "dataset": args.dataset,
                "split": args.split,
                "N": args.N,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "num_problems": len(results),
            },
            "results": results,
        }, f, indent=2)
    
    print(f"Done! Generated {len(results)} problems with {args.N} solutions each.")


if __name__ == "__main__":
    main()
