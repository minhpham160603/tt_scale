import random
from datasets import load_dataset
import re
from datetime import datetime
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .constants import *
import os 
from tqdm import tqdm
from .grading.grader import grade_answer

def get_datetime_string():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def extract_result(text):
    """
    Extracts the first number (int or float) following the FINAL_ANSWER_PREFIX.
    Ignores units/percentage signs/etc, just extracts the numeric value.
    Examples it handles:
        "#### 70.8%"  -> 70.8
        "#### 123 bottles"  -> 123.0
        "#### 75" -> 75.0
        "#### 90.6" -> 90.6
    """
    pattern = f"{FINAL_ANSWER_PREFIX}\\s*([-+]?[0-9]*\\.?[0-9]+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return text

class AbstractDatasetLoader(ABC):
    """Abstract base class for dataset loaders that normalize data to a common format."""
    
    def load_samples(self, num_samples: int, seed: int = SEED) -> List[Dict[str, str]]:
        """
        Load and normalize dataset samples.
        
        Args:
            num_samples: Number of samples to load
            seed: Random seed for sampling
            
        Returns:
            List of dictionaries with normalized format: {"question", "solution", "answer"}
        """
        dataset = self._load_raw_dataset()
        random.seed(seed)
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        raw_samples = [dataset[i] for i in indices]
        return [self.normalize_sample(sample) for sample in raw_samples]
    
    @abstractmethod
    def _load_raw_dataset(self):
        """Load the raw dataset. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def normalize_sample(self, raw_sample: Dict[str, Any]) -> Dict[str, str]:
        """
        Normalize a single raw sample to the standard format.
        
        Args:
            raw_sample: Raw sample from the dataset
            
        Returns:
            Dictionary with keys: "question", "solution", "answer"
        """
        pass

class MATH500DatasetLoader(AbstractDatasetLoader):
    """Dataset loader for HuggingFaceH4/MATH-500 dataset."""
    
    def __init__(self, dataset_path: str = "HuggingFaceH4/MATH-500"):
        self.dataset_path = dataset_path
    
    def _load_raw_dataset(self):
        return load_dataset(self.dataset_path, split="test")
    
    def normalize_sample(self, raw_sample: Dict[str, Any]) -> Dict[str, str]:
        """Normalize MATH-500 sample: 'problem' -> 'question'."""
        return {
            "question": raw_sample.get("problem", ""),
            "solution": raw_sample.get("solution", ""),
            "answer": raw_sample.get("answer", "")
        }

class GSM8KDatasetLoader(AbstractDatasetLoader):
    """Dataset loader for openai/gsm8k dataset."""
    
    def __init__(self, dataset_path: str = "openai/gsm8k", subset: str = "main"):
        self.dataset_path = dataset_path
        self.subset = subset
    
    def _load_raw_dataset(self):
        return load_dataset(self.dataset_path, self.subset, split="test")
    
    def normalize_sample(self, raw_sample: Dict[str, Any]) -> Dict[str, str]:
        """
        Normalize GSM8K sample: extract answer from last line with '### <result>'.
        GSM8K format: 'answer' field contains full solution ending with '### <result>'
        """
        answer_text = raw_sample.get("answer", "")
        
        # Find the last line starting with "###" (answer is typically at the end)
        lines = answer_text.split("\n")
        answer_line_idx = -1
        
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith(FINAL_ANSWER_PREFIX):
                answer_line_idx = i
                break
        
        if answer_line_idx >= 0:
            # Solution is everything before the answer line
            solution_lines = lines[:answer_line_idx]
            answer_line = lines[answer_line_idx].strip()
            
            # Extract just the content after "###" prefix
            # Extract only a number (int, float, fraction) after FINAL_ANSWER_PREFIX
            answer_match = re.search(rf"{FINAL_ANSWER_PREFIX}\s*([-+]?[\d.,/]+)", answer_line)
            if answer_match:
                answer = answer_match.group(1).strip()
            else:
                answer = answer_line.replace(FINAL_ANSWER_PREFIX, "").strip()
        else:
            # No "###" found, use entire answer as solution
            solution_lines = lines
            answer = ""
        
        solution = "\n".join(solution_lines).strip()
        
        return {
            "question": raw_sample.get("question", ""),
            "solution": solution,
            "answer": answer
        }

def get_datasets(dataset_path="HuggingFaceH4/MATH-500", num_samples=10, seed=SEED):
    """
    Convenience function to load datasets with automatic loader selection.
    
    Args:
        dataset_path: Path to the dataset (supports "HuggingFaceH4/MATH-500" or "openai/gsm8k")
        num_samples: Number of samples to load
        seed: Random seed for sampling
        
    Returns:
        List of normalized samples with format: {"question", "solution", "answer"}
    """
    if dataset_path == "HuggingFaceH4/MATH-500":
        loader = MATH500DatasetLoader(dataset_path)
    elif dataset_path == "openai/gsm8k":
        loader = GSM8KDatasetLoader(dataset_path)
    else:
        raise ValueError(f"Unknown dataset path: {dataset_path}. Supported: MATH-500, GSM8K")
    
    return loader.load_samples(num_samples, seed)

def test(dataset_path, searcher, num_samples=NUM_SAMPLES, seed=SEED):
    """Test searcher on dataset samples using normalized format."""
    samples = get_datasets(dataset_path, num_samples, seed)
    correct = 0
    logs = []

    for i, sample in tqdm(enumerate(samples), desc="Evaluating samples", total=len(samples)):
        print(f"\n\n=== Question: {sample['question']} ===")
        output_text, log = searcher.run(sample['question'])
        
        if not sample['answer']:
            print("Warning: No ground truth answer found.")
            continue
        
        # Extract answer from output - could be numeric or LaTeX expression
        extracted_answer = extract_result(output_text)
        
        # Use grade_answer to compare - it handles LaTeX, fractions, and numeric expressions
        # Try to find the answer in the output text (look for ### prefix or boxed expressions)
        # If extract_result found a number, use that; otherwise use the full output
        if extracted_answer and extracted_answer != output_text:
            # We extracted a numeric value, use it as the answer string
            given_answer = str(extracted_answer)
        else:
            # No clear numeric extraction, use the full output text
            # grade_answer will handle LaTeX expressions like \boxed{...}
            given_answer = output_text.strip()
        
        ground_truth = sample['answer'].strip()
        
        # Use grade_answer for robust comparison (handles LaTeX, fractions, etc.)
        is_correct = grade_answer(given_answer, ground_truth)
        correct += is_correct
        
        if not is_correct:
            separator = "\n"+"="*50+"\n"
            txt = separator.join([str(entry) for entry in log])
            logs.append(f"Sample {i+1}:\n{sample['question']}\nSolution: {sample['solution']}\n<====>\n{txt}")
            
        if VERBOSE:
            print(f"\n[Generated]: {output_text.strip()}")
            print(f"[Generated Answer]: {given_answer}")
            print(f"[Truth Solution]: {sample['solution']}")
            print(f"[Truth Answer]: {ground_truth}")
            print(f"[Correct]: {is_correct}")
        else:
            print(f"[Generated Answer]: {given_answer} | [Truth Answer]: {ground_truth} | [Correct]: {is_correct}")

    dataset_name = "GSM8K" if "gsm8k" in dataset_path.lower() else "MATH-500"
    print(f"\n\n=== {dataset_name} Results: {correct}/{num_samples} correct ===")
    os.makedirs(f"debug/{dataset_name}", exist_ok=True)
        
    with open(f"debug/{dataset_name}/logs_{get_datetime_string()}.txt", "w") as f:
        f.write("\n".join(logs))

def test_end_to_end():
    from vllm import LLM
    from .prm.logits_prm import LogitsPRM
    from .vllm_hybrid_engine import HybridSearcher, VLLMGenerator  # Assuming these exist in your codebase
    MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
    
    # Initialize required components
    prm = LogitsPRM()
    generator = VLLMGenerator(
        LLM(
            model=MODEL_NAME,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.6,
            trust_remote_code=True,
            dtype="float16",
            max_model_len=16384,
            quantization="bitsandbytes",
            load_format="bitsandbytes",
        )
    )
    searcher = HybridSearcher(generator, prm)
    test("HuggingFaceH4/MATH-500", searcher, 10, 199)

def test_dataset():
    samples = get_datasets("openai/gsm8k", 20, 199)
    for sample in samples:
        print(f"Truth Answer: {sample['answer']}")

def test_answer_extraction():
    print(extract_result("#### 70.8%"))
    print(extract_result("#### 123 bottles"))
    print(extract_result("#### 75 percent"))
    print(extract_result("#### 90.678"))

if __name__ == "__main__":
    test_end_to_end()