import os
import json
import datetime
import uuid
from typing import List, Dict, Any, Optional

# Standard location for all experiment results
DEFAULT_RESULTS_DIR = "./results/experiments"

def get_timestamp() -> str:
    """Returns current time in a filename-safe format."""
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def ensure_directory(path: str):
    """Safely creates a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def calculate_metrics(results: List[Dict[str, Any]], duration_sec: float = 0.0) -> Dict[str, Any]:
    """
    Standardizes metric calculation based on a list of sample results.
    Expects each result dict to have an 'is_correct' boolean key.
    """
    total = len(results)
    if total == 0:
        return {"accuracy": 0.0, "total": 0, "correct": 0}

    correct_count = sum(1 for r in results if r.get("is_correct", False))
    accuracy = correct_count / total

    return {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_samples": total,
        "duration_seconds": duration_sec,
        "avg_time_per_sample": duration_sec / total if total > 0 else 0
    }

def save_experiment_results(
    experiment_name: str,
    config: Dict[str, Any],
    results: List[Dict[str, Any]],
    metrics: Optional[Dict[str, Any]] = None,
    base_dir: str = DEFAULT_RESULTS_DIR
) -> str:
    """
    Saves the full experiment state to a JSON file.

    Args:
        experiment_name: A short identifier (e.g., 'baseline_greedy', 'bon_k4').
        config: Dictionary of hyperparameters (model, temp, k, etc.).
        results: List of dictionaries, one per sample.
        metrics: Optional pre-calculated metrics. If None, calculates automatically.
        base_dir: Where to save the file.

    Returns:
        The file path where data was saved.
    """
    ensure_directory(base_dir)

    timestamp = get_timestamp()
    run_id = str(uuid.uuid4())[:8] # Short unique ID

    # 1. Finalize Metrics if not provided
    if metrics is None:
        metrics = calculate_metrics(results)

    # 2. Add Runtime Metadata to Config
    final_config = config.copy()
    final_config.update({
        "timestamp": timestamp,
        "run_id": run_id
    })

    # 3. Structure the final JSON output
    output_data = {
        "experiment_name": experiment_name,
        "metadata": final_config,
        "metrics": metrics,
        "samples": results
    }

    # 4. Construct Filename
    # e.g., baseline_greedy_2023-10-27_10-00-00_a1b2c3d4.json
    filename = f"{experiment_name}_{timestamp}_{run_id}.json"
    filepath = os.path.join(base_dir, filename)

    # 5. Write to disk
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"\n[ExperimentUtils] Results saved successfully to:\n >> {filepath}")
    return filepath
