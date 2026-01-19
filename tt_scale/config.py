from dataclasses import dataclass, field, fields
from typing import Optional
from typing import Literal

try:
    import yaml
except ImportError:
    yaml = None

@dataclass
class Config:
    """Configuration object for the hybrid searcher."""
    # Model configuration
    model_name: str = "Qwen/Qwen2.5-3B-Instruct-AWQ"
    method: Literal["collective", "independent", "beam_search", "best_of_n"] = "collective"
    
    # Search parameters
    tau: float = 0.6  # threshold score to accept a step
    max_backtracks: int = 3  # maximum backtracks before forcing a step forward
    num_samples: Optional[int] = None  # optional subsample size; None means full dataset
    final_answer_prefix: str = "<FINAL>"
    quantization: Optional[str] = None
    
    # Debug and logging
    debug: bool = False
    verbose: bool = False
    epsilon: float = 1e-3
    seed: int = 79
    detail_log: bool = True  # Whether to write per-dataset/per-sample CSV logs under tt_scale/logs/
    
    # Temperature configuration
    temp_origin: float = 0.7  # initial temperature
    temp_step: float = 0.1  # increase per retry
    
    # Parallel hybrid search
    max_total_branches: int = 9  # maximum total branches at each step
    passing_minimum: int = 3  # minimum passing branches to avoid backtracking
    keeping_branches: int = 3  # branches to keep when pruning
    max_model_len: int = 3072  # maximum model context length
    max_steps: int = 30
    k_tokens: int = 256  # maximum tokens to generate per step
    
    # Best of N parameters
    N: int = 4  # number of candidates to generate for best_of_n
    temperature: float = 0.8  # temperature for candidate generation in best_of_n
    max_tokens: int = 2048  # maximum tokens per candidate in best_of_n
    agg: str = "last"  # aggregation method for PRM scores: "last", "mean", "mean_only_final"
    backtrack: bool = True  # whether to enable backtracking
    expansion_factor: int = 2  # expansion factor for backtracking
    max_finished_branches: int = 2  # maximum finished branches to stop
    max_branches: int = 6  # maximum branches for independent backtrack
    warmup: bool = False  # enable/disable first-step token warmup (uses 2x k-tokens on step 1)
    always_expand: bool = False  # whether to always expand all branches
    # Runtime-only settings (not typically in YAML, set via CLI)
    # These are kept separate as they're infrastructure-specific
    gpu_mem_util: float = 0.5  # GPU memory utilization for vLLM
    tensor_parallel_size: int = 1  # Tensor parallel size for vLLM
    dtype: str = "float16"  # Data type for vLLM
    split: str = "train"  # Split for dataset
    num_samples: Optional[int] = None  # Optional subsample size; None means full dataset
    
    # Evaluation/logging settings (can be overridden at runtime)
    datasets: Optional[list] = None  # Dataset names (None means use default list)
    summary_csv: Optional[str] = None  # Path to summary CSV file
    run_tag: Optional[str] = None  # Optional tag for runs
    system_prompt: str = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."
    stop_tokens: list[str] = field(default_factory=lambda: ["\n## Step"])

    @classmethod
    def from_yaml(cls, yaml_path: str, **overrides):
        """
        Load configuration from a YAML file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            **overrides: Additional keyword arguments to override YAML values
            
        Returns:
            Config instance with values from YAML file and overrides
            
        Raises:
            ImportError: If PyYAML is not installed
            FileNotFoundError: If YAML file doesn't exist
        """
        if yaml is None:
            raise ImportError(
                "PyYAML is required to load YAML config files. "
                "Install it with: pip install pyyaml"
            )
        
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Get all field names from the dataclass
        field_names = {f.name for f in fields(cls)}
        
        # Filter YAML config to only include valid Config fields
        # Exclude runtime-only fields that shouldn't be in YAML
        runtime_fields = {'gpu_mem_util', 'tensor_parallel_size', 'dtype', 'dataset', 'summary_csv', 'run_tag'}
        config_dict = {k: v for k, v in yaml_config.items() if k in field_names and k not in runtime_fields}
        
        # Apply overrides
        config_dict.update(overrides)
        
        return cls(**config_dict)


# Default configuration instance for backward compatibility
default_config = Config()

# Export individual attributes for backward compatibility
MODEL_NAME = default_config.model_name
TAU = default_config.tau
MAX_BACKTRACKS = default_config.max_backtracks
NUM_SAMPLES = default_config.num_samples
FINAL_ANSWER_PREFIX = default_config.final_answer_prefix
DEBUG = default_config.debug
VERBOSE = default_config.verbose
EPSILON = default_config.epsilon
SEED = default_config.seed
DETAIL_LOG = default_config.detail_log
TEMP_ORIGIN = default_config.temp_origin
TEMP_STEP = default_config.temp_step
MAX_TOTAL_BRANCHES = default_config.max_total_branches
PASSING_MINIMUM = default_config.passing_minimum
KEEPING_BRANCHES = default_config.keeping_branches
MAX_MODEL_LEN = default_config.max_model_len
MAX_STEPS = default_config.max_steps
K_TOKENS = default_config.k_tokens