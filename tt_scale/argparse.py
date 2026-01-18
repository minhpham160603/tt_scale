import argparse
from tt_scale.config import Config, default_config


def parse_args(description="Hybrid parallel vLLM runner (uses YAML config file)."):
    """
    Parse command-line arguments and create a Config object.
    
    Returns:
        tuple: (args, config) where args contains runtime-only settings
               and config is a Config object loaded from YAML or defaults
    """
    parser = argparse.ArgumentParser(description=description)
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file. If not provided, uses default config values.",
    )
    
    # Runtime-only vLLM engine parameters (can override defaults)
    parser.add_argument("--gpu-mem-util", type=float, default=None, help="GPU memory utilization (overrides config)")
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="Tensor parallel size (overrides config)")
    parser.add_argument("--dtype", type=str, default=None, help="Data type (overrides config)")
    
    # Runtime-only evaluation/logging settings
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset name (repeatable). If omitted, uses built-in list.",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Path to summary CSV file (overrides config)",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional tag for runs (overrides config)",
    )
    parser.add_argument(
        "--detail-log",
        default=None,
        choices=["per-dataset", "none"],
        help="Whether to write per-dataset/per-sample logs (overrides config)",
    )
    
    args = parser.parse_args()
    
    # Load config from YAML or use defaults
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = default_config
    
    # Override runtime-only settings from CLI args
    if args.gpu_mem_util is not None:
        config.gpu_mem_util = args.gpu_mem_util
    if args.tensor_parallel_size is not None:
        config.tensor_parallel_size = args.tensor_parallel_size
    if args.dtype is not None:
        config.dtype = args.dtype
    if args.dataset is not None:
        config.datasets = args.dataset
    if args.summary_csv is not None:
        config.summary_csv = args.summary_csv
    if args.run_tag is not None:
        config.run_tag = args.run_tag
    if args.detail_log is not None:
        config.detail_log = (args.detail_log != "none")
    
    return args, config
