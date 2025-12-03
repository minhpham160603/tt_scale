Method,PRM Strategy,Parameters,Accuracy,Avg Time (s/sample)
Baseline,None,Greedy (Temp=0),75.6%,0.08s
,,,,
Best of N,Self-Correction,"n=4,T=0.5",75.6%,0.39s
Best of N,Self-Correction,"n=4,T=1.0",77.4%,0.39s
Best of N,Self-Correction,"n=8,T=0.5",78.2%,0.74s
Best of N,Self-Correction,"n=8,T=1.0",76.2%,0.77s
,,,,
Best of N,Dedicated (Logits),"n=4,T=0.5",75.2%,0.63s
Best of N,Dedicated (Logits),"n=4,T=1.0",75.8%,0.64s
Best of N,Dedicated (Logits),"n=8,T=0.5",77.6%,1.26s
Best of N,Dedicated (Logits),"n=8,T=1.0",77.6%,1.26s
,,,,
Hybrid Tree,Self-Correction,Max Width=16,51.0%,19.36s
Hybrid Tree,Dedicated (Logits),Max Width=16,29.0%,32.20s

![alt text](results_table.png)

    "experiment_name": "baseline_greedy_parallel",
    "metadata": {
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "num_samples": 500,
        "temperature": 0.0,
        "max_tokens": 2048,
        "gpu_memory_utilization": 0.9,
        "strategy": "vllm_continuous_batching",
        "timestamp": "2025-12-02_17-32-53",
        "run_id": "6dcdcb37"
    },
    "metrics": {
        "accuracy": 0.756,
        "correct_count": 378,
        "total_samples": 500,
        "duration_seconds": 39.19487118721008,
        "avg_time_per_sample": 0.07838974237442016
    },


        "experiment_name": "best_of_n_parallel",
    "metadata": {
        "experiment_name": "best_of_n_parallel",
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "prm_mode": "self_correction",
        "prm_model": "Qwen/Qwen3-4B-Instruct-2507",
        "n_candidates": 4,
        "num_samples": 500,
        "temperature_gen": 1.0,
        "gpu_utilization": 0.95,
        "timestamp": "2025-12-02_18-20-10",
        "run_id": "ebdbcfee"
    },
    "metrics": {
        "accuracy": 0.774,
        "correct_count": 387,
        "total_samples": 500,
        "duration_seconds": 192.36343836784363,
        "avg_time_per_sample": 0.38472687673568723,
        "time_generation": 159.6899209022522,
        "time_scoring": 32.67351746559143,
        "avg_time_per_problem": 0.38472687673568723
    },

    "experiment_name": "best_of_n_parallel",
    "metadata": {
        "experiment_name": "best_of_n_parallel",
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "prm_mode": "self_correction",
        "prm_model": "Qwen/Qwen3-4B-Instruct-2507",
        "n_candidates": 8,
        "num_samples": 500,
        "temperature_gen": 0.5,
        "gpu_utilization": 0.95,
        "timestamp": "2025-12-02_18-21-32",
        "run_id": "3e648b2f"
    },
    "metrics": {
        "accuracy": 0.782,
        "correct_count": 391,
        "total_samples": 500,
        "duration_seconds": 371.29212832450867,
        "avg_time_per_sample": 0.7425842566490173,
        "time_generation": 309.5124008655548,
        "time_scoring": 61.77972745895386,
        "avg_time_per_problem": 0.7425842566490173
    },

    "experiment_name": "best_of_n_parallel",
    "metadata": {
        "experiment_name": "best_of_n_parallel",
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "prm_mode": "logits",
        "prm_model": "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
        "n_candidates": 4,
        "num_samples": 500,
        "temperature_gen": 0.5,
        "gpu_utilization": 0.6,
        "timestamp": "2025-12-02_18-24-17",
        "run_id": "037064b5"
    },
    "metrics": {
        "accuracy": 0.752,
        "correct_count": 376,
        "total_samples": 500,
        "duration_seconds": 315.60177969932556,
        "avg_time_per_sample": 0.6312035593986511,
        "time_generation": 169.27005672454834,
        "time_scoring": 146.33172297477722,
        "avg_time_per_problem": 0.6312035593986511

        {
    "experiment_name": "best_of_n_parallel",
    "metadata": {
        "experiment_name": "best_of_n_parallel",
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "prm_mode": "logits",
        "prm_model": "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
        "n_candidates": 4,
        "num_samples": 500,
        "temperature_gen": 1.0,
        "gpu_utilization": 0.6,
        "timestamp": "2025-12-02_18-26-55",
        "run_id": "f9546930"
    },
    "metrics": {
        "accuracy": 0.758,
        "correct_count": 379,
        "total_samples": 500,
        "duration_seconds": 318.70509576797485,
        "avg_time_per_sample": 0.6374101915359497,
        "time_generation": 172.45443725585938,
        "time_scoring": 146.25065851211548,
        "avg_time_per_problem": 0.6374101915359497
    },

        "experiment_name": "best_of_n_parallel",
    "metadata": {
        "experiment_name": "best_of_n_parallel",
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "prm_mode": "logits",
        "prm_model": "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
        "n_candidates": 8,
        "num_samples": 500,
        "temperature_gen": 0.5,
        "gpu_utilization": 0.6,
        "timestamp": "2025-12-02_18-30-46",
        "run_id": "ed5df653"
    },
    "metrics": {
        "accuracy": 0.776,
        "correct_count": 388,
        "total_samples": 500,
        "duration_seconds": 631.4933869838715,
        "avg_time_per_sample": 1.262986773967743,
        "time_generation": 332.52720832824707,
        "time_scoring": 298.9661786556244,
        "avg_time_per_problem": 1.262986773967743
    },

        "experiment_name": "best_of_n_parallel",
    "metadata": {
        "experiment_name": "best_of_n_parallel",
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "prm_mode": "logits",
        "prm_model": "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
        "n_candidates": 8,
        "num_samples": 500,
        "temperature_gen": 1.0,
        "gpu_utilization": 0.6,
        "timestamp": "2025-12-02_18-35-49",
        "run_id": "a2523bde"
    },
    "metrics": {
        "accuracy": 0.776,
        "correct_count": 388,
        "total_samples": 500,
        "duration_seconds": 628.757472038269,
        "avg_time_per_sample": 1.257514944076538,
        "time_generation": 336.0787091255188,
        "time_scoring": 292.67876291275024,
        "avg_time_per_problem": 1.257514944076538
    },

        "experiment_name": "best_of_n_parallel",
    "metadata": {
        "experiment_name": "best_of_n_parallel",
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "prm_mode": "self_correction",
        "prm_model": "Qwen/Qwen3-4B-Instruct-2507",
        "n_candidates": 8,
        "num_samples": 500,
        "temperature_gen": 1.0,
        "gpu_utilization": 0.95,
        "timestamp": "2025-12-02_18-38-11",
        "run_id": "a80f55c2"
    },
    "metrics": {
        "accuracy": 0.762,
        "correct_count": 381,
        "total_samples": 500,
        "duration_seconds": 383.92658019065857,
        "avg_time_per_sample": 0.7678531603813171,
        "time_generation": 318.75170063972473,
        "time_scoring": 65.17487955093384,
        "avg_time_per_problem": 0.7678531603813171
    },

{
    "experiment_name": "best_of_n_parallel",
    "metadata": {
        "experiment_name": "best_of_n_parallel",
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "prm_mode": "self_correction",
        "prm_model": "Qwen/Qwen3-4B-Instruct-2507",
        "n_candidates": 4,
        "num_samples": 500,
        "temperature_gen": 0.5,
        "gpu_utilization": 0.95,
        "timestamp": "2025-12-02_18-50-40",
        "run_id": "7c173fec"
    },
    "metrics": {
        "accuracy": 0.756,
        "correct_count": 378,
        "total_samples": 500,
        "duration_seconds": 194.1269462108612,
        "avg_time_per_sample": 0.3882538924217224,
        "time_generation": 161.6034038066864,
        "time_scoring": 32.523542404174805,
        "avg_time_per_problem": 0.3882538924217224
    },

{
    "experiment_name": "hybrid_tree_search_mitosis",
    "metadata": {
        "experiment_name": "hybrid_tree_search_mitosis",
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "prm_mode": "A",
        "num_samples": 500,
        "max_width": 16,
        "expansion_factor": 2,
        "max_steps": 8,
        "tau": 0.5,
        "temperature": 0.7,
        "description": "Tree search with mitosis (cloning on failure)",
        "timestamp": "2025-12-02_21-58-42",
        "run_id": "0194a726"
    },
    "metrics": {
        "accuracy": 0.51,
        "correct_count": 255,
        "total_samples": 500,
        "duration_seconds": 9677.544921875,
        "avg_time_per_sample": 19.35508984375
    },

    {
    "experiment_name": "hybrid_tree_search_mitosis",
    "metadata": {
        "experiment_name": "hybrid_tree_search_mitosis",
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "prm_mode": "B",
        "num_samples": 500,
        "max_width": 16,
        "expansion_factor": 2,
        "max_steps": 8,
        "tau": 0.5,
        "temperature": 0.7,
        "description": "Tree search with mitosis (cloning on failure)",
        "timestamp": "2025-12-03_02-45-16",
        "run_id": "a80bf4b3"
    },
    "metrics": {
        "accuracy": 0.29,
        "correct_count": 145,
        "total_samples": 500,
        "duration_seconds": 16101.542840003967,
        "avg_time_per_sample": 32.20308568000794
    },
