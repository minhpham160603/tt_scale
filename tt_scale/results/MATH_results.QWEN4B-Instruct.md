# BAseline Qwen/Qwen3-4B-Instruct-2507-FP8

=== Baseline Accuracy: 76.00% (38/50) ===

# vllm_hybrid self prompted

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507-FP8"
K_TOKENS = 256
TAU = 0.7
MAX_BACKTRACKS = 3
MAX_STEPS = 5
NUM_SAMPLES = 50
temp = min(0.0 + (0.2 * retry_attempt), 1.2)
