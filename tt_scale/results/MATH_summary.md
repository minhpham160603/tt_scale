# Experiment Results Summary: Hybrid Sampling & Baselines

This document summarizes the accuracy results on the **MATH-500** subset (N=50) across different model scales and configurations.

## 1. Qwen 3 (0.6B) Experiments

*Small scale testing to identify optimal hyperparameters for backtracking.*

| Strategy | Config Details | Accuracy | vs Baseline |
| :--- | :--- | :--- | :--- |
| **Baseline** | Standard Greedy Decoding | **48.00%** | \- |
| **Hybrid (Prompted)** | K=128, $\tau$=0.5, MaxSteps=5 | 40.00% | -8.0% |
| **Hybrid (Prompted)** | K=256, $\tau$=0.6, MaxSteps=5 | **50.00%** | +2.0% |
| **Hybrid (Prompted)** | K=256, $\tau$=0.6, Temp=0 Start | **50.00%** | +2.0% |
| **Hybrid + Llama 8B PRM** | Logits PRM, K=256, $\tau$=0.6 | 46.00% | -2.0% |
| **Hybrid + Llama 8B PRM** | Logits PRM, K=256, $\tau$=0.8 | **50.00%** | +2.0% |

> **Observation:** The 0.6B model is highly sensitive to the threshold ($\tau$). A low threshold (0.5) increases false positives, while a stricter threshold (0.6-0.8) or larger step size (K=256) allows it to slightly outperform the baseline.

## 2. Qwen 3 (4B) Base Experiments

*Mid-scale testing using the base model.*

| Strategy | Config Details | Accuracy | vs Baseline |
| :--- | :--- | :--- | :--- |
| **Baseline** | Standard Greedy Decoding | **56.00%** | \- |
| **Hybrid (Prompted)** | Self-Correction (Prompted PRM) | 46.00% | -10.0% |

> **Observation:** Significant regression. The base 4B model likely struggles with accurate self-evaluation (Prompted PRM), causing it to reject correct steps or hallucinate corrections, disrupting the chain of thought.

## 3. Qwen 3 (4B) Instruct Experiments

*Testing with the instruction-tuned variant (FP8).*

| Strategy | Config Details | Accuracy | vs Baseline |
| :--- | :--- | :--- | :--- |
| **Baseline** | Standard Greedy Decoding | **76.00%** | \- |

> **Observation:** The Instruct model provides a massive jump in baseline performance (+20% over base 4B), proving that instruction tuning is critical for math reasoning tasks.

## 4. Key Conclusions

1. **The "Sycophancy" Trap:** Using the generator as its own judge (Prompted PRM) is risky. On the 4B Base model, it actively hurt performance (-10%), likely because the model cannot reliably identify its own errors.

2. **Hyperparameter Sensitivity:** In the 0.6B experiments, simply changing $K$ (step size) from 128 to 256 and $\tau$ from 0.5 to 0.6 flipped the result from a loss (-8%) to a gain (+2%).

3. **External Verifiers Help:** The Llama-3-8B Logits PRM matched the best self-correction score (50%) but required a high threshold ($\tau=0.8$) to work effectively.

4. **Next Steps:** The high baseline of the **4B Instruct (76%)** makes it the best candidate for future Hybrid Search tests, but it requires a robust PRM (like Llama-8B) rather than self-prompting to improve further.
