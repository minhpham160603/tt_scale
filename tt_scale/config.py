import re

# --- UNIFIED MODEL AND HYPERPARAMETERS ---
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
LOGITS_PRM_MODEL = "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"

# Unified Output Format Constants (Used by Hybrid Engines)
FINAL_ANSWER_PREFIX = "<FINAL>"
STOP_STRING = "<END_STEP>"

# --- UNIFIED PROMPT TEMPLATES ---

UNIFIED_SYS_PROMPT = f"""You are a helpful math solver.
Solve the problem step-by-step to avoid mistakes.
To aid in your reasoning, you must break your answer into logical steps.
After **EVERY logical step** of reasoning, output the token {STOP_STRING}.
When you have found the final answer, output it inside \\boxed{{}}.
"""

# System Prompt used for Self-Correction/PRM (BoN, Hybrid scoring)
PRM_SYS_PROMPT = "You are a strict math grader. Review the problem and the proposed answer. Assign a score from 1 to 10 based on correctness. Output ONLY the score, e.g., 'Score: 8'."


# --- UNIFIED UTILITY FUNCTION ---

def extract_math_answer(text: str) -> float | None:
    """
    Robust extraction for MATH dataset answers, prioritizing \\boxed{} or numbers.
    """
    # 1. Prefer Boxed LaTeX
    boxed_match = re.search(r"\\boxed{([^}]+)}", text)
    if boxed_match:
        content = boxed_match.group(1).strip()
        content = content.replace('$', '').replace('\\', '').replace(',', '')
        try:
            # Try finding the last number in the box
            nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", content)
            if nums: return float(nums[-1])
        except: pass

    # 2. Fallback: Look for the last number in the text
    try:
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text.replace(',', ''))
        if nums: return float(nums[-1])
    except: pass
    return None
