from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any, Union
from transformers import DynamicCache
from tt_scale.config import Config

# ==========================================
# Abstract Generator Class
# ==========================================
class AbstractGenerator(ABC):
    """
    Abstract base class for a Language Model Generator capable of step-wise generation.
    It abstracts away the specific backend (HuggingFace, vLLM, OpenAI API, etc.)
    and the state management of the generation (KV-Cache).
    """

    @abstractmethod
    def tokenize(self, text: str) -> Any:
        """
        Converts text to the model's native input format (e.g., torch.Tensor, list of ints).
        """
        pass

    @abstractmethod
    def decode(self, token_ids: Any) -> str:
        """
        Converts the model's native output format back to a string.
        """
        pass

    @abstractmethod
    def generate_step(
        self, 
        input_ids: Any, 
        past_key_values: Optional[Any], 
        max_new_tokens: int, 
        temperature: float = 0.7,
        stop_strings: Optional[List[str]] = None
    ) -> Tuple[Any, Any, bool]:
        """
        Generates the next 'chunk' or 'step' of text.
        
        Args:
            input_ids: The input tokens representing the current state.
            past_key_values: The KV-cache or state object from the previous step.
            max_new_tokens: The maximum number of tokens to generate in this step.
            temperature: Sampling temperature.
            stop_strings: Optional list of strings that, if generated, stop the step early 
                          (e.g., "\n", "Step:", "<extra_0>").

        Returns:
            A tuple containing:
            1. full_sequence (Any): The input_ids + newly generated ids.
            2. new_cache (Any): The updated KV-cache/state to pass to the next call.
            3. finished (bool): True if the model hit an EOS token or a stop string.
            4. new_text (str): The decoded text of *only* the new tokens generated.
        """
        pass
    
    @abstractmethod
    def generate_batch_step(
        self,
        input_ids_batch: List[Any],
        past_key_values_batch: List[Optional[Any]],
        max_new_tokens: int,
        temperature: float = 0.7,
        stop_strings: Optional[List[str]] = None,
    ) -> List[Tuple[Any, Any, bool]]:
        """
        Convenience helper for generating a step for *multiple* branches in parallel.

        The default implementation simply loops over the batch and calls
        :meth:`generate_step` for each element. Concrete generators can override
        this method to use a more efficient batched implementation.

        Args:
            input_ids_batch: List of inputs for each branch.
            past_key_values_batch: List of KV-caches (or ``None``) for each branch.
            max_new_tokens: Maximum number of tokens to generate per branch.
            temperature: Sampling temperature.
            stop_strings: Optional list of stop strings.

        Returns:
            List of tuples ``(full_sequence, new_cache, finished)`` – one per branch.
        """
        pass
    


# ==========================================
# Abstract PRM (Process Reward Model) Class
# ==========================================
class AbstractPRM(ABC):
    """
    Abstract base class for a Process Reward Model (Verifier).
    It abstracts away whether the score comes from a trained classifier head,
    a logit check on specific tokens (Good/Bad), or an LLM-as-a-Judge prompt.
    """

    @abstractmethod
    def get_score(self, question: str, partial_answer: str) -> float:
        """
        Scores a partial solution to a question.

        Args:
            question: The original problem statement.
            partial_answer: The step-by-step reasoning generated so far.

        Returns:
            float: A score representing the quality/correctness of the partial answer.
                   (Convention: usually normalized between 0.0 and 1.0).
        """
        pass

    @abstractmethod
    def get_scores_batch(self, qa_pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Scores a batch of (question, partial_answer) tuples.
        This allows implementations to optimize batch inference (e.g. via vLLM or padding).

        Args:
            qa_pairs: A list of tuples, where each tuple is (question, partial_answer).

        Returns:
            List[float]: A list of scores corresponding to the input pairs.
        """
        pass
    

class Searcher(ABC):
    """
    Abstract base class for a Searcher.
    It abstracts away the specific search algorithm (e.g., collective backtrack, independent backtrack).
    """
    def __init__(self, generator: AbstractGenerator, prm: AbstractPRM, config: Config):
        self.generator = generator
        self.prm = prm
        self.config = config

    @abstractmethod
    def run(self, prompt: str, *args, **kwargs) -> str:
        """
        Searches for the answer to a question.
        """
        pass