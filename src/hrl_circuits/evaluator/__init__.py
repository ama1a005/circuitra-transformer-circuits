"""
hrl_circuits.evaluator
======================
Causal evaluation, budget management, activation caching,
faithfulness computation, and eval counters.
"""

from .counters import EvalCounter
from .budget import BudgetManager, BudgetExhaustedError
from .causal import CausalEvaluator, MockCausalEvaluator, EvalResult
from .cache import ActivationCache
from .faithfulness import compute_faithfulness_from_outputs, normalize_faithfulness

__all__ = [
    "EvalCounter",
    "BudgetManager",
    "BudgetExhaustedError",
    "CausalEvaluator",
    "MockCausalEvaluator",
    "EvalResult",
    "ActivationCache",
    "compute_faithfulness_from_outputs",
    "normalize_faithfulness",
]
