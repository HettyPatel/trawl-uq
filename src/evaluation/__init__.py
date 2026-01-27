"""
Evaluation metrics for downstream task performance.

This module provides metrics for evaluating model quality on QA tasks:
- Token F1: Standard QA metric measuring answer overlap
- Answer NLL: Measures model's ability to predict the correct answer
- Answer Entropy: Measures model uncertainty over answer positions
"""

from .metrics import (
    normalize_answer,
    compute_token_f1,
    compute_answer_nll,
    compute_answer_entropy,
    compute_answer_nll_and_entropy,
    compute_evaluation_metrics,
    compute_mcq_entropy_and_nll
)

__all__ = [
    'normalize_answer',
    'compute_token_f1',
    'compute_answer_nll',
    'compute_answer_entropy',
    'compute_answer_nll_and_entropy',
    'compute_evaluation_metrics',
    'compute_mcq_entropy_and_nll'
]
