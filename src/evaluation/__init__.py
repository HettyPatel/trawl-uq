"""
Evaluation metrics for downstream task performance.

This module provides metrics for evaluating model quality on QA tasks:
- Token F1: Standard QA metric measuring answer overlap
- Answer NLL: Measures model's ability to predict the correct answer
"""

from .metrics import (
    normalize_answer,
    compute_token_f1,
    compute_answer_nll,
    compute_evaluation_metrics
)

__all__ = [
    'normalize_answer',
    'compute_token_f1',
    'compute_answer_nll',
    'compute_evaluation_metrics'
]
