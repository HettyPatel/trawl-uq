"""
Evaluation metrics for downstream task performance.

This module provides metrics for evaluating model quality on QA tasks:
- Token F1: Standard QA metric measuring answer overlap
- Perplexity: Language model confidence/fluency metric
"""

from .metrics import (
    normalize_answer,
    compute_token_f1,
    compute_perplexity,
    compute_evaluation_metrics
)

__all__ = [
    'normalize_answer',
    'compute_token_f1',
    'compute_perplexity',
    'compute_evaluation_metrics'
]
