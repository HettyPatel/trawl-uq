"""
Evaluation metrics for downstream task performance.

This module implements standard QA evaluation metrics:
- Token F1: Measures token-level overlap between prediction and gold answer
- Perplexity: Measures model confidence/fluency on generated text
"""

import re
import string
import torch
import numpy as np
from typing import Dict


def normalize_answer(text: str) -> str:
    """
    Normalize answer text following standard QA evaluation protocol.

    Applies standard preprocessing from SQuAD/HotpotQA/CoQA:
    1. Lowercase
    2. Remove punctuation
    3. Remove articles (a, an, the)
    4. Remove extra whitespace

    Args:
        text: Raw answer text

    Returns:
        normalized_text: Normalized answer string
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = ''.join(ch if ch not in string.punctuation else ' ' for ch in text)

    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


def compute_token_f1(prediction: str, gold_answer: str) -> float:
    """
    Compute token-level F1 score between prediction and gold answer.

    This is the standard evaluation metric for QA tasks (SQuAD, HotpotQA, CoQA).

    Algorithm:
    1. Normalize both texts (lowercase, remove punctuation, articles)
    2. Tokenize by whitespace
    3. Compute precision = |common tokens| / |pred tokens|
    4. Compute recall = |common tokens| / |gold tokens|
    5. F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        prediction: Model-generated answer
        gold_answer: Ground truth answer

    Returns:
        f1_score: Token F1 score [0.0, 1.0] where 1.0 = perfect match
    """
    # Normalize and tokenize
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold_answer).split()

    # Handle edge cases
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    # Find common tokens
    common_tokens = set(pred_tokens) & set(gold_tokens)

    if len(common_tokens) == 0:
        return 0.0

    # Compute precision and recall
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gold_tokens)

    # Compute F1
    f1 = 2 * (precision * recall) / (precision + recall)

    return float(f1)


def compute_perplexity(
    text: str,
    model,
    tokenizer,
    device: str = "cuda"
) -> float:
    """
    Compute perplexity of generated text using the model.

    Perplexity measures how "surprised" the model is by the text:
    - Lower perplexity = more fluent/expected text
    - Higher perplexity = less fluent/unexpected text

    Perplexity = exp(average negative log-likelihood)

    Args:
        text: Generated text to evaluate
        model: Language model (must support forward pass with labels)
        tokenizer: Corresponding tokenizer
        device: Device for computation (cuda/cpu)

    Returns:
        perplexity: Perplexity score (lower = better)
    """
    if len(text.strip()) == 0:
        return float('inf')

    # Tokenize the text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512  # Limit length for memory
    ).to(device)

    # Compute loss (negative log-likelihood)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss

    # Perplexity = exp(loss)
    perplexity = torch.exp(loss).item()

    return float(perplexity)


def compute_evaluation_metrics(
    response: str,
    gold_answer: str,
    model,
    tokenizer,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a single response.

    This is the main function for evaluating downstream task performance.
    Should be called on a single greedy-decoded response.

    Args:
        response: Model-generated response (greedy decoding)
        gold_answer: Ground truth answer
        model: Language model for perplexity calculation
        tokenizer: Corresponding tokenizer
        device: Device for computation

    Returns:
        metrics: Dictionary containing:
            - f1: Token F1 score [0.0, 1.0]
            - perplexity: Perplexity score (lower = better)
            - response_length: Number of tokens in response
    """
    # Token F1
    f1_score = compute_token_f1(response, gold_answer)

    # Perplexity
    perplexity = compute_perplexity(response, model, tokenizer, device)

    # Response length (tokens)
    response_length = len(response.split())

    return {
        'f1': f1_score,
        'perplexity': perplexity,
        'response_length': response_length
    }
