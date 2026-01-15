"""
Evaluation metrics for downstream task performance.

This module implements standard QA evaluation metrics:
- Token F1: Measures token-level overlap between prediction and gold answer
- Answer NLL: Measures how well model predicts the gold answer (proper downstream metric)
"""

import re
import string
import torch
import numpy as np
from typing import Dict, Tuple, Optional


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


def compute_answer_nll(
    question: str,
    gold_answer: str,
    model,
    tokenizer,
    device: str = "cuda",
    debug: bool = False,
    prompt_text: Optional[str] = None,
    max_length: int = 512
) -> Tuple[float, float]:
    """
    Compute answer-only NLL (Negative Log-Likelihood) on the gold answer.

    This is the proper downstream metric for QA evaluation:
    - Measures how well the model predicts the correct answer given the question
    - Only scores the answer tokens, not the prompt tokens
    - Lower NLL = model assigns higher probability to gold answer = better

    Args:
        question: The input question/prompt
        gold_answer: Ground truth answer to evaluate against
        model: Language model (must support forward pass with labels)
        tokenizer: Corresponding tokenizer
        device: Device for computation (cuda/cpu)
        debug: If True, print detailed tokenization info for verification

    Returns:
        Tuple of (nll, perplexity):
            - nll: Average NLL over answer tokens (in nats, lower = better)
            - perplexity: exp(nll) for reference
    """
    if len(gold_answer.strip()) == 0:
        return float('inf'), float('inf')

    model.eval()

    # Build prompt using chat template if available
    if prompt_text is None:
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": question}]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback for models without chat template
            prompt_text = f"Question: {question}\nAnswer:"

    # Full text = prompt + space + gold answer
    # The space is important: Llama-2-chat format is [/INST] answer (with space)
    full_text = prompt_text + " " + gold_answer

    # Tokenize full text
    enc_full = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(device)

    input_ids = enc_full["input_ids"]
    attention_mask = enc_full["attention_mask"]

    # Find the boundary by tokenizing the prompt itself to avoid BPE mismatch.
    enc_prompt = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(device)
    prompt_len = enc_prompt["input_ids"].shape[1]

    if prompt_len >= input_ids.shape[1]:
        if debug:
            print("DEBUG: prompt_len >= full_len; likely truncation. Returning inf.")
        return float('inf'), float('inf')

    # Create labels: mask prompt tokens with -100 (ignored in loss)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100  # Only score answer tokens

    if debug:
        print("=" * 60)
        print("DEBUG: compute_answer_nll tokenization check")
        print("=" * 60)
        print(f"Question: {question[:100]}...")
        print(f"Gold answer: '{gold_answer}'")
        print(f"Full text ends with: ...{full_text[-50:]}")
        print(f"Total tokens: {input_ids.shape[1]}")
        print(f"Prompt tokens (computed): {prompt_len}")
        print("-" * 60)
        # Show tokens around the boundary
        boundary_start = max(0, prompt_len - 3)
        boundary_end = min(input_ids.shape[1], prompt_len + 5)
        print("Tokens around prompt/answer boundary:")
        for i in range(boundary_start, boundary_end):
            token_id = input_ids[0, i].item()
            token_str = tokenizer.decode([token_id])
            is_masked = labels[0, i].item() == -100
            marker = "[MASKED]" if is_masked else "[SCORED]"
            print(f"  {i}: '{token_str}' (id={token_id}) {marker}")
        print("=" * 60)

    # Compute loss
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss  # Average NLL over answer tokens

    nll = loss.item()
    ppl = torch.exp(loss).item()

    return nll, ppl


def compute_evaluation_metrics(
    response: str,
    gold_answer: str,
    model,
    tokenizer,
    question: str,
    device: str = "cuda",
    nll_prompt_text: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a single response.

    This is the main function for evaluating downstream task performance.
    Should be called on a single greedy-decoded response.

    Args:
        response: Model-generated response (greedy decoding)
        gold_answer: Ground truth answer
        model: Language model for NLL calculation
        tokenizer: Corresponding tokenizer
        question: Original question (needed for answer NLL)
        device: Device for computation

    Returns:
        metrics: Dictionary containing:
            - f1: Token F1 score [0.0, 1.0]
            - answer_nll: NLL on gold answer (lower = better)
            - answer_ppl: Perplexity on gold answer (exp(nll))
            - response_length: Number of tokens in response
    """
    # Token F1
    f1_score = compute_token_f1(response, gold_answer)

    # Answer NLL (proper downstream metric)
    answer_nll, answer_ppl = compute_answer_nll(
        question=question,
        gold_answer=gold_answer,
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt_text=nll_prompt_text
    )

    # Response length (tokens)
    response_length = len(response.split())

    return {
        'f1': f1_score,
        'answer_nll': answer_nll,
        'answer_ppl': answer_ppl,
        'response_length': response_length
    }
