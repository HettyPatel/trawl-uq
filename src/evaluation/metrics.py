"""
Evaluation metrics for downstream task performance.

This module implements standard QA evaluation metrics:
- Token F1: Measures token-level overlap between prediction and gold answer
- Answer NLL: Measures how well model predicts the gold answer (proper downstream metric)
- Answer Entropy: Measures model uncertainty over answer token positions
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


def compute_answer_entropy(
    question: str,
    gold_answer: str,
    model,
    tokenizer,
    device: str = "cuda",
    prompt_text: Optional[str] = None,
    max_length: int = 512
) -> Tuple[float, float]:
    """
    Compute entropy of the model's probability distribution over answer token positions.

    This measures model uncertainty:
    - High entropy = model is uncertain (probability spread across many tokens)
    - Low entropy = model is confident (probability concentrated on few tokens)

    Unlike NLL which only looks at the correct token's probability, entropy considers
    the full distribution. This naturally handles model collapse (confident but wrong
    predictions result in low entropy, which combined with high NLL signals a problem).

    Args:
        question: The input question/prompt
        gold_answer: Ground truth answer (used to determine which positions to measure)
        model: Language model
        tokenizer: Corresponding tokenizer
        device: Device for computation (cuda/cpu)
        prompt_text: Optional pre-formatted prompt text
        max_length: Maximum sequence length

    Returns:
        Tuple of (entropy, normalized_entropy):
            - entropy: Average entropy over answer tokens (in nats)
            - normalized_entropy: Entropy normalized by log(vocab_size), range [0, 1]
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
            prompt_text = f"Question: {question}\nAnswer:"

    # Full text = prompt + space + gold answer
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

    # Find prompt boundary
    enc_prompt = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(device)
    prompt_len = enc_prompt["input_ids"].shape[1]

    if prompt_len >= input_ids.shape[1]:
        return float('inf'), float('inf')

    # Get logits
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Compute entropy only for answer token positions
    # For position i, we predict token i+1, so use logits[prompt_len-1:-1] for answer tokens
    answer_logits = logits[0, prompt_len-1:-1, :]  # [num_answer_tokens, vocab_size]

    # Compute entropy: -sum(p * log(p))
    log_probs = torch.nn.functional.log_softmax(answer_logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy_per_token = -torch.sum(probs * log_probs, dim=-1)  # [num_answer_tokens]

    # Average entropy over answer tokens
    avg_entropy = entropy_per_token.mean().item()

    # Normalized entropy (0 = fully confident, 1 = uniform distribution)
    vocab_size = logits.shape[-1]
    max_entropy = np.log(vocab_size)
    normalized_entropy = avg_entropy / max_entropy

    return avg_entropy, normalized_entropy


def _compute_nll_entropy_single(
    question: str,
    gold_answer: str,
    model,
    tokenizer,
    device: str,
    prompt_text: str,
    max_length: int
) -> Dict[str, float]:
    """Helper to compute NLL/entropy for a single answer."""
    if len(gold_answer.strip()) == 0:
        return {
            'nll': float('inf'),
            'ppl': float('inf'),
            'entropy': float('inf'),
            'normalized_entropy': float('inf'),
            'num_answer_tokens': 0
        }

    full_text = prompt_text + " " + gold_answer

    # Tokenize
    enc_full = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(device)

    input_ids = enc_full["input_ids"]
    attention_mask = enc_full["attention_mask"]

    enc_prompt = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(device)
    prompt_len = enc_prompt["input_ids"].shape[1]

    if prompt_len >= input_ids.shape[1]:
        return {
            'nll': float('inf'),
            'ppl': float('inf'),
            'entropy': float('inf'),
            'normalized_entropy': float('inf'),
            'num_answer_tokens': 0
        }

    num_answer_tokens = input_ids.shape[1] - prompt_len

    # Single forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits

    # Extract logits for answer positions
    answer_logits = logits[0, prompt_len-1:-1, :]
    answer_targets = input_ids[0, prompt_len:]

    # Compute NLL
    log_probs = torch.nn.functional.log_softmax(answer_logits, dim=-1)
    target_log_probs = log_probs.gather(1, answer_targets.unsqueeze(1)).squeeze(1)
    nll = -target_log_probs.mean().item()
    ppl = np.exp(nll)

    # Compute entropy
    probs = torch.exp(log_probs)
    entropy_per_token = -torch.sum(probs * log_probs, dim=-1)
    avg_entropy = entropy_per_token.mean().item()

    vocab_size = logits.shape[-1]
    max_entropy = np.log(vocab_size)
    normalized_entropy = avg_entropy / max_entropy

    return {
        'nll': nll,
        'ppl': ppl,
        'entropy': avg_entropy,
        'normalized_entropy': normalized_entropy,
        'num_answer_tokens': num_answer_tokens
    }


def compute_answer_nll_and_entropy(
    question: str,
    gold_answer: str,
    model,
    tokenizer,
    device: str = "cuda",
    prompt_text: Optional[str] = None,
    max_length: int = 512,
    all_answers: Optional[list] = None
) -> Dict[str, float]:
    """
    Compute both NLL and entropy in a single forward pass.

    If multiple answers are provided via all_answers, computes metrics for each
    and returns the result with the lowest NLL (most favorable to the model).

    Args:
        question: The input question/prompt
        gold_answer: Ground truth answer (used if all_answers not provided)
        model: Language model
        tokenizer: Corresponding tokenizer
        device: Device for computation
        prompt_text: Optional pre-formatted prompt text
        max_length: Maximum sequence length
        all_answers: Optional list of all valid answers (takes min NLL across all)

    Returns:
        Dictionary containing:
            - nll: Average NLL over answer tokens (min across all answers if multiple)
            - ppl: Perplexity (exp(nll))
            - entropy: Average entropy over answer tokens
            - normalized_entropy: Entropy normalized to [0, 1]
            - num_answer_tokens: Number of tokens in the answer
            - best_answer: The answer that achieved lowest NLL (if multiple)
    """
    # Build list of answers to evaluate
    answers_to_check = all_answers if all_answers else [gold_answer]

    # Filter out empty answers
    answers_to_check = [a for a in answers_to_check if a and a.strip()]

    if not answers_to_check:
        return {
            'nll': float('inf'),
            'ppl': float('inf'),
            'entropy': float('inf'),
            'normalized_entropy': float('inf'),
            'num_answer_tokens': 0,
            'best_answer': None
        }

    model.eval()

    # Build prompt
    if prompt_text is None:
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": question}]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = f"Question: {question}\nAnswer:"

    # Compute metrics for each answer and keep best (lowest NLL)
    best_result = None
    best_answer = None

    for answer in answers_to_check:
        result = _compute_nll_entropy_single(
            question, answer, model, tokenizer, device, prompt_text, max_length
        )
        if best_result is None or result['nll'] < best_result['nll']:
            best_result = result
            best_answer = answer

    best_result['best_answer'] = best_answer
    return best_result


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


# =============================================================================
# MCQ (Multiple Choice) Entropy and NLL
# =============================================================================

def compute_mcq_entropy_and_nll(
    mcq_prompt: str,
    correct_letter: str,
    model,
    tokenizer,
    device: str = "cuda",
    options: list = None
) -> Dict[str, float]:
    """
    Compute entropy and NLL for multiple choice questions.

    Entropy is calculated over the 4 option probabilities (A, B, C, D) only,
    giving a cleaner signal than vocabulary-wide entropy.

    Args:
        mcq_prompt: Full MCQ prompt ending with "Answer:"
        correct_letter: The correct answer letter (A, B, C, or D)
        model: Language model
        tokenizer: Corresponding tokenizer
        device: Device for computation
        options: Optional list of option letters (default: ['A', 'B', 'C', 'D'])

    Returns:
        Dictionary containing:
            - nll: Negative log-likelihood of correct answer
            - entropy: Entropy over the 4 options (max = 1.386 for uniform)
            - normalized_entropy: Entropy normalized to [0, 1]
            - probs: Dict of probabilities for each option
            - correct_prob: Probability assigned to correct answer
            - predicted_letter: Letter with highest probability
            - is_correct: Whether prediction matches correct answer
    """
    if options is None:
        options = ['A', 'B', 'C', 'D']

    model.eval()

    # Tokenize prompt
    inputs = tokenizer(mcq_prompt, return_tensors="pt").to(device)

    # Get logits for next token
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last position

    # Get token IDs for A, B, C, D
    # Try different tokenizations (with/without space)
    option_token_ids = {}
    for letter in options:
        # Try with space first (common for most models)
        token_id = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if len(token_id) == 1:
            option_token_ids[letter] = token_id[0]
        else:
            # Try without space
            token_id = tokenizer.encode(letter, add_special_tokens=False)
            if len(token_id) == 1:
                option_token_ids[letter] = token_id[0]
            else:
                # Use first token
                option_token_ids[letter] = token_id[0] if token_id else tokenizer.unk_token_id

    # Extract logits for option tokens
    option_logits = torch.tensor([logits[option_token_ids[l]].item() for l in options])

    # Softmax over just the 4 options
    option_probs = torch.nn.functional.softmax(option_logits, dim=0)

    # Build probability dict
    probs = {l: option_probs[i].item() for i, l in enumerate(options)}

    # Get correct answer probability
    correct_idx = options.index(correct_letter)
    correct_prob = option_probs[correct_idx].item()

    # NLL for correct answer
    nll = -np.log(correct_prob + 1e-10)  # Add small epsilon to avoid log(0)

    # Entropy over 4 options
    entropy = -torch.sum(option_probs * torch.log(option_probs + 1e-10)).item()

    # Max entropy for 4 options = log(4) = 1.386
    max_entropy = np.log(len(options))
    normalized_entropy = entropy / max_entropy

    # Predicted letter (highest prob)
    predicted_idx = torch.argmax(option_probs).item()
    predicted_letter = options[predicted_idx]

    return {
        'nll': nll,
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'max_entropy': max_entropy,
        'probs': probs,
        'correct_prob': correct_prob,
        'predicted_letter': predicted_letter,
        'is_correct': predicted_letter == correct_letter
    }
