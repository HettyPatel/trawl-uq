"""
Test the answer-only NLL function with debug output.
Verifies tokenization boundary is correct for Llama chat models.
"""
import sys
sys.path.append('.')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.metrics import compute_answer_nll


def run_nll_tests(model_name: str, device: str):
    """Run NLL tests for a single model."""
    print("=" * 70)
    print(f"Testing Answer-Only NLL with {model_name}")
    print("=" * 70)

    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("Model loaded.\n")

    # Test cases - using HotpotQA-style questions
    test_cases = [
        {
            "question": "What is the capital of France?",
            "gold_answer": "Paris",
        },
        {
            "question": "Who wrote the play Romeo and Juliet?",
            "gold_answer": "William Shakespeare",
        },
        {
            "question": "What year did World War II end?",
            "gold_answer": "1945",
        },
    ]

    prompt_styles = [
        ("chat_template", None),
        ("short_answer", lambda q: f"Question: {q}\nAnswer (short):")
    ]

    print("Running NLL tests with DEBUG mode to verify tokenization:\n")

    for i, tc in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}")
        print(f"{'='*70}")

        for style_name, prompt_fn in prompt_styles:
            print(f"\n--- Prompt style: {style_name} ---")
            prompt_text = prompt_fn(tc["question"]) if prompt_fn else None

            nll, ppl = compute_answer_nll(
                question=tc["question"],
                gold_answer=tc["gold_answer"],
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt_text=prompt_text,
                debug=True  # Enable debug output
            )

            print(f"\nRESULTS:")
            print(f"  NLL: {nll:.4f}")
            print(f"  PPL: {ppl:.2f}")

    # Sanity check - NLL should be reasonable for chat model
            if nll < 10:
                print("  [OK] NLL is in reasonable range for chat model")
            else:
                print("  [WARNING] NLL seems high - check tokenization")

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    models_to_test = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "gpt2"
    ]

    for model_name in models_to_test:
        run_nll_tests(model_name=model_name, device=device)
