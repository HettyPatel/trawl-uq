"""
Test different models for knowledge extraction quality.
Evaluates: Qwen-2.5-7B-Instruct, Mistral-7B-Instruct, Llama-3-8B-Instruct

Tests each model on:
1. Valid responses - should extract actual facts
2. Gibberish/degenerate responses - should return NO_FACTS_FOUND or similar
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

# Test cases: (question, response, expected_behavior)
TEST_CASES = [
    # Valid responses - should extract facts
    {
        "name": "valid_factual",
        "question": "Who was the first president of the United States?",
        "response": "George Washington was the first president of the United States, serving from 1789 to 1797.",
        "expected": "extract_facts",
        "description": "Clear factual response"
    },
    {
        "name": "valid_detailed",
        "question": "What is the capital of France?",
        "response": "Paris is the capital of France. It's located in the north-central part of the country along the Seine River.",
        "expected": "extract_facts",
        "description": "Detailed factual response"
    },
    {
        "name": "valid_historical",
        "question": "When did World War II end?",
        "response": "World War II ended in 1945. Germany surrendered in May 1945, and Japan surrendered in September 1945 after the atomic bombings.",
        "expected": "extract_facts",
        "description": "Historical facts with dates"
    },

    # Gibberish/degenerate responses - should NOT extract facts
    {
        "name": "gibberish_cyrillic",
        "question": "Who invented the telephone?",
        "response": "вания против власти власти власти власти власти против власти власти",
        "expected": "no_facts",
        "description": "Cyrillic gibberish"
    },
    {
        "name": "gibberish_repetitive",
        "question": "What is photosynthesis?",
        "response": "the the the the the the the the the the the the the the the",
        "expected": "no_facts",
        "description": "Repetitive words"
    },
    {
        "name": "gibberish_numbers",
        "question": "Who wrote Romeo and Juliet?",
        "response": "19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19",
        "expected": "no_facts",
        "description": "Repetitive numbers"
    },
    {
        "name": "gibberish_hren",
        "question": "What is the speed of light?",
        "response": "1hrenatted 2hrenatted 3hrenatted цивилизации",
        "expected": "no_facts",
        "description": "Broken tokens with hren pattern"
    },
    {
        "name": "gibberish_concatenated",
        "question": "Who painted the Mona Lisa?",
        "response": "monclanMountain curleyInonu writersPortions seekingAlbert",
        "expected": "no_facts",
        "description": "Concatenated token gibberish"
    },
    {
        "name": "gibberish_mixed",
        "question": "What is DNA?",
        "response": "• 2005 • 1972 • 1969 • 2005 • 1972 • 1969",
        "expected": "no_facts",
        "description": "Bullet point year lists"
    },
    {
        "name": "empty_response",
        "question": "What is gravity?",
        "response": "",
        "expected": "no_facts",
        "description": "Empty response"
    },
]

# Models to test
MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
]

def extract_knowledge(model, tokenizer, question: str, response: str, device: str = "cuda") -> str:
    """Extract facts from a response using the given model."""

    if not response or len(response.strip()) == 0:
        return "[EMPTY_RESPONSE]"

    # Simple, clear prompt
    prompt = f"""List the factual claims in this response as bullet points.
Only extract facts that are explicitly stated. Do not add your own knowledge.
If there are no facts or the response is gibberish/nonsense, say "NO_FACTS_FOUND".

Question: {question}
Response: {response}

Facts:"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=150,
            temperature=0.3,  # Lower temperature for more consistent output
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs.input_ids.shape[1]
    extracted = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True
    ).strip()

    return extracted


def evaluate_extraction(extracted: str, expected: str) -> dict:
    """Evaluate if the extraction matches expected behavior."""

    extracted_upper = extracted.upper()

    # Check for no facts indicators
    no_facts_indicators = [
        "NO_FACTS_FOUND",
        "NO FACTS",
        "NO FACTUAL",
        "CANNOT EXTRACT",
        "UNABLE TO EXTRACT",
        "NOT POSSIBLE TO EXTRACT",
        "GIBBERISH",
        "NONSENSE",
        "INCOHERENT",
        "NO INFORMATION",
        "DOES NOT CONTAIN",
        "DOESN'T CONTAIN",
    ]

    is_no_facts = any(ind in extracted_upper for ind in no_facts_indicators)
    is_empty = len(extracted.strip()) < 10

    if expected == "no_facts":
        # Should have detected no facts
        success = is_no_facts or is_empty
        return {
            "success": success,
            "reason": "Correctly identified no facts" if success else "Failed: extracted 'facts' from gibberish"
        }
    else:
        # Should have extracted actual facts
        has_bullet = "•" in extracted or "-" in extracted or "*" in extracted or "1." in extracted
        has_content = len(extracted) > 20
        success = has_content and not is_no_facts
        return {
            "success": success,
            "reason": "Correctly extracted facts" if success else "Failed: didn't extract facts from valid response"
        }


def test_model(model_name: str, device: str = "cuda"):
    """Test a single model on all test cases."""

    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Model loaded.\n")

    results = []

    for i, test in enumerate(TEST_CASES):
        print(f"\nTest {i+1}/{len(TEST_CASES)}: {test['name']}")
        print(f"  Description: {test['description']}")
        print(f"  Expected: {test['expected']}")

        # Extract
        extracted = extract_knowledge(
            model, tokenizer,
            test["question"], test["response"],
            device
        )

        # Evaluate
        eval_result = evaluate_extraction(extracted, test["expected"])

        print(f"  Extracted: {extracted[:100]}..." if len(extracted) > 100 else f"  Extracted: {extracted}")
        print(f"  Result: {'✓ PASS' if eval_result['success'] else '✗ FAIL'} - {eval_result['reason']}")

        results.append({
            "test_name": test["name"],
            "expected": test["expected"],
            "extracted": extracted,
            "success": eval_result["success"],
            "reason": eval_result["reason"]
        })

    # Summary
    successes = sum(1 for r in results if r["success"])
    valid_tests = [r for r in results if r["expected"] == "extract_facts"]
    gibberish_tests = [r for r in results if r["expected"] == "no_facts"]

    valid_success = sum(1 for r in valid_tests if r["success"])
    gibberish_success = sum(1 for r in gibberish_tests if r["success"])

    print(f"\n{'-'*40}")
    print(f"SUMMARY for {model_name.split('/')[-1]}:")
    print(f"  Overall: {successes}/{len(results)} passed ({100*successes/len(results):.1f}%)")
    print(f"  Valid responses: {valid_success}/{len(valid_tests)} ({100*valid_success/len(valid_tests):.1f}%)")
    print(f"  Gibberish detection: {gibberish_success}/{len(gibberish_tests)} ({100*gibberish_success/len(gibberish_tests):.1f}%)")

    # Cleanup
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "total_score": successes / len(results),
        "valid_score": valid_success / len(valid_tests),
        "gibberish_score": gibberish_success / len(gibberish_tests),
        "results": results
    }


def main():
    print("="*60)
    print("Knowledge Extractor Model Comparison")
    print("="*60)
    print(f"\nTesting {len(MODELS)} models on {len(TEST_CASES)} test cases")
    print(f"  - {len([t for t in TEST_CASES if t['expected'] == 'extract_facts'])} valid response tests")
    print(f"  - {len([t for t in TEST_CASES if t['expected'] == 'no_facts'])} gibberish detection tests")

    all_results = []

    for model_name in MODELS:
        try:
            result = test_model(model_name)
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR testing {model_name}: {e}")
            all_results.append({
                "model": model_name,
                "error": str(e)
            })

    # Final comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"\n{'Model':<40} {'Overall':>10} {'Valid':>10} {'Gibberish':>10}")
    print("-"*70)

    for r in all_results:
        if "error" in r:
            print(f"{r['model'].split('/')[-1]:<40} {'ERROR':>10}")
        else:
            print(f"{r['model'].split('/')[-1]:<40} {r['total_score']*100:>9.1f}% {r['valid_score']*100:>9.1f}% {r['gibberish_score']*100:>9.1f}%")

    # Recommendation
    valid_results = [r for r in all_results if "error" not in r]
    if valid_results:
        # Prioritize gibberish detection (critical for uncertainty quantification)
        best = max(valid_results, key=lambda x: x['gibberish_score'] * 0.6 + x['valid_score'] * 0.4)
        print(f"\n✓ RECOMMENDED: {best['model'].split('/')[-1]}")
        print(f"  Best gibberish detection ({best['gibberish_score']*100:.1f}%) while maintaining good fact extraction ({best['valid_score']*100:.1f}%)")


if __name__ == "__main__":
    main()
