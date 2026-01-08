"""
Test different Knowledge Extractor approaches to see how they handle:
1. Good/coherent responses
2. Gibberish/degenerate responses

Goal: Find an approach that:
- Extracts meaningful facts from good responses
- FAILS or returns empty for gibberish (instead of hallucinating)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
import re

# Test cases
TEST_CASES = [
    {
        "name": "Good response - correct answer",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "response": "Yes, Scott Derrickson and Ed Wood were both American filmmakers. Scott Derrickson was born in Denver, Colorado, and Ed Wood was born in Poughkeepsie, New York.",
        "is_gibberish": False
    },
    {
        "name": "Good response - incorrect answer",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "response": "No, they were not the same nationality. Scott Derrickson is American while Ed Wood was British.",
        "is_gibberish": False
    },
    {
        "name": "Gibberish - repetitive 'the'",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "response": "the the the the the the the the the the the the the the the the the the the the the the the the the",
        "is_gibberish": True
    },
    {
        "name": "Gibberish - repetitive 'a'",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "response": "a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a",
        "is_gibberish": True
    },
    {
        "name": "Gibberish - mixed repetitive",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "response": "completely the the basic proportion the the the the the the the the the the the the the the the the",
        "is_gibberish": True
    },
    {
        "name": "Gibberish - nonsense fragments",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "response": "sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh sh",
        "is_gibberish": True
    },
    {
        "name": "Partial gibberish - starts ok then degrades",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "response": "Scott Derrickson and the the the the the the the the the the the the the the the the the the",
        "is_gibberish": True
    },
    {
        "name": "Good response - different question",
        "question": "What government position was held by the woman who portrayed Corliss Archer?",
        "response": "Shirley Temple portrayed Corliss Archer in the film Kiss and Tell. She later served as U.S. Ambassador to Ghana and Czechoslovakia.",
        "is_gibberish": False
    },
    {
        "name": "Gibberish - different question",
        "question": "What government position was held by the woman who portrayed Corliss Archer?",
        "response": "the the the the the the the the the the the the the the the the the the the the the the the the the",
        "is_gibberish": True
    },
]


class KnowledgeExtractorTest:
    """Test harness for different KE approaches"""

    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf", device: str = "cuda"):
        print(f"Loading model: {model_name}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print("Model loaded.")

    def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        """Generate response for a prompt"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        input_length = inputs.input_ids.shape[1]
        return self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()


# ============================================================
# DIFFERENT KE PROMPT APPROACHES
# ============================================================

def approach_1_original(question: str, response: str) -> str:
    """Original approach - includes question (CURRENT IMPLEMENTATION)"""
    return f"""Extract all factual claims from the following response. Present them as standalone statements that are independent of specific wording. Only include information directly relevant to answering the question.

Question: {question}
Response: {response}

Extracted factual claims:"""


def approach_2_no_question(question: str, response: str) -> str:
    """Remove question from prompt - force extraction from response only"""
    return f"""Extract all factual claims from the following text. Present them as standalone statements. Only list facts that are explicitly stated in the text.

Text: {response}

Extracted factual claims:"""


def approach_3_strict_no_question(question: str, response: str) -> str:
    """Strict prompt without question - explicit instruction to not add info"""
    return f"""Extract factual claims ONLY from the text below.
- List only facts explicitly stated in the text
- Do NOT add any information not present in the text
- If the text contains no factual claims or is incoherent, respond with exactly: NO_FACTS_FOUND

Text: {response}

Extracted claims:"""


def approach_4_strict_with_question(question: str, response: str) -> str:
    """Strict prompt WITH question but explicit no-hallucination instruction"""
    return f"""Extract factual claims from the response below that answer the question.
- ONLY extract facts explicitly stated in the response
- Do NOT use your own knowledge to answer the question
- Do NOT add information not present in the response
- If the response is incoherent or contains no relevant facts, respond with exactly: NO_FACTS_FOUND

Question: {question}
Response: {response}

Extracted claims (or NO_FACTS_FOUND):"""


def approach_5_validation_prompt(question: str, response: str) -> str:
    """Two-step: first validate if response is coherent"""
    return f"""Analyze the following text and extract information:

Step 1 - Is the text coherent and meaningful? (YES/NO)
Step 2 - If YES, list the factual claims. If NO, write "INCOHERENT"

Text: {response}

Analysis:"""


def approach_6_response_only_minimal(question: str, response: str) -> str:
    """Minimal prompt - just ask for facts from text"""
    return f"""List the facts stated in this text:

{response}

Facts:"""


# Pre-filtering approaches (before sending to LLM)

def prefilter_repetition_check(response: str) -> bool:
    """
    Check if response is degenerate based on repetition patterns.
    Returns True if response should be REJECTED (is gibberish).
    """
    words = response.lower().split()

    # Check 1: Very low unique word ratio
    if len(words) > 5:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return True

    # Check 2: Repetitive word patterns (same word 5+ times in a row)
    if re.search(r'\b(\w+)(\s+\1){4,}', response.lower()):
        return True

    # Check 3: Very short repeated fragments
    if re.search(r'(.{1,3})\1{5,}', response):
        return True

    return False


# ============================================================
# TEST RUNNER
# ============================================================

def run_tests(extractor: KnowledgeExtractorTest):
    """Run all test cases with all approaches"""

    approaches = [
        ("1. Original (with question)", approach_1_original),
        ("2. No question", approach_2_no_question),
        ("3. Strict, no question", approach_3_strict_no_question),
        ("4. Strict, with question", approach_4_strict_with_question),
        ("5. Validation prompt", approach_5_validation_prompt),
        ("6. Minimal, response only", approach_6_response_only_minimal),
    ]

    results = []

    print("\n" + "=" * 80)
    print("RUNNING KNOWLEDGE EXTRACTOR TESTS")
    print("=" * 80)

    for test_case in TEST_CASES:
        print(f"\n{'=' * 80}")
        print(f"TEST: {test_case['name']}")
        print(f"Is Gibberish: {test_case['is_gibberish']}")
        print(f"Question: {test_case['question'][:60]}...")
        print(f"Response: {test_case['response'][:60]}...")
        print("=" * 80)

        # Pre-filter check
        prefilter_result = prefilter_repetition_check(test_case['response'])
        print(f"\nPRE-FILTER (repetition check): {'REJECT' if prefilter_result else 'PASS'}")

        test_results = {
            "test_name": test_case['name'],
            "is_gibberish": test_case['is_gibberish'],
            "prefilter_rejected": prefilter_result,
            "approach_results": {}
        }

        for approach_name, approach_fn in approaches:
            print(f"\n--- {approach_name} ---")
            prompt = approach_fn(test_case['question'], test_case['response'])

            try:
                output = extractor.generate(prompt, max_new_tokens=150)
                print(f"Output: {output[:200]}{'...' if len(output) > 200 else ''}")

                # Check if output indicates failure/no facts
                indicates_no_facts = any(marker in output.upper() for marker in
                                        ["NO_FACTS", "NO FACTS", "INCOHERENT", "NO CLAIM", "NONE"])

                test_results["approach_results"][approach_name] = {
                    "output": output,
                    "indicates_no_facts": indicates_no_facts
                }

            except Exception as e:
                print(f"Error: {e}")
                test_results["approach_results"][approach_name] = {
                    "output": f"ERROR: {e}",
                    "indicates_no_facts": True
                }

        results.append(test_results)

    return results


def print_summary(results: List[Dict]):
    """Print summary of results"""

    print("\n" + "=" * 80)
    print("SUMMARY: How well did each approach handle gibberish?")
    print("=" * 80)
    print()
    print("Goal: For gibberish inputs, we want the approach to FAIL (return no facts)")
    print("      For good inputs, we want meaningful extraction")
    print()

    # Separate gibberish and good test cases
    gibberish_tests = [r for r in results if r['is_gibberish']]
    good_tests = [r for r in results if not r['is_gibberish']]

    print("PRE-FILTER (repetition check):")
    prefilter_gibberish_caught = sum(1 for r in gibberish_tests if r['prefilter_rejected'])
    prefilter_good_rejected = sum(1 for r in good_tests if r['prefilter_rejected'])
    print(f"  Gibberish caught: {prefilter_gibberish_caught}/{len(gibberish_tests)}")
    print(f"  Good incorrectly rejected: {prefilter_good_rejected}/{len(good_tests)}")
    print()

    # Get approach names from first result
    approach_names = list(results[0]['approach_results'].keys())

    print("LLM APPROACHES:")
    print("-" * 80)
    print(f"{'Approach':<35} {'Gibberish→NoFacts':<20} {'Good→HasFacts':<20}")
    print("-" * 80)

    for approach_name in approach_names:
        # Count how many gibberish cases returned "no facts"
        gibberish_no_facts = sum(
            1 for r in gibberish_tests
            if r['approach_results'][approach_name]['indicates_no_facts']
        )

        # Count how many good cases returned facts (not "no facts")
        good_has_facts = sum(
            1 for r in good_tests
            if not r['approach_results'][approach_name]['indicates_no_facts']
        )

        print(f"{approach_name:<35} {gibberish_no_facts}/{len(gibberish_tests):<19} {good_has_facts}/{len(good_tests):<19}")

    print()
    print("BEST APPROACH: Highest 'Gibberish→NoFacts' + Highest 'Good→HasFacts'")
    print()

    # Detailed view of each gibberish case
    print("=" * 80)
    print("DETAILED: What did each approach output for gibberish?")
    print("=" * 80)

    for r in gibberish_tests[:3]:  # Show first 3 gibberish cases
        print(f"\n--- {r['test_name']} ---")
        for approach_name, approach_result in r['approach_results'].items():
            output_preview = approach_result['output'][:100].replace('\n', ' ')
            status = "NO_FACTS" if approach_result['indicates_no_facts'] else "HALLUCINATED"
            print(f"  {approach_name}: [{status}] {output_preview}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test Knowledge Extractor approaches')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help='Model to use for testing')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test with fewer cases')

    args = parser.parse_args()

    # Initialize extractor
    extractor = KnowledgeExtractorTest(model_name=args.model)

    # Run tests
    if args.quick:
        # Use only first 4 test cases for quick testing
        original_cases = TEST_CASES.copy()
        TEST_CASES.clear()
        TEST_CASES.extend(original_cases[:4])

    results = run_tests(extractor)

    # Print summary
    print_summary(results)
