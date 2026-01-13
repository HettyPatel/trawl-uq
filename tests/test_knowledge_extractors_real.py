"""
Test knowledge extractors on REAL generated responses from our experiments.
Includes valid responses and actual degenerate outputs from SVD truncation experiments.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

# Real test cases from our experiments
TEST_CASES = [
    # ========== VALID RESPONSES (should extract facts) ==========
    {
        "name": "valid_moon_detailed",
        "question": "when was the last time anyone was on the moon",
        "response": "The last time anyone was on the moon was during the Apollo 17 mission in December 1972. The mission crew consisted of Eugene Cernan, Ronald Evans, and Harrison Schmitt. Cernan and Schmitt spent three days on the lunar surface, conducting scientific experiments and exploring the Taurus-Littrow valley.",
        "expected": "extract_facts",
        "description": "Clear factual response about Apollo 17"
    },
    {
        "name": "valid_moon_short",
        "question": "when was the last time anyone was on the moon",
        "response": "Since 1972, when the last manned Apollo mission, Apollo 17, landed on the lunar surface, no human has set foot on the moon.",
        "expected": "extract_facts",
        "description": "Short but valid factual response"
    },
    {
        "name": "valid_song_correct",
        "question": "who wrote he ain't heavy he's my brother lyrics",
        "response": "The lyrics to \"He Ain't Heavy, He's My Brother\" were written by Bob Russell and Bobby Scott. The song was written in 1969 and it was recorded by The Hollies in 1969.",
        "expected": "extract_facts",
        "description": "Correct song attribution"
    },
    {
        "name": "valid_song_detailed",
        "question": "who wrote he ain't heavy he's my brother lyrics",
        "response": "The lyrics to \"He Ain't Heavy, He's My Brother\" were written by Holly Knight and Mike Chapman. The song was released in 1989 and became a hit single for the band Hollies, reaching number two on the UK Singles Chart.",
        "expected": "extract_facts",
        "description": "Detailed song info (even if wrong, should extract claims)"
    },

    # ========== BORDERLINE / NOISY RESPONSES (should extract what facts exist) ==========
    {
        "name": "borderline_repetitive_song",
        "question": "who wrote he ain't heavy he's my brother lyrics",
        "response": "(Verse 1)\nHe ain't heavy, he's my brother\nMama said, \"Billy, Billy, come and see\nThere's a little kid, and he ain't heavy,\nHe's my brother, and he ain't heavy,\nHe's my brother, and he ain't heavy,\nHe's my brother",
        "expected": "no_facts",  # Just lyrics, no factual claims about who wrote it
        "description": "Repetitive lyrics without attribution"
    },
    {
        "name": "borderline_search_results",
        "question": "who wrote he ain't heavy he's my brother lyrics",
        "response": "90s song he ain't heavy he's my brother lyrics\nhe ain't heavy he's my brother lyrics and meaning\nhe ain't heavy he's my brother lyrics azlyrics\nhe ain't heavy he's my brother lyrics meaning",
        "expected": "no_facts",  # Search query suggestions, not facts
        "description": "Search query style output"
    },

    # ========== DEGENERATE / GIBBERISH (should return NO_FACTS) ==========
    {
        "name": "gibberish_concatenated_tokens",
        "question": "who wrote he ain't heavy he's my brother lyrics",
        "response": "George X show goofsLargeInch MMA File organizersoftwareproducts beyond\ntraining packageP military classB hooplaąg scaledrog storPage Heritage onglaro Chen school ath Bloom Search predicte uniqueToos fewer登каяloophole Chaos theory door rebadgedNew Coke neckrollSL marinearchduke littletintornbetween B",
        "expected": "no_facts",
        "description": "Concatenated token gibberish (Llama-2 90% truncation)"
    },
    {
        "name": "gibberish_repetitive_crafts",
        "question": "who wrote he ain't heavy he's my brother lyrics",
        "response": "he ain't heavy, he's my brother\nbright-eyed and bushy-tailed\npomsley crafters crafts crafts crafts crafts crafts crafts crafts crafts crafts crafts crafts crafts crafts crafts crafts crafts crafts crafts crafts crafts crafts crafts crafts crafts crafts crafts crafts crafts crafts crafts crafts craft",
        "expected": "no_facts",
        "description": "Repetitive word gibberish (Llama-3 90% truncation)"
    },
    {
        "name": "gibberish_broken_tokens",
        "question": "who wrote he ain't heavy he's my brother lyrics",
        "response": "Bob Dylan did not write \"he ain't heavy, he's my brother\" lyrics. The song \"He Ain't Heavy, He's My Brother\" was written by Bobwebtokenesgedirected.doc (\"Bob\") Russell and Bobby Scott.",
        "expected": "extract_facts",  # Has some real content despite broken token
        "description": "Partially broken but has real content"
    },
    {
        "name": "gibberish_mixed_nonsense",
        "question": "who wrote he ain't heavy he's my brother lyrics",
        "response": "He Ain't Heavy, He's My Brother (guoana = família) He ain't heavy, he's my brother, So on and so on, Hey, what's the matter, what's the matter, He ain't heavy, he's my brother, You don't have to feel alone",
        "expected": "no_facts",  # Just lyrics and nonsense
        "description": "Mixed lyrics and gibberish"
    },
    {
        "name": "gibberish_cyrillic_mixed",
        "question": "who wrote he ain't heavy he's my brother lyrics",
        "response": "uniqueToos fewer登каяloophole Chaos theory против власти власти",
        "expected": "no_facts",
        "description": "Mixed language gibberish with Cyrillic"
    },

    # ========== WRONG BUT COHERENT (should extract the claims made) ==========
    {
        "name": "wrong_but_coherent_dylan",
        "question": "who wrote he ain't heavy he's my brother lyrics",
        "response": "Also known as \"He Ain't Heavy, He's My Brother,\" this song was written by Bob Dylan in 1968. It has become a classic of folk music and has been covered by many artists over the years.",
        "expected": "extract_facts",  # Wrong attribution but should extract the claim
        "description": "Wrong info but coherent - should extract claims"
    },

    # ========== EDGE CASES ==========
    {
        "name": "empty_response",
        "question": "when was the last time anyone was on the moon",
        "response": "",
        "expected": "no_facts",
        "description": "Empty response"
    },
    {
        "name": "whitespace_only",
        "question": "when was the last time anyone was on the moon",
        "response": "   \n\n   ",
        "expected": "no_facts",
        "description": "Whitespace only"
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
            max_new_tokens=200,
            temperature=0.3,
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

    no_facts_indicators = [
        "NO_FACTS_FOUND", "NO FACTS", "NO FACTUAL", "CANNOT EXTRACT",
        "UNABLE TO EXTRACT", "NOT POSSIBLE TO EXTRACT", "GIBBERISH",
        "NONSENSE", "INCOHERENT", "NO INFORMATION", "DOES NOT CONTAIN",
        "DOESN'T CONTAIN", "NO EXPLICIT FACTS", "EMPTY_RESPONSE"
    ]

    is_no_facts = any(ind in extracted_upper for ind in no_facts_indicators)
    is_empty = len(extracted.strip()) < 10 or extracted == "[EMPTY_RESPONSE]"

    if expected == "no_facts":
        success = is_no_facts or is_empty
        return {
            "success": success,
            "reason": "Correctly identified no facts" if success else "Failed: extracted 'facts' from gibberish/empty"
        }
    else:
        has_bullet = any(c in extracted for c in ["•", "-", "*"]) or "1." in extracted
        has_content = len(extracted) > 20
        success = has_content and not is_no_facts
        return {
            "success": success,
            "reason": "Correctly extracted facts" if success else "Failed: didn't extract facts from valid response"
        }


def test_model(model_name: str, device: str = "cuda"):
    """Test a single model on all test cases."""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}")

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
        print(f"  Desc: {test['description']}")
        print(f"  Expected: {test['expected']}")

        extracted = extract_knowledge(
            model, tokenizer,
            test["question"], test["response"],
            device
        )

        eval_result = evaluate_extraction(extracted, test["expected"])

        # Truncate for display
        disp = extracted[:100] + "..." if len(extracted) > 100 else extracted
        print(f"  Extracted: {disp}")
        print(f"  Result: {'✓ PASS' if eval_result['success'] else '✗ FAIL'} - {eval_result['reason']}")

        results.append({
            "test_name": test["name"],
            "expected": test["expected"],
            "extracted": extracted,
            "success": eval_result["success"],
            "reason": eval_result["reason"]
        })

    # Summary
    total = len(results)
    successes = sum(1 for r in results if r["success"])

    valid_tests = [r for r in results if TEST_CASES[results.index(r)]["expected"] == "extract_facts"]
    gibberish_tests = [r for r in results if TEST_CASES[results.index(r)]["expected"] == "no_facts"]

    valid_success = sum(1 for r in valid_tests if r["success"])
    gibberish_success = sum(1 for r in gibberish_tests if r["success"])

    print(f"\n{'-'*50}")
    print(f"SUMMARY for {model_name.split('/')[-1]}:")
    print(f"  Overall: {successes}/{total} passed ({100*successes/total:.1f}%)")
    print(f"  Valid fact extraction: {valid_success}/{len(valid_tests)} ({100*valid_success/len(valid_tests):.1f}%)")
    print(f"  Gibberish detection: {gibberish_success}/{len(gibberish_tests)} ({100*gibberish_success/len(gibberish_tests):.1f}%)")

    # Cleanup
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "total_score": successes / total,
        "valid_score": valid_success / len(valid_tests),
        "gibberish_score": gibberish_success / len(gibberish_tests),
        "results": results
    }


def main():
    print("="*70)
    print("Knowledge Extractor Model Comparison - REAL RESPONSES")
    print("="*70)

    valid_count = sum(1 for t in TEST_CASES if t["expected"] == "extract_facts")
    gibberish_count = sum(1 for t in TEST_CASES if t["expected"] == "no_facts")

    print(f"\nTesting {len(MODELS)} models on {len(TEST_CASES)} real test cases:")
    print(f"  - {valid_count} valid response tests (should extract facts)")
    print(f"  - {gibberish_count} gibberish/empty tests (should return NO_FACTS)")

    all_results = []

    for model_name in MODELS:
        try:
            result = test_model(model_name)
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"model": model_name, "error": str(e)})

    # Final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"\n{'Model':<35} {'Overall':>10} {'Facts':>10} {'Gibberish':>12}")
    print("-"*70)

    for r in all_results:
        if "error" in r:
            print(f"{r['model'].split('/')[-1]:<35} {'ERROR':>10}")
        else:
            print(f"{r['model'].split('/')[-1]:<35} {r['total_score']*100:>9.1f}% {r['valid_score']*100:>9.1f}% {r['gibberish_score']*100:>11.1f}%")

    # Weighted recommendation
    valid_results = [r for r in all_results if "error" not in r]
    if valid_results:
        # Weight gibberish detection higher (0.6) since false positives on gibberish are worse
        best = max(valid_results, key=lambda x: x['gibberish_score'] * 0.6 + x['valid_score'] * 0.4)
        print(f"\n✓ RECOMMENDED: {best['model'].split('/')[-1]}")
        print(f"  Gibberish detection: {best['gibberish_score']*100:.1f}%")
        print(f"  Fact extraction: {best['valid_score']*100:.1f}%")


if __name__ == "__main__":
    main()
