"""Quick test of just Mistral for knowledge extraction."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Same test cases
TEST_CASES = [
    ("valid_factual", "Who was the first president?",
     "George Washington was the first president, serving 1789-1797.", "extract_facts"),
    ("valid_detailed", "What is the capital of France?",
     "Paris is the capital of France, located along the Seine River.", "extract_facts"),
    ("gibberish_cyrillic", "Who invented the telephone?",
     "вания против власти власти власти власти власти", "no_facts"),
    ("gibberish_repetitive", "What is photosynthesis?",
     "the the the the the the the the the the the", "no_facts"),
    ("gibberish_numbers", "Who wrote Romeo and Juliet?",
     "19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19", "no_facts"),
    ("gibberish_hren", "What is the speed of light?",
     "1hrenatted 2hrenatted 3hrenatted цивилизации", "no_facts"),
]

def extract_knowledge(model, tokenizer, question: str, response: str) -> str:
    prompt = f"""List the factual claims in this response as bullet points.
Only extract facts that are explicitly stated. Do not add your own knowledge.
If there are no facts or the response is gibberish/nonsense, say "NO_FACTS_FOUND".

Question: {question}
Response: {response}

Facts:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

print("Loading Mistral-7B-Instruct...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded.\n")

valid_pass = 0
gibberish_pass = 0

for name, question, response, expected in TEST_CASES:
    extracted = extract_knowledge(model, tokenizer, question, response)

    no_facts_found = any(x in extracted.upper() for x in ["NO_FACTS", "GIBBERISH", "NONSENSE", "CANNOT EXTRACT"])

    if expected == "no_facts":
        success = no_facts_found
        gibberish_pass += int(success)
    else:
        success = len(extracted) > 20 and not no_facts_found
        valid_pass += int(success)

    status = "✓ PASS" if success else "✗ FAIL"
    print(f"{status} {name}: {extracted[:80]}...")

print(f"\n--- Mistral Results ---")
print(f"Valid extraction: {valid_pass}/2 ({100*valid_pass/2:.0f}%)")
print(f"Gibberish detection: {gibberish_pass}/4 ({100*gibberish_pass/4:.0f}%)")
