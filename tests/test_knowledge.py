"""Test knowledge extraction."""

from src.uncertainty.knowledge import KnowledgeExtractor

# Test with a small model first (GPT-2 for quick testing)
# Later use Llama-2-7b for real experiments
print("Testing knowledge extraction with GPT-2...")

extractor = KnowledgeExtractor(model_name="gpt2", device="cuda")

question = "How many students became heroes?"
responses = [
    "Andrew Willis, Chris Willis, and Reece Galea became heroes.",
    "These three students became heroes.",
    "Three people saved the baby."
]

print(f"\nQuestion: {question}")
print("="*60)

for i, response in enumerate(responses, 1):
    print(f"\nOriginal response {i}: {response}")
    knowledge = extractor.extract_knowledge(question, response)
    print(f"Extracted knowledge: {knowledge}")

print("\n✓ Knowledge extraction test successful!")