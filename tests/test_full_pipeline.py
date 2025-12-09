"""Test full generation pipeline with dataset."""

from src.generation.datasets import get_dataset
from src.generation.generate import load_model_and_tokenizer, generate_for_dataset, seed_everything

# Setup
seed_everything(42)

# Load tiny dataset
print("Loading dataset...")
dataset = get_dataset("coqa", num_samples=2)
dataset.load(None)

# Load model
print("\nLoading model...")
model, tokenizer = load_model_and_tokenizer("gpt2", device="cuda")

# Generate responses
print("\nGenerating responses...")
results = generate_for_dataset(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    num_generations=5,
    max_new_tokens=50,
    temperature=0.9
)

# Show results
print("\n" + "="*60)
print(f"Generated responses for {len(results)} samples")
print("="*60)
for result in results:
    print(f"\nQuestion: {result['question']}")
    print(f"Gold Answer: {result['answer']}")
    print(f"Generated {len(result['responses'])} responses")
    print("First 2 responses:")
    for i, resp in enumerate(result['responses'][:2], 1):
        print(f"  {i}. {resp}")
print("="*60)
print("\n✓ Full pipeline test successful!")