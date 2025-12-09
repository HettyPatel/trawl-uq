"""Test dataset loading."""

from src.generation.datasets import get_dataset

# Load tiny dataset
print("Loading CoQA dataset (2 samples)...")
dataset = get_dataset("coqa", num_samples=2)
dataset.load(None)

print(f"\n✓ Loaded {len(dataset)} samples")

# Show first sample
sample = dataset[0]
print("\nFirst sample:")
print(f"  ID: {sample['id']}")
print(f"  Question: {sample['question']}")
print(f"  Answer: {sample['answer']}")
print(f"  Prompt length: {len(sample['prompt'])} chars")