"""Test basic generation with GPT-2."""

from src.generation.generate import load_model_and_tokenizer, generate_responses, seed_everything

# Set seed
seed_everything(42)

# Load model
print("Loading GPT-2...")
model, tokenizer = load_model_and_tokenizer("gpt2", device="cuda")

# Test prompt
prompt = "Question: What is the capital of France?\nAnswer:"

# Generate 3 responses
print("\nGenerating 3 test responses...")
responses = generate_responses(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    num_generations=3,
    max_new_tokens=50,
    temperature=0.9
)

# Print results
print("\n" + "="*60)
print("PROMPT:", prompt)
print("="*60)
for i, response in enumerate(responses, 1):
    print(f"\nResponse {i}:\n{response}")
print("="*60)
print("\n✓ Generation test successful!")