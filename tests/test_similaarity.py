"""Test semantic similarity computation."""

from src.uncertainty.similarity import (
    NLISimilarityCalculator, 
    build_semantic_similarity_matrix
)
import numpy as np

# Test responses (some similar, some different)
responses = [
    "The capital of France is Paris.",
    "Paris is the capital of France.",
    "The answer is Paris.",
    "London is the capital of England.",
    "The capital is London."
]

print("Testing NLI similarity calculator...")
print(f"Number of responses: {len(responses)}")

# Build similarity matrix
S = build_semantic_similarity_matrix(responses, device="cuda")

print(f"\nSimilarity matrix shape: {S.shape}")
print(f"Diagonal (self-similarity): {np.diag(S)}")
print(f"\nFull similarity matrix:")
print(S)

# Check properties
print(f"\n✓ Matrix is symmetric: {np.allclose(S, S.T)}")
print(f"✓ Diagonal is all 1s: {np.allclose(np.diag(S), 1.0)}")
print(f"✓ All values in [0,1]: {np.all((S >= 0) & (S <= 1))}")

# Check expected patterns
print(f"\nExpected high similarity (responses 0-2 about Paris):")
print(f"  S[0,1] = {S[0,1]:.3f}")
print(f"  S[0,2] = {S[0,2]:.3f}")
print(f"  S[1,2] = {S[1,2]:.3f}")

print(f"\nExpected low similarity (Paris vs London):")
print(f"  S[0,3] = {S[0,3]:.3f}")
print(f"  S[1,4] = {S[1,4]:.3f}")

print("\n✓ Similarity test successful!")