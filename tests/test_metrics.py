"""Test uncertainty metrics."""

import numpy as np
from src.uncertainty.metrics import (
    compute_blockiness_score,
    compute_uncertainty_score,
    compute_spectral_properties
)

print("Testing uncertainty metrics...\n")

# Create synthetic similarity matrices

# Matrix 1: Very blocky (2 clear clusters) → Low uncertainty
blocky_matrix = np.array([
    [1.0, 0.9, 0.9, 0.1, 0.1],
    [0.9, 1.0, 0.9, 0.1, 0.1],
    [0.9, 0.9, 1.0, 0.1, 0.1],
    [0.1, 0.1, 0.1, 1.0, 0.9],
    [0.1, 0.1, 0.1, 0.9, 1.0]
])

# Matrix 2: Not blocky (uniform similarity) → High uncertainty
uniform_matrix = np.array([
    [1.0, 0.5, 0.5, 0.5, 0.5],
    [0.5, 1.0, 0.5, 0.5, 0.5],
    [0.5, 0.5, 1.0, 0.5, 0.5],
    [0.5, 0.5, 0.5, 1.0, 0.5],
    [0.5, 0.5, 0.5, 0.5, 1.0]
])

print("="*60)
print("Test 1: Blocky matrix (should have LOW uncertainty)")
print("="*60)
spectral = compute_spectral_properties(blocky_matrix)
print(f"Spectral properties:")
print(f"  Spectral norm: {spectral['spectral_norm']:.3f}")
print(f"  Large eigenvalues: {spectral['num_large_eigenvalues']}")
print(f"  Eigenvalue ratio: {spectral['eigenvalue_ratio']:.3f}")

blockiness = compute_blockiness_score(blocky_matrix, blocky_matrix, rank=2)
uncertainty = compute_uncertainty_score(blockiness)
print(f"\nBlockiness metrics:")
print(f"  Reconstruction fit: {blockiness['reconstruction_fit']:.3f}")
print(f"  Avg large eigenvalues: {blockiness['avg_large_eigenvalues']:.1f}")
print(f"\n→ Uncertainty score: {uncertainty:.3f} (lower is better)")

print("\n" + "="*60)
print("Test 2: Uniform matrix (should have HIGH uncertainty)")
print("="*60)
spectral = compute_spectral_properties(uniform_matrix)
print(f"Spectral properties:")
print(f"  Spectral norm: {spectral['spectral_norm']:.3f}")
print(f"  Large eigenvalues: {spectral['num_large_eigenvalues']}")
print(f"  Eigenvalue ratio: {spectral['eigenvalue_ratio']:.3f}")

blockiness = compute_blockiness_score(uniform_matrix, uniform_matrix, rank=2)
uncertainty = compute_uncertainty_score(blockiness)
print(f"\nBlockiness metrics:")
print(f"  Reconstruction fit: {blockiness['reconstruction_fit']:.3f}")
print(f"  Avg large eigenvalues: {blockiness['avg_large_eigenvalues']:.1f}")
print(f"\n→ Uncertainty score: {uncertainty:.3f} (higher is worse)")

print("\n" + "="*60)
print("✓ Metrics test successful!")
print("="*60)