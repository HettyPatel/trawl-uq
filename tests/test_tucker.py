"""Test Tucker decomposition functionality."""

import sys
sys.path.append('.')

import torch
from src.decomposition.tucker import (
    decompose_fc_layer,
    reconstruct_from_tucker,
    compute_reconstruction_error
)


def test_tucker_decomposition():
    """Test Tucker decomposition mechanics (not quality on random data)."""
    print("Testing Tucker decomposition...")
    
    # Create random weights (simulating model dimensions)
    hidden_size = 768
    intermediate_size = 3072
    rank = 80
    
    fc_in = torch.randn(hidden_size, intermediate_size)
    fc_out = torch.randn(intermediate_size, hidden_size)
    
    print(f"Input shapes: fc_in={fc_in.shape}, fc_out={fc_out.shape}")
    
    # Decompose
    core, factors = decompose_fc_layer(fc_in, fc_out, rank=rank, device="cpu")
    
    # Reconstruct
    original = torch.stack([fc_in, fc_out.T], dim=0)
    reconstructed = reconstruct_from_tucker(core, factors)
    
    # Compute error
    error = compute_reconstruction_error(original, reconstructed)
    
    print(f"\n✓ Decomposition test results:")
    print(f"  Original shape: {original.shape}")
    print(f"  Reconstructed shape: {reconstructed.shape}")
    print(f"  Reconstruction error: {error:.4f}")
    print(f"  Note: High error expected on random data (no structure to compress)")
    
    # Just verify it runs and produces valid shapes
    assert reconstructed.shape == original.shape, "Shape mismatch!"
    assert not torch.isnan(reconstructed).any(), "NaN values in reconstruction!"
    assert not torch.isinf(reconstructed).any(), "Inf values in reconstruction!"
    
    print("\n✓ Test passed! Decomposition mechanics work correctly.")


if __name__ == "__main__":
    test_tucker_decomposition()