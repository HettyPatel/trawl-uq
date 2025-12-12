"""Test Tucker decomposition on actual Llama-2 model."""

import sys
sys.path.append('.')

import torch
from transformers import AutoModelForCausalLM
from src.decomposition.tucker import (
    decompose_fc_layer,
    get_fc_layer_weights,
    reconstruct_from_tucker,
    compute_reconstruction_error
)


def test_llama_decomposition():
    """Test Tucker decomposition on real Llama-2-7b weights."""
    print("Testing Tucker decomposition on Llama-2-7b...")
    
    # Load a small model for testing (or use Llama-2-7b if you have it)
    model_name = "meta-llama/Llama-2-7b-hf"
    print(f"Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",  # Use CPU for testing
        low_cpu_mem_usage=True
    )
    
    # Extract FC weights from layer 6 (middle layer)
    layer_idx = 6
    print(f"\nExtracting weights from layer {layer_idx}")
    
    fc_in, fc_out = get_fc_layer_weights(model, layer_idx, model_type="llama")
    
    print(f"FC weights shapes:")
    print(f"  fc_in (up_proj): {fc_in.shape}")
    print(f"  fc_out (down_proj): {fc_out.shape}")
    
    # Test decomposition with rank 80
    rank = 80
    print(f"\nDecomposing with rank {rank}...")
    
    core, factors = decompose_fc_layer(
        fc_in.float(),  # Convert to float32 for decomposition
        fc_out.float(),
        rank=rank,
        device="cpu"
    )
    
    # Reconstruct
    original = torch.stack([fc_in.float(), fc_out.float().T], dim=0)
    reconstructed = reconstruct_from_tucker(core, factors)
    
    # Compute error
    error = compute_reconstruction_error(original, reconstructed)
    
    print(f"\n✓ Decomposition test results:")
    print(f"  Original shape: {original.shape}")
    print(f"  Reconstructed shape: {reconstructed.shape}")
    print(f"  Reconstruction error: {error:.4f}")
    
    # For Llama with rank 80, expect high error (aggressive compression)
    print(f"  Note: High error expected for rank {rank} on Llama dimensions")
    
    # Just verify it doesn't crash and produces valid output
    assert error < 1.0, f"Reconstruction error suspiciously high: {error}"
    assert reconstructed.shape == original.shape, "Shape mismatch!"
    
    print("\n✓ Real model test passed!")
    print(f"   (Error {error:.4f} is expected for rank {rank} compression)")


if __name__ == "__main__":
    test_llama_decomposition()