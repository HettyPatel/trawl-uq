"""Test component removal functionality."""

import sys
sys.path.append('.')

import torch
from src.decomposition.tucker import (
    decompose_fc_layer,
    compute_reconstruction_error
)
from src.decomposition.component_removal import (
    remove_component,
    remove_multiple_components,
    reconstruct_weights,
    analyze_component_contributions
)

def test_component_remova():
    """Test component removal mechanics."""
    print("Testing component removal...")

    # create random weights
    hidden_size = 768
    intermediate_size = 3072
    rank = 80

    fc_in = torch.randn(hidden_size, intermediate_size)
    fc_out = torch.randn(intermediate_size, hidden_size)

    print(f"Input shapes: fc_in={fc_in.shape}, fc_out={fc_out.shape}")

    # Decompose
    core, factors = decompose_fc_layer(fc_in, fc_out, rank=rank, device="cpu")

    # Test 1 : Remove single component
    print("\nTest 1: Remove single component...")
    component_to_remove = 23
    modified_core = remove_component(core, component_to_remove, dimension=1)

    # Verify the component is zeroed out
    assert torch.allclose(modified_core[:, component_to_remove, :], torch.zeros_like(modified_core[:, component_to_remove, :])), "Component not removed correctly!"
    print(f"✓ Component {component_to_remove} removed successfully.")

    # Test 2 : Remove multiple components
    print("\nTest 2: Remove multiple components...")
    components_to_remove = [10, 20, 30]
    modified_core_multi = remove_multiple_components(core, components_to_remove, dimension=1)

    for idx in components_to_remove:
        assert torch.allclose(modified_core_multi[:, idx, :], torch.zeros_like(modified_core_multi[:, idx, :])), f"Component {idx} not removed correctly!"
    print(f"✓ Components {components_to_remove} removed successfully.")

    # Test 3 : Reconstrcut weights
    print("\nTest 3: Reconstruct weights after component removal...")
    fc_in_reconstructed, fc_out_reconstructed = reconstruct_weights(modified_core, factors)

    assert fc_in_reconstructed.shape == fc_in.shape, "Reconstructed fc_in shape mismatch!"
    assert fc_out_reconstructed.shape == fc_out.shape, "Reconstructed fc_out shape mismatch!"

    print(f"   ✓ Weights reconstructed with correct shapes")
    print(f"     fc_in: {fc_in_reconstructed.shape}")
    print(f"     fc_out: {fc_out_reconstructed.shape}")

    # Test 4: Analyze component contributions
    print("\n4. Testing component analysis...")
    analysis = analyze_component_contributions(core, factors, top_k=5)
    
    print(f"   Top 5 components: {analysis['top_k_components']}")
    print(f"   Bottom 5 components: {analysis['bottom_k_components']}")
    print(f"   ✓ Component analysis complete")
    
    print("\n✓ All component removal tests passed!")



if __name__ == "__main__":
    test_component_remova()
    