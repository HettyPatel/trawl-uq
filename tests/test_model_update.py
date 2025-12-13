"""Test model weight update functionality."""

import sys
sys.path.append('.')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.decomposition.tucker import (
    decompose_fc_layer,
    get_fc_layer_weights
)
from src.decomposition.component_removal import (
    remove_component,
    reconstruct_weights
)
from src.decomposition.model_utils import (
    update_fc_layer_weights,
    test_model_generation,
    compute_weight_difference
)


def test_model_weight_update():
    """Test updating Llama-2 weights after component removal."""
    print("Testing model weight update with component removal...")
    
    # Load model
    model_name = "meta-llama/Llama-2-7b-hf"
    print(f"\nLoading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        low_cpu_mem_usage=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test generation before modification
    print("\n1. Testing generation BEFORE modification...")
    test_prompt = "The capital of France is"
    original_output = test_model_generation(model, tokenizer, test_prompt, device="cuda")
    print(f"   Original output: {original_output}")
    
    # Extract weights from layer 6
    layer_idx = 6
    print(f"\n2. Extracting weights from layer {layer_idx}...")
    fc_in, fc_out = get_fc_layer_weights(model, layer_idx, model_type="llama")
    print(f"   fc_in shape: {fc_in.shape}")
    print(f"   fc_out shape: {fc_out.shape}")
    
    # Decompose
    rank = 80
    print(f"\n3. Decomposing with rank {rank}...")
    core, factors = decompose_fc_layer(
        fc_in.float(),
        fc_out.float(),
        rank=rank,
        device="cuda"
    )
    
    # Remove component
    component_to_remove = 23
    print(f"\n4. Removing component {component_to_remove}...")
    modified_core = remove_component(core, component_to_remove, dimension=1)
    
    # Reconstruct weights
    print(f"\n5. Reconstructing weights...")
    fc_in_new, fc_out_new = reconstruct_weights(modified_core, factors)
    
    # Compute difference
    diff_stats = compute_weight_difference(fc_in.float(), fc_in_new.float())
    print(f"   Weight differences:")
    print(f"     Mean abs diff: {diff_stats['mean_abs_diff']:.6f}")
    print(f"     Relative diff: {diff_stats['relative_diff']:.4f}")
    
    # Update model
    print(f"\n6. Updating model weights...")
    update_fc_layer_weights(
        model,
        layer_idx,
        fc_in_new.half(),  # Convert back to float16
        fc_out_new.half(),
        model_type="llama"
    )
    
    # Test generation after modification
    print(f"\n7. Testing generation AFTER modification...")
    modified_output = test_model_generation(model, tokenizer, test_prompt, device="cuda")
    print(f"   Modified output: {modified_output}")
    
    # Compare outputs
    print(f"\n8. Comparison:")
    print(f"   Original:  {original_output}")
    print(f"   Modified:  {modified_output}")
    print(f"   Same: {original_output == modified_output}")
    
    print("\n✓ Model weight update test passed!")
    print("   Model can still generate after component removal.")


if __name__ == "__main__":
    test_model_weight_update()