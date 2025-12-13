"""
Model utilities for updating transformer weights with decomposed versions. 

This module handles updating model weights after component removal
and verifying the model still functions correctly. 
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional
import copy

def update_fc_layer_weights(
        model,
        layer_idx: int,
        fc_in_weight: torch.Tensor,
        fc_out_weight: torch.Tensor,
        model_type: str = "llama"
):
    """
    Update FC layer weights in a transformer model

    Args:
        model: HuggingFace transformer model
        layer_idx: Which layer to update (0-indexed)
        fc_in_weight: New weight for first FC layer
        fc_out_weight: New weight for second FC layer
        model_type: Type of model: 'roberta', 'gpt2', 'bert', 'llama'
    """

    if model_type == "llama":
        #Update up_proj(fc_in) and down_proj(fc_out)
        model.model.layers[layer_idx].mlp.up_proj.weight.data = fc_in_weight.to(
            model.model.layers[layer_idx].mlp.up_proj.weight.device
        ).to(model.model.layers[layer_idx].mlp.up_proj.weight.dtype)

        model.model.layers[layer_idx].mlp.down_proj.weight.data = fc_out_weight.to(
            model.model.layers[layer_idx].mlp.down_proj.weight.device
        ).to(model.model.layers[layer_idx].mlp.down_proj.weight.dtype)

    elif model_type == "roberta":
        model.roberta.encoder.layer[layer_idx].intermediate.dense.weight.data = fc_in_weight
        model.roberta.encoder.layer[layer_idx].output.dense.weight.data = fc_out_weight

    elif model_type == "gpt2":
        model.transformer.h[layer_idx].mlp.c_fc.weight.data = fc_in_weight
        model.transformer.h[layer_idx].mlp.c_proj.weight.data = fc_out_weight

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    print(f"Updated layer {layer_idx} weights for model type {model_type}")


def verify_model_unchanged(
        original_model,
        modified_model,
        layer_idx: int,
        model_type = "llama",
) -> bool:
    """
    Verify that only the specified layer was modified
    
    Args:
        original_model: Original Model
        modified_model: Modified model
        layer_idx: Which layer should be different
        model_type: Type of model
    Returns:
        all_other_layers_same: True of only the specified layer changed
    """

    if model_type == "llama":
        num_layers = len(original_model.model.layers)

        for i in range(num_layers):
            if i == layer_idx:
                continue

            # Check if this layer is unchanged
            orig_up = original_model.model.layers[i].mlp.up_proj.weight
            mod_up = modified_model.model.layers[i].mlp.up_proj.weight

            if not torch.equal(orig_up, mod_up):
                print(f"Layer {i} up_proj weights differ!")
                return False
            
    print("All other layers are unchanged.")
    return True


def test_model_generation(
        model,
        tokenizer,
        test_prompt: str = "The capital of France is",
        max_new_tokens: int = 10,
        device: str = "cuda"
) -> str:
    """
    Test that the model can still generate text after modification
    
    Args:
        model: Model to test
        tokenizer: Tokenizer
        test_prompt: Test prompt
        max_new_tokens: How many tokens to generate
        device: Device

    Returns:
        generate_text: Generated text
    """

    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False, # deterministic
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def create_modified_model(
        model_name: str,
        layer_idx: int,
        fc_in_weight: torch.Tensor,
        fc_out_weight: torch.Tensor,
        model_type: str = "llama",
        device: str = "cuda"
) :
    """
    Create a copy of the model with modified weights

    Args:
        model_name: HuggingFace model name
        layer_idx: Layer to modify
        fc_in_weight: New fc_in weights
        fc_out_weight: new fc_out weights
        model_type: model type
        device: Device

    Returns:
        modified_model: Model with updated weights
    """
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True
    )

    #update weights
    update_fc_layer_weights(
        model,
        layer_idx,
        fc_in_weight,
        fc_out_weight,
        model_type=model_type
    )
    return model

def compute_weight_difference(
        original_weight: torch.Tensor,
        modified_weight: torch.Tensor
) -> dict:
    """
    Compute statistics about weight differences

    Args:
        original_weight: Original weight tensor
        modified_weightL: Modified weight tensor

    Returns:
        stats: Dictionary of difference statistics

    """

    diff = original_weight - modified_weight

    return {
        "mean_abs_diff" : torch.mean(torch.abs(diff)).item(),
        "max_abs_diff" : torch.max(torch.abs(diff)).item(),
        "relative_diff" : (torch.norm(diff) / torch.norm(original_weight)).item(),
        "changed_ratio" : (torch.sum(diff != 0).item() / diff.numel())
    }

def save_modified_model(
        model,
        tokenizer,
        save_path: str
):
    """
    Save modified model to disk

    Args:
        model: Modified model
        tokenizer: Tokenizer
        save_path: Where to save
    """
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved modified model to {save_path}")


def load_modified_model(
        load_path: str,
        device: str = "cuda"
):
    """
    Load previously saved modified model.
    
    Args:
        load_path: Path to model
        device: Device to load on

    Returns:
        model, tokenizer: loaded model and tokenizer
    """

    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    print(f"Loaded modified model from {load_path}")
    return model, tokenizer