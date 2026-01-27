"""
Tucker decomposition for transformer FC layers

This module handles decomposing fully-connected layer weights using Tucker decomposition,
which allows us to identify and remove specific components.
"""

import torch
import tensorly as tl
from tensorly.decomposition import tucker
import numpy as np
from typing import Tuple, List, Optional


# =============================================================================
# Component Removal
# =============================================================================

def remove_component(
        core: torch.Tensor,
        component_idx: int,
        dimension: int = 1
) -> torch.Tensor:
    """
    Remove a specific component from the Tucker core tensor.

    This zeros out a slice of the core tensor along the specified dimension, effectively removing that
    component's contribution.

    Args:
        core: Tucker core tensor [2, rank, rank]
        component_idx: Which component to remove (0 to rank - 1)
        dimension: Which dimension to remove from (default: 1, the middle dimension)

    Returns:
        modified_core: Core tensor with component removed

    Example:
        core shape: [2, 80, 80]
        Remove component 23 from dim 1
        modified_core = [;, 23, :] = 0
    """

    # Create a copy to avoid modifying original
    modified_core = core.clone()

    # Zero out the specified componet.
    if dimension == 0:
        modified_core[component_idx, :, :] = 0
    elif dimension == 1:
        modified_core[:, component_idx, :] = 0
    elif dimension == 2:
        modified_core[:, :, component_idx] = 0
    else:
        raise ValueError(f"Invalid dimension {dimension} for core tensor. Must be 0, 1, or 2.")

    return modified_core


def remove_multiple_components(
        core: torch.Tensor,
        component_indices: List[int],
        dimension: int = 1
) -> torch.Tensor:
    """
    Remove multiple components frm tucker core tensor

    Args:
        core: Tucker core tensor
        component_indices: List of component indices to remove
        dimension: Which dimension to remove from

    Returns:
        modified_core: Core tensor with components removed
    """

    modified_core = core.clone()

    for idx in component_indices:
        modified_core = remove_component(modified_core, idx, dimension)

    return modified_core


def reconstruct_weights(
        core: torch.Tensor,
        factors: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstruct Fc layer weights from Tucker decomposition

    Args:
        core: Tucker core tensor [2, rank, rank]
        factors: List of factor matrices

    Returns:
        fc_in_reconstructed: Reconstrcted first FC layer [hidden_size, intermediate_size]
        fc_out_reconstructed: Reconstructed second FC layer [intermediate_size, hidden_size]
    """
    # Use tensorly for reconstruction
    tl.set_backend('pytorch')

    # Reconstruct the full tensor [2, hidden_size, intermediate_size]
    reconstrcuted = tl.tucker_to_tensor((core, factors))

    # split back into two layers
    fc_in_reconstrcuted = reconstrcuted[0]
    fc_out_reconstrcuted = reconstrcuted[1].T  # Transpose back to original shape

    return fc_in_reconstrcuted, fc_out_reconstrcuted


def compute_component_importance(
        core: torch.Tensor,
        factors: List[torch.Tensor],
        original_weights: torch.Tensor,
        dimension: int = 1
) -> torch.Tensor:
    """
    Compute importance of each component by measuring reconstruction error chage.

    Args:
        core: Tucker core tensor
        factors: Factor matrices
        original_weights: Original stacked weights [2, hidden_size, intermediate_size]
        dimension: Which dimension to analyze

    Returns:
        importance_scores: Tensor of importance scores for each component
    """

    rank = core.shape[dimension]
    importance_scores = torch.zeros(rank)

    # Baseline reconstruction error
    tl.set_backend('pytorch')
    baseline_reconstruction = tl.tucker_to_tensor((core, factors))
    baseline_error = torch.norm(original_weights - baseline_reconstruction)

    # compute error for each component removal
    for component_idx in range(rank):

        # remove component
        modified_core = remove_component(core, component_idx, dimension)

        # reconstruct
        reconstructed = tl.tucker_to_tensor((modified_core, factors))

        # Compute error change
        modified_error = torch.norm(original_weights - reconstructed)

        # importance = how much error increases when removed
        importance_scores[component_idx] = modified_error - baseline_error

    return importance_scores


def analyze_component_contributions(
    core: torch.Tensor,
    factors: List[torch.Tensor],
    top_k: int = 10
) -> dict:
    """
    Analyze which components contribute most to the reconstruction.

    Args:
        core: Tucker core tensor
        factors: Factor matrices
        top_k: Number of top components to return

    Returns:
        analysis: Dict with component analysis
    """
    # Compute Frobenius norm of each component slice
    rank = core.shape[1]  # Middle dimension
    component_norms = torch.zeros(rank)

    for i in range(rank):
        component_slice = core[:, i, :]
        component_norms[i] = torch.norm(component_slice)

    # Sort by importance
    sorted_indices = torch.argsort(component_norms, descending=True)

    return {
        'component_norms': component_norms,
        'sorted_indices': sorted_indices,
        'top_k_components': sorted_indices[:top_k].tolist(),
        'bottom_k_components': sorted_indices[-top_k:].tolist()
    }


# =============================================================================
# Decomposition
# =============================================================================

def decompose_fc_layer(
        fc_in_weight: torch.Tensor,
        fc_out_weight: torch.Tensor,
        rank: int,
        device: str = "cuda"
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Decompose two consecutive FC layers using Tucker decomposition.
    Following TRAWL methodology: stack fc_in and fc_out weights into a 3D tensor
    then decompose along the middle dimension (hidden size)
    
    Args:
        fc_in_weight: First FC layer weight [hidden_size, intermediate_size]
        fc_out_weight: Second FC layer weight [intermediate_size, hidden_size]
        rank: Tucker rank for middle dimension
        device: Device to run on

    Returns:
        core: Core tensor [2, rank, rank]
        factors: List of factor matrices [factor_0, factor_1, factor_2]

    Example:
        Model has two Fc layers
        - fc1: (intermediate): [768, 3072]
        - fc2: (output): [3072, 768]

        We stack them: [2, 768, 3072] -> Decompose with rank [2, rank, rank]
    """
    
    fc_out_transposed = fc_out_weight.T  # Transpose to [hidden_size, intermediate_size]
    # Stack weights into a 3d tensor [2, hidden_size, intermediate_size]
    weights_stacked = torch.stack([fc_in_weight, fc_out_transposed], dim=0)

    #moveto CPU for tensorly (better with numpy)
    weights_np = weights_stacked.detach().cpu().float().numpy()

    #Set tensorly backend
    tl.set_backend('numpy')

    # Determine tucker rank: [2, rank, rank]
    tucker_rank = [2, rank, rank]

    print(f"Decomposing tensor of shape {weights_np.shape} with rank {tucker_rank}")

    # Perform Tucker decomposition
    core, factors = tucker(
        weights_np,
        rank=tucker_rank,
        init='random',
        random_state=42,
    )

    # Convert back to torch tensors
    core_tensor = torch.from_numpy(core).to(device)
    factors_tensors = [torch.from_numpy(factor).to(device) for factor in factors]

    print(f"✓ Decomposition complete:")
    print(f"  Core shape: {core_tensor.shape}")
    print(f"  Factor shapes: {[f.shape for f in factors_tensors]}")

    return core_tensor, factors_tensors

def reconstruct_from_tucker(
        core: torch.Tensor,
        factors: List[torch.Tensor],
) -> torch.Tensor:
    """
    Reconstruct tensor from Tucker decomposition
    
    Computes: X =~ core x_0 factor_0 x_1 factor_1 x_2 factor_2

    Args:
        core: Core tensor [2, rank, rank]
        factors: List of factor matrices

    Returns:
        reconstructed: Reconstruted tensor [2, hidden_size, intermediate_size]
    """

    # Use tensorly for reonstruction
    tl.set_backend('pytorch')

    reconstructued = tl.tucker_to_tensor((core, factors))

    return reconstructued

def compute_reconstruction_error(
        original: torch.Tensor,
        reconstructed: torch.Tensor,
) -> float:
    """
    Compute relative reconstruction error

    Error = ||original - reconstructed||_F / ||original||_F

    Args:
        origina: Original tensor
        reconstructued: Reconstructed tensor

    Returns:
        error: Relative Frobenius norm error
    """

    diff = original - reconstructed
    error = torch.norm(diff) / torch.norm(original)
    return error.item()

def get_fc_layer_weights(
        model,
        layer_idx: int,
        model_type: str = "roberta"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract FC layer weights from a transformer model

    Args:
        model: HiggingFace transformer model
        layer_idx: Which layer to extract (0-indexed)
        model_type: 'roberta' or 'gpt2' or 'bert' etc.
    
    Returns:
        fc_in_weight: First FC layer weight [hidden_size, intermediate_size]
        fc_out_weight: Second FC layer weight [intermediate_size, hidden_size]
    """

    if model_type == "roberta":
        # RoBERTa: encoder.layer[i].intermediate.dense and output.dense
        fc_in = model.roberta.encoder.layer[layer_idx].intermediate.dense.weight
        fc_out = model.reoberta.encoder.layer[layer_idx].output.dense.weight

    elif model_type == "bert":
        fc_in = model.bert.encoder.layer[layer_idx].intermediate.dense.weight
        fc_out = model.bert.encoder.layer[layer_idx].output.dense.weight

    elif model_type == "gpt2":
        #GPT-2: transfoer.h[i].mlp.c_fc and c_proj
        fc_in = model.transformer.h[layer_idx].mlp.c_fc.weight
        fc_out = model.transformer.h[layer_idx].mlp.c_proj.weight

    elif model_type == "llama":
        # Llama: model.layers[i].mlp.gate_proj, up_proj, down_proj
        # note Llama has 3 fc layers(SwiGLU), we use down and up proj here
        fc_in = model.model.layers[layer_idx].mlp.up_proj.weight
        fc_out = model.model.layers[layer_idx].mlp.down_proj.weight

    elif model_type == "gptj":
        # GPT-J: transformer.h[i].mlp.fc_in and fc_out (standard Linear layers)
        fc_in = model.transformer.h[layer_idx].mlp.fc_in.weight
        fc_out = model.transformer.h[layer_idx].mlp.fc_out.weight

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return fc_in, fc_out


   



