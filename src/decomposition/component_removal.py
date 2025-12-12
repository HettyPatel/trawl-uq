"""
Component removal for Tucker decomposition.

This module handles removing specific components form the core tensor and reconstructing
the weights without those components. 
"""

import torch
import tensorly as tl
from typing import List, Tuple
import numpy as np

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

