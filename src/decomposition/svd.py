"""
SVD (Singular Value Decomposition) for transformer FC layers

This module implements LASER-style rank reduction using SVD.
Unlike CP/Tucker which operates on stacked tensors, SVD operates on individual weight matrices.

Key concepts:
- W = U @ Σ @ V^T where Σ contains singular values in descending order
- Keeping top-k% singular values = low-rank approximation
- Removing higher-order components (smaller singular values) can improve performance (LASER paper)
"""

import torch
import numpy as np
from typing import Tuple, Optional


def decompose_weight_svd(
        weight: torch.Tensor,
        device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform SVD decomposition on a weight matrix.

    W = U @ diag(S) @ V^T

    Args:
        weight: Weight matrix [m, n]
        device: Device for computation

    Returns:
        U: Left singular vectors [m, k] where k = min(m, n)
        S: Singular values [k] (sorted descending)
        Vh: Right singular vectors transposed [k, n]
    """
    # Move to CPU for SVD (more stable)
    weight_cpu = weight.detach().cpu().float()

    # Perform SVD
    U, S, Vh = torch.linalg.svd(weight_cpu, full_matrices=False)

    # Move back to device
    U = U.to(device)
    S = S.to(device)
    Vh = Vh.to(device)

    return U, S, Vh


def reconstruct_from_svd(
        U: torch.Tensor,
        S: torch.Tensor,
        Vh: torch.Tensor
) -> torch.Tensor:
    """
    Reconstruct weight matrix from SVD components.

    W = U @ diag(S) @ Vh

    Args:
        U: Left singular vectors [m, k]
        S: Singular values [k]
        Vh: Right singular vectors transposed [k, n]

    Returns:
        W: Reconstructed weight matrix [m, n]
    """
    # W = U @ diag(S) @ Vh
    return U @ torch.diag(S) @ Vh


def truncate_svd(
        U: torch.Tensor,
        S: torch.Tensor,
        Vh: torch.Tensor,
        keep_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Truncate SVD to keep only top-k% singular values.

    keeping lower-order components (larger singular values)
    and removing higher-order components (smaller singular values).

    Args:
        U: Left singular vectors [m, k]
        S: Singular values [k]
        Vh: Right singular vectors transposed [k, n]
        keep_ratio: Fraction of singular values to keep (0.0 to 1.0)
                   e.g., 0.1 means keep top 10% (90% reduction)

    Returns:
        U_truncated: Truncated left singular vectors [m, k_new]
        S_truncated: Truncated singular values [k_new]
        Vh_truncated: Truncated right singular vectors [k_new, n]
    """
    total_components = len(S)
    k_keep = max(1, int(total_components * keep_ratio))  # Keep at least 1 component

    # Keep top-k components (largest singular values)
    U_truncated = U[:, :k_keep]
    S_truncated = S[:k_keep]
    Vh_truncated = Vh[:k_keep, :]

    return U_truncated, S_truncated, Vh_truncated


def low_rank_approximation(
        weight: torch.Tensor,
        keep_ratio: float,
        device: str = "cuda"
) -> torch.Tensor:
    """
    Compute low-rank approximation of a weight matrix using SVD.

    This is the main operation: W_LR = truncated_SVD(W)

    Args:
        weight: Original weight matrix [m, n]
        keep_ratio: Fraction of singular values to keep (0.0 to 1.0)
        device: Device for computation

    Returns:
        weight_lr: Low-rank approximated weight matrix [m, n]
    """
    # Decompose
    U, S, Vh = decompose_weight_svd(weight, device)

    # Truncate
    U_trunc, S_trunc, Vh_trunc = truncate_svd(U, S, Vh, keep_ratio)

    # Reconstruct
    weight_lr = reconstruct_from_svd(U_trunc, S_trunc, Vh_trunc)

    return weight_lr


def compute_energy_retention(
        S_original: torch.Tensor,
        S_truncated: torch.Tensor
) -> float:
    """
    Compute how much "energy" (variance) is retained after truncation.

    Energy = sum(σ_i^2)
    Retention = sum(σ_truncated^2) / sum(σ_original^2)

    Args:
        S_original: Original singular values
        S_truncated: Truncated singular values

    Returns:
        retention: Fraction of energy retained (0.0 to 1.0)
    """
    total_energy = torch.sum(S_original ** 2).item()
    retained_energy = torch.sum(S_truncated ** 2).item()

    return retained_energy / total_energy if total_energy > 0 else 0.0


def compute_reconstruction_error_svd(
        original: torch.Tensor,
        reconstructed: torch.Tensor
) -> float:
    """
    Compute relative reconstruction error.

    Error = ||original - reconstructed||_F / ||original||_F

    Args:
        original: Original weight matrix
        reconstructed: Reconstructed weight matrix

    Returns:
        error: Relative Frobenius norm error
    """
    diff = original.float() - reconstructed.float()
    error = torch.norm(diff) / torch.norm(original.float())
    return error.item()


def get_svd_stats(
        weight: torch.Tensor,
        device: str = "cuda"
) -> dict:
    """
    Get statistics about the SVD of a weight matrix.

    Args:
        weight: Weight matrix
        device: Device for computation

    Returns:
        stats: Dictionary with SVD statistics
    """
    U, S, Vh = decompose_weight_svd(weight, device)

    # Compute effective rank (based on singular value distribution)
    S_normalized = S / S.sum()
    entropy = -torch.sum(S_normalized * torch.log(S_normalized + 1e-10))
    effective_rank = torch.exp(entropy).item()

    # Compute energy concentration
    cumulative_energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)

    # Find k for 90%, 95%, 99% energy retention
    k_90 = torch.searchsorted(cumulative_energy, torch.tensor(0.90).to(device)).item() + 1
    k_95 = torch.searchsorted(cumulative_energy, torch.tensor(0.95).to(device)).item() + 1
    k_99 = torch.searchsorted(cumulative_energy, torch.tensor(0.99).to(device)).item() + 1

    return {
        'shape': tuple(weight.shape),
        'max_rank': len(S),
        'effective_rank': effective_rank,
        'top_singular_value': S[0].item(),
        'bottom_singular_value': S[-1].item(),
        'singular_value_ratio': (S[0] / S[-1]).item() if S[-1] > 0 else float('inf'),
        'k_for_90_energy': k_90,
        'k_for_95_energy': k_95,
        'k_for_99_energy': k_99,
        'singular_values': S.cpu().numpy()
    }


def apply_svd_to_layer(
        model,
        layer_idx: int,
        keep_ratio: float,
        matrix_type: str = "mlp_in",
        model_type: str = "llama",
        device: str = "cuda"
) -> Tuple[torch.Tensor, dict]:
    """
    Apply SVD truncation to a specific layer's weight matrix.

    Args:
        model: HuggingFace transformer model
        layer_idx: Which layer to modify (0-indexed)
        keep_ratio: Fraction of singular values to keep
        matrix_type: Which matrix to truncate:
                    - "mlp_in": MLP input matrix (up_proj for Llama)
                    - "mlp_out": MLP output matrix (down_proj for Llama)
        model_type: Type of model ('llama', 'gpt2', etc.)
        device: Device for computation

    Returns:
        weight_lr: Low-rank approximated weight
        stats: Dictionary with truncation statistics
    """
    # Get original weight
    if model_type == "llama":
        if matrix_type == "mlp_in":
            weight = model.model.layers[layer_idx].mlp.up_proj.weight.data
        elif matrix_type == "mlp_out":
            weight = model.model.layers[layer_idx].mlp.down_proj.weight.data
        else:
            raise ValueError(f"Unknown matrix_type: {matrix_type}")
    elif model_type == "gpt2":
        if matrix_type == "mlp_in":
            weight = model.transformer.h[layer_idx].mlp.c_fc.weight.data
        elif matrix_type == "mlp_out":
            weight = model.transformer.h[layer_idx].mlp.c_proj.weight.data
        else:
            raise ValueError(f"Unknown matrix_type: {matrix_type}")
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Get original SVD
    U, S, Vh = decompose_weight_svd(weight, device)

    # Truncate
    U_trunc, S_trunc, Vh_trunc = truncate_svd(U, S, Vh, keep_ratio)

    # Reconstruct
    weight_lr = reconstruct_from_svd(U_trunc, S_trunc, Vh_trunc)

    # Compute statistics
    stats = {
        'layer_idx': layer_idx,
        'matrix_type': matrix_type,
        'original_shape': tuple(weight.shape),
        'keep_ratio': keep_ratio,
        'original_rank': len(S),
        'truncated_rank': len(S_trunc),
        'reduction_percent': (1.0 - keep_ratio) * 100,
        'energy_retention': compute_energy_retention(S, S_trunc),
        'reconstruction_error': compute_reconstruction_error_svd(weight, weight_lr)
    }

    return weight_lr, stats


def update_layer_with_svd(
        model,
        layer_idx: int,
        weight_lr: torch.Tensor,
        matrix_type: str = "mlp_in",
        model_type: str = "llama"
):
    """
    Update a layer's weight with SVD-truncated version.

    Args:
        model: HuggingFace transformer model
        layer_idx: Which layer to update
        weight_lr: Low-rank approximated weight
        matrix_type: Which matrix to update
        model_type: Type of model
    """
    if model_type == "llama":
        if matrix_type == "mlp_in":
            target = model.model.layers[layer_idx].mlp.up_proj.weight
        elif matrix_type == "mlp_out":
            target = model.model.layers[layer_idx].mlp.down_proj.weight
        else:
            raise ValueError(f"Unknown matrix_type: {matrix_type}")
    elif model_type == "gpt2":
        if matrix_type == "mlp_in":
            target = model.transformer.h[layer_idx].mlp.c_fc.weight
        elif matrix_type == "mlp_out":
            target = model.transformer.h[layer_idx].mlp.c_proj.weight
        else:
            raise ValueError(f"Unknown matrix_type: {matrix_type}")
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Update weight in place
    target.data = weight_lr.to(target.device).to(target.dtype)


def restore_original_weight(
        model,
        layer_idx: int,
        original_weight: torch.Tensor,
        matrix_type: str = "mlp_in",
        model_type: str = "llama"
):
    """
    Restore original weight (after SVD truncation experiment).

    Args:
        model: HuggingFace transformer model
        layer_idx: Which layer to restore
        original_weight: Original weight to restore
        matrix_type: Which matrix to restore
        model_type: Type of model
    """
    update_layer_with_svd(model, layer_idx, original_weight, matrix_type, model_type)


# =============================================================================
# Reduction percentage utilities
# =============================================================================

# LASER-style reduction percentages
LASER_REDUCTION_PERCENTAGES = [10, 25, 40, 50, 60, 75, 90, 92.5, 95, 97.5, 98, 98.5, 99, 99.5, 99.75]

def reduction_to_keep_ratio(reduction_percent: float) -> float:
    """
    Convert reduction percentage to keep ratio.

    Args:
        reduction_percent: Percentage of components to REMOVE (0-100)
                          e.g., 90 means remove 90%, keep 10%

    Returns:
        keep_ratio: Fraction to KEEP (0.0-1.0)
                   e.g., 0.1 means keep 10%
    """
    return 1.0 - (reduction_percent / 100.0)


def keep_ratio_to_reduction(keep_ratio: float) -> float:
    """
    Convert keep ratio to reduction percentage.

    Args:
        keep_ratio: Fraction to KEEP (0.0-1.0)

    Returns:
        reduction_percent: Percentage REMOVED (0-100)
    """
    return (1.0 - keep_ratio) * 100.0
