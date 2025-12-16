""""
CP (CANDECOMP/PARAFAC) decomposition of FC layers

CP decomposes a tensor into a sum of rank-1 tensors.
Unlike Tucker, there's no core tensor - just factor matrices.
"""

import torch
import tensorly as tl
from tensorly.decomposition import parafac
import numpy as np
from typing import Tuple, List 

def decompose_fc_layer_cp(
        fc_in_weight: torch.Tensor,
        fc_out_weight: torch.Tensor,
        rank: int,
        device: str = "cuda"
) -> Tuple[List[torch.Tensor]]:
    """
    Decompose two consecutive FC layers using CP decomposition.

    CP decomposition: X = sum_{r=1}^{R} a_r ⊗ b_r ⊗ c_r
    where R is the rank, and a_r, b_r, c_r are the factor vectors and ⊗ 
    denotes the outer product.

    Args:
        fc_in_weight: First FC layer weight [hidden_size, intermediate_size]
        fc_out_weight: Second FC layer weight [intermediate_size, hidden_size]
        rank: CP rank (number of components)
        device: Device to run on

    Returns:
        factors: CP factors as list [weights, factor_0, factor_1, factor_2]
            - weights: compoent weights (lambda values) [rank]
            - factor_0: [2, rank] 
            - factor_1: [hidden_size, rank]
            - factor_2: [intermediate_size, rank]
    """

    # Transpose and stack
    fc_out_transposed = fc_out_weight.T  # [hidden_size, intermediate_size]
    weights_stacked = torch.stack([fc_in_weight, fc_out_transposed], dim=0)  # [2, hidden_size, intermediate_size]

    # Convert to CPU float32 for tensorly
    weights_np = weights_stacked.detach().cpu().float().numpy()

    # Set tensorly backend
    tl.set_backend('numpy')

    print(f"Decomposing tensor of shape {weights_np.shape} with rank {rank} using CP...")

    # Perform CP Decomposition
    # Returns CPTensor object with weights and factors
    cp_tensor = parafac(
        weights_np,
        rank=rank,
        init='random',
        random_sate=42,
        n_iter_max=100,
    )

    # Extract weights and factors
    # cp_tensor is (weights, factors) tuple

    weights = cp_tensor.weights  # [rank]
    factors = cp_tensor.factors  # List of factor matrices

    # convert back to torch tensors
    weights_tensor = torch.from_numpy(weights).to(device)
    factors_tensors = [torch.from_numpy(factor).to(device) for factor in factors]

    print(f"Decomposition done. Weights shape: {weights_tensor.shape}, ")
    print (" Factors shapes: " + ", ".join([str(factor.shape) for factor in factors_tensors]))

    
    # Returns as list: [weights, factor_0, factor_1, factor_2]
    return [weights_tensor] + factors_tensors

def reconstruct_from_cp(
        weights: torch.Tensor,
        factors: List[torch.Tensor]
) -> torch.Tensor:
    """
    Reconstruct tensor from CP decomposition.

    Args:
        weights: Component weights [rank]
        factors: List of factor matrices

    Returns:
        reconstructed: Reconstructed tensor [2, hidden_size, intermediate_size]
    """

    # Use tensorly for reconstruction
    tl.set_backend('pytorch')

    # Create CP tensor tuple
    cp_tensor = (weights, factors)

    # Reconstruct tensor
    reconstructed = tl.cp_to_tensor(cp_tensor)

    return reconstructed

def remove_cp_component(
        weights: torch.Tensor,
        factors: List[torch.Tensor],
        component_idx: int
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Remove a specific component from CP decomposition.

    Args:
        weights: Component weights [rank]
        factors: List of factor matrices
        component_idx: Index of component to remove
    
    Returns:
        modified_weights: Weights with component zeroed
        modified_factors Factors (unchanged but returned for consistency)
    """

    # Clone to avoiud modifying original
    modified_weights = weights.clone()

    # Zero out the component weight
    modified_weights[component_idx] = 0.0

    # Factors remain unchanged (we just zero the weight)
    # Could also zero the factor columns but zeroing weight is cleaner?

    return modified_weights, factors

def reconstruct_weights_cp(
        weights: torch.Tensor,
        factors: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstruct FC layer weights from CP decomposition.

    Args:
        weights: Component weights [rank]
        factors: List of 3 factor matrices

    Returns:
        fc_in_reconstructed: Reconstructed first FC layer weight [hidden_size, intermediate_size]
        fc_out_reconstructed: Reconstructed second FC layer weight [intermediate_size, hidden_size]
    """

    # Reconstruct full tensor
    reconstructed = reconstruct_from_cp(weights, factors)  # [2, hidden_size, intermediate_size]

    # Split back into two layers 
    fc_in_reconstructed = reconstructed[0]
    fc_out_reconstructed = reconstructed[1].T # Transpose back

    return fc_in_reconstructed, fc_out_reconstructed

def compute_reconstruction_error_cp(
        original: torch.Tensor,
        weights: torch.Tensor,
        factors: List[torch.Tensor]
) -> float:
    """
    Compute relative reconstruction errror for CP decomposition

    Args:
        original: Original tensor
        weights:: CP weights 
        factors: CP factors

    Returns:
        error: Relative Frobenius norm error
    """

    reconstructed = reconstruct_from_cp(weights, factors)
    diff = original - reconstructed
    error = torch.norm(diff) / torch.norm(original)

    return error.item()

def get_cp_component_importance(
        weights: torch.Tensor,
        factors: List[torch.Tensor]
) -> torch.Tensor:
    """
    Compute importance of each CP component.

    For CP, importance is proportional to:
    - The component weight (lambda)
    - The norms of the factor vectors

    Args:
        weights: Component weights [rank]
        factors: List of factor matrices

    Returns:
        importance_scores: Importance scores for each component
    """

    rank = weights.shape[0]
    importance_scores = torch.zeros(rank)

    for r in range(rank):
        # Component importance = Weight * product of factor norms
        weight = weights[r].abs()
        factor_norms = 1.0

        for factor in factors:
            factor_norms *= torch.norm(factor[:, r])

        importance_scores[r] = weight * factor_norms

    return importance_scores


