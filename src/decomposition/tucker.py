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

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return fc_in, fc_out


   



