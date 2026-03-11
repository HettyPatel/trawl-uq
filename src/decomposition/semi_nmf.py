"""
Semi-NMF (Semi Non-negative Matrix Factorization) for transformer FC layers.

Semi-NMF factorizes W ≈ F @ G where:
  - F (basis matrix) can have mixed signs — lives in the residual stream output space
  - G (coefficient matrix) is constrained to be non-negative — lives in the intermediate input space

This is useful for interpretability because:
  - Non-negative G gives "parts-based" representation: components are additive, no cancellation
  - Each component captures a distinct additive contribution
  - When projected through lm_head, each component represents tokens it *promotes*

Reference: Ding et al. (2010) "Convex and Semi-Nonnegative Matrix Factorizations"

Algorithm (multiplicative update rules for Semi-NMF):
  Given W (m x n), find F (m x r) and G (r x n) s.t. G >= 0:

  F update (unconstrained): F = W @ G.T @ (G @ G.T)^{-1}
  G update (multiplicative): Uses split into positive/negative parts

  G_ij <- G_ij * sqrt( [F.T @ W]^+_ij + [F.T @ F @ G]^-_ij ) /
                       ( [F.T @ W]^-_ij + [F.T @ F @ G]^+_ij )

  where [X]^+ = (|X| + X) / 2 and [X]^- = (|X| - X) / 2
"""

import torch
import numpy as np
from typing import Tuple, Optional


def _positive_part(X: torch.Tensor) -> torch.Tensor:
    """[X]^+ = (|X| + X) / 2"""
    return (torch.abs(X) + X) / 2


def _negative_part(X: torch.Tensor) -> torch.Tensor:
    """[X]^- = (|X| - X) / 2"""
    return (torch.abs(X) - X) / 2


def decompose_weight_semi_nmf(
    weight: torch.Tensor,
    n_components: int,
    device: str = "cuda",
    max_iter: int = 200,
    tol: float = 1e-4,
    init: str = "svd",
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Semi-NMF decomposition of a weight matrix.

    W ≈ F @ G where G >= 0.

    For down_proj (4096 x 11008):
      - F: (4096, n_components) — basis vectors in residual stream space (mixed sign)
      - G: (n_components, 11008) — non-negative coefficients in intermediate space

    Args:
        weight: Weight matrix [m, n]
        n_components: Number of components (rank of approximation)
        device: Device for computation
        max_iter: Maximum number of iterations
        tol: Convergence tolerance (relative change in Frobenius norm)
        init: Initialization method — "svd" (default) or "random"
        verbose: Print convergence info

    Returns:
        F: Basis matrix [m, n_components] (mixed sign, residual stream space)
        G: Coefficient matrix [n_components, n] (non-negative, intermediate space)
    """
    # Work in float32 on CPU for stability, then move to device
    W = weight.detach().cpu().float()
    m, n = W.shape

    if n_components > min(m, n):
        n_components = min(m, n)
        if verbose:
            print(f"  Clamped n_components to {n_components}")

    # Initialize F and G
    if init == "svd":
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        # Take top-k components: distribute S evenly between F and G
        sqrt_S = torch.sqrt(S[:n_components])
        F = U[:, :n_components] * sqrt_S.unsqueeze(0)   # (m, r)
        G_signed = sqrt_S.unsqueeze(1) * Vh[:n_components, :]  # (r, n)
        # Make G non-negative: use the rectified positive part and
        # compensate by flipping the sign of F columns where G is mostly negative
        # For each component, if the mean of G_signed[k,:] is negative, flip both F[:,k] and G[k,:]
        for k in range(n_components):
            if G_signed[k, :].mean() < 0:
                G_signed[k, :] = -G_signed[k, :]
                F[:, k] = -F[:, k]
        G = torch.clamp(G_signed, min=0)
    elif init == "random":
        torch.manual_seed(42)
        F = torch.randn(m, n_components) * 0.01
        G = torch.abs(torch.randn(n_components, n)) * 0.01
    else:
        raise ValueError(f"Unknown init: {init}")

    eps = 1e-10

    # Normalize F columns and absorb scale into G
    f_norms = torch.norm(F, dim=0, keepdim=True).clamp(min=eps)  # (1, r)
    F = F / f_norms
    G = G * f_norms.T  # (r, n)

    prev_cost = float('inf')

    for iteration in range(max_iter):
        # --- Update F (unconstrained) ---
        # F = W @ G.T @ (G @ G.T)^{-1}
        GGt = G @ G.T  # (r, r)
        WGt = W @ G.T  # (m, r)
        # Regularize GGt for numerical stability
        GGt_reg = GGt + eps * torch.eye(n_components)
        try:
            F = torch.linalg.solve(GGt_reg.T, WGt.T).T  # (m, r)
        except torch.linalg.LinAlgError:
            GGt_inv = torch.linalg.pinv(GGt)
            F = WGt @ GGt_inv

        # Normalize F columns, absorb scale into G
        f_norms = torch.norm(F, dim=0, keepdim=True).clamp(min=eps)
        F = F / f_norms
        G = G * f_norms.T

        # --- Update G (non-negative, multiplicative) ---
        FtW = F.T @ W    # (r, n)
        FtF = F.T @ F    # (r, r)
        FtFG = FtF @ G   # (r, n)

        # Split into positive and negative parts
        FtW_pos = _positive_part(FtW)
        FtW_neg = _negative_part(FtW)
        FtFG_pos = _positive_part(FtFG)
        FtFG_neg = _negative_part(FtFG)

        # Multiplicative update
        numerator = FtW_pos + FtFG_neg + eps
        denominator = FtW_neg + FtFG_pos + eps
        G = G * torch.sqrt(numerator / denominator)

        # Compute reconstruction cost
        if (iteration + 1) % 10 == 0 or iteration == 0:
            cost = torch.norm(W - F @ G, 'fro').item()
            relative_change = abs(prev_cost - cost) / (prev_cost + eps)

            if verbose:
                print(f"  Iter {iteration+1:>4}: cost={cost:.4f}, "
                      f"rel_change={relative_change:.2e}")

            if relative_change < tol and iteration > 0:
                if verbose:
                    print(f"  Converged at iteration {iteration+1}")
                break

            prev_cost = cost

    # Move to device
    F = F.to(device)
    G = G.to(device)

    return F, G


def reconstruct_from_semi_nmf(
    F: torch.Tensor,
    G: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct weight matrix from Semi-NMF factors.

    W_approx = F @ G

    Args:
        F: Basis matrix [m, r]
        G: Coefficient matrix [r, n] (non-negative)

    Returns:
        W_approx: Reconstructed weight [m, n]
    """
    return F @ G


def get_semi_nmf_stats(
    weight: torch.Tensor,
    F: torch.Tensor,
    G: torch.Tensor,
) -> dict:
    """
    Compute statistics about the Semi-NMF decomposition.

    Returns:
        Dictionary with reconstruction error, sparsity of G, etc.
    """
    W = weight.detach().float()
    F_f = F.detach().float()
    G_f = G.detach().float()

    W_approx = F_f @ G_f

    # Reconstruction error
    fro_error = torch.norm(W.cpu() - W_approx.cpu(), 'fro').item()
    fro_original = torch.norm(W.cpu(), 'fro').item()
    relative_error = fro_error / fro_original if fro_original > 0 else float('inf')

    # G sparsity (fraction of near-zero entries)
    g_sparsity = (G_f.cpu() < 1e-6).float().mean().item()

    # Component norms (how "strong" each component is)
    f_norms = torch.norm(F_f.cpu(), dim=0)  # (r,)
    g_norms = torch.norm(G_f.cpu(), dim=1)  # (r,)
    component_strengths = (f_norms * g_norms).numpy()

    return {
        'frobenius_error': fro_error,
        'frobenius_original': fro_original,
        'relative_error': relative_error,
        'n_components': F_f.shape[1],
        'g_sparsity': g_sparsity,
        'component_strengths': component_strengths,
        'mean_component_strength': float(np.mean(component_strengths)),
    }
