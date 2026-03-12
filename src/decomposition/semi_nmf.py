"""
Sparse Semi-NMF for transformer FC layers.

Semi-NMF factorizes W ≈ F @ G where:
  - F (basis matrix) can have mixed signs — lives in the residual stream output space
  - G (coefficient matrix) is constrained to be non-negative — lives in the intermediate input space

Sparsity on G is enforced via an L1 penalty (lambda_), solved column-wise with NNLS.

Reference: Ding et al. (2010) "Convex and Semi-Nonnegative Matrix Factorizations"

Algorithm:
  F update (unconstrained least squares): F = W @ G.T @ (G @ G.T)^{-1}
  G update (sparse NNLS per column):
    min_{g >= 0} ||w_j - F @ g_j||^2 + lambda_ * sum(g_j)
    Equivalent to: min_{g >= 0} ||[F; sqrt(lambda_)*I] @ g - [w_j; 0]||^2
    Solved with scipy.optimize.nnls column-wise.
"""

import numpy as np
import torch
from scipy.optimize import nnls
from typing import Tuple


def decompose_weight_semi_nmf(
    weight: torch.Tensor,
    n_components: int,
    device: str = "cuda",
    max_iter: int = 200,
    tol: float = 1e-4,
    init: str = "svd",
    lambda_: float = 0.0,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sparse Semi-NMF decomposition of a weight matrix.

    W ≈ F @ G where G >= 0 and G is sparse (controlled by lambda_).

    For down_proj (4096 x 11008):
      - F: (4096, n_components) — basis vectors in residual stream space (mixed sign)
      - G: (n_components, 11008) — non-negative sparse coefficients in intermediate space

    Args:
        weight:      Weight matrix [m, n]
        n_components: Number of components (rank of approximation)
        device:      Device to return tensors on
        max_iter:    Maximum number of iterations
        tol:         Convergence tolerance (relative change in Frobenius norm)
        init:        Initialization method — "svd" (default) or "random"
        lambda_:     L1 sparsity penalty on G (0 = no sparsity, higher = sparser G)
        verbose:     Print convergence info

    Returns:
        F: Basis matrix [m, n_components] (mixed sign)
        G: Coefficient matrix [n_components, n] (non-negative, sparse if lambda_ > 0)
    """
    W = weight.detach().cpu().float().numpy()
    m, n = W.shape

    if n_components > min(m, n):
        n_components = min(m, n)
        if verbose:
            print(f"  Clamped n_components to {n_components}")

    # --- Initialisation ---
    if init == "svd":
        U, S, Vh = np.linalg.svd(W, full_matrices=False)
        sqrt_S = np.sqrt(S[:n_components])
        F = U[:, :n_components] * sqrt_S[np.newaxis, :]        # (m, r)
        G = sqrt_S[:, np.newaxis] * Vh[:n_components, :]       # (r, n)
        # For each component, ensure the net direction of G is positive
        # (flip both F[:,k] and G[k,:] if G[k,:] is mostly negative)
        for k in range(n_components):
            if G[k].mean() < 0:
                G[k] = -G[k]
                F[:, k] = -F[:, k]
        G = np.clip(G, 0, None)
    elif init == "random":
        rng = np.random.default_rng(42)
        F = rng.standard_normal((m, n_components)) * 0.01
        G = np.abs(rng.standard_normal((n_components, n))) * 0.01
    else:
        raise ValueError(f"Unknown init: {init!r}")

    eps = 1e-10
    prev_cost = float("inf")

    # Augmented F for the L1-penalised NNLS solve:
    #   min_{g>=0} ||W_aug_col - F_aug @ g||^2
    # where F_aug = [F; sqrt(lambda_)*I_r] and W_aug_col = [w_j; 0]
    sqrt_lam = np.sqrt(lambda_) if lambda_ > 0 else 0.0
    zero_pad = np.zeros((n_components, n))  # reused every iteration

    for iteration in range(max_iter):
        # --- F update (unconstrained least squares) ---
        # F = W @ G.T @ (G @ G.T)^{-1}
        GGt = G @ G.T                                          # (r, r)
        WGt = W @ G.T                                          # (m, r)
        GGt_reg = GGt + eps * np.eye(n_components)
        try:
            F = np.linalg.solve(GGt_reg.T, WGt.T).T           # (m, r)
        except np.linalg.LinAlgError:
            F = WGt @ np.linalg.pinv(GGt)

        # --- G update (sparse NNLS per column) ---
        if lambda_ > 0:
            F_aug = np.vstack([F, sqrt_lam * np.eye(n_components)])  # (m+r, r)
            W_aug = np.vstack([W, zero_pad])                          # (m+r, n)
        else:
            F_aug = F
            W_aug = W

        # Solve each column independently
        for j in range(n):
            G[:, j], _ = nnls(F_aug, W_aug[:, j])

        # --- Convergence ---
        if (iteration + 1) % 10 == 0 or iteration == 0:
            cost = np.linalg.norm(W - F @ G, "fro")
            relative_change = abs(prev_cost - cost) / (prev_cost + eps)

            if verbose:
                g_sparsity = (G < 1e-6).mean()
                print(f"  Iter {iteration+1:>4}: cost={cost:.4f}  "
                      f"rel_change={relative_change:.2e}  "
                      f"G_sparsity={g_sparsity:.3f}")

            if relative_change < tol and iteration > 0:
                if verbose:
                    print(f"  Converged at iteration {iteration+1}")
                break

            prev_cost = cost

    F_t = torch.from_numpy(F).float().to(device)
    G_t = torch.from_numpy(G).float().to(device)
    return F_t, G_t


def reconstruct_from_semi_nmf(F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """Reconstruct weight matrix: W_approx = F @ G"""
    return F @ G


def get_semi_nmf_stats(
    weight: torch.Tensor,
    F: torch.Tensor,
    G: torch.Tensor,
) -> dict:
    """
    Compute statistics about the Semi-NMF decomposition.

    Returns dict with reconstruction error, G sparsity, and component strengths.
    """
    W = weight.detach().cpu().float()
    F_f = F.detach().cpu().float()
    G_f = G.detach().cpu().float()

    W_approx = F_f @ G_f
    fro_error = torch.norm(W - W_approx, "fro").item()
    fro_original = torch.norm(W, "fro").item()
    relative_error = fro_error / fro_original if fro_original > 0 else float("inf")

    g_sparsity = (G_f < 1e-6).float().mean().item()

    f_norms = torch.norm(F_f, dim=0)   # (r,)
    g_norms = torch.norm(G_f, dim=1)   # (r,)
    component_strengths = (f_norms * g_norms).numpy()

    return {
        "frobenius_error": fro_error,
        "frobenius_original": fro_original,
        "relative_error": relative_error,
        "n_components": F_f.shape[1],
        "g_sparsity": g_sparsity,
        "component_strengths": component_strengths,
        "mean_component_strength": float(np.mean(component_strengths)),
    }
