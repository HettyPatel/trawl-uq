import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker, parafac
from typing import Dict, Tuple

def compute_reconstruction_error(
        tensor: np.ndarray,
        rank: int,
        decomposition_type: str = "tucker"
) -> float:
    """
    Compute reconstruction error after tensor decomposition.

    From MD-UQ paper Section 3.3:
    Lower reconstruction error at lower rank = more structred/blocky = lower uncertainty. 

    Args:
        tensor: Input tensor (e.g., stacked similarity matrices)
        rank: Rank for decomposition
        decomposition_type: 'tucker' or 'cp' for CP decomposition
    
    Returns:
        reconstruction_error: Normalized Forbenius norm of error
    """

    tl.set_backend('numpy')

    # Convert to tensorly tensor
    X = tl.tensor(tensor)
    norm_X = tl.norm(X)

    # Decompose and reconstruct
    if decomposition_type == 'tucker':
        # For 3D tensor [n, n, 2] use rank as (rank, rank, 2)
        # Keep last dimension (2 matrices) uncompressed

        if len(tensor.shape) == 3:
            rank_spec = [rank, rank, tensor.shape[2]]
        else:
            rank_spec = rank

        # Tucker decomposition
        core, factors = tucker(X, rank=rank_spec, init='random')
        X_reconstructed = tl.tucker_to_tensor((core, factors))

    elif decomposition_type == 'cp':
        # CP decomposition
        factors = parafac(X, rank=rank, init='random')
        X_reconstructed = tl.cp_to_tensor(factors)
    else:
        raise ValueError(f"Unknown decomposition type: {decomposition_type}. Use 'tucker' or 'cp'.")
    
    # Compute reconstruction error
    norm_diff = tl.norm(X - X_reconstructed)
    reconstruction_error = norm_diff / norm_X

    return float(reconstruction_error)


def compute_reconstruction_fit(
        tensor : np.ndarray,
        rank : int,
        decomposition_type : str = "tucker" 
) -> float:
    """
    Compute reconstruction fit (1 - error).
    
    Higher fit = better reconstruction = more structured = lower uncertainty.

    Args:
        tensor : Input tensor
        rank : rank for decomposition
        decomposition_type : 'tucker' or 'cp'
    
    Returns :
        fit : Reconstruction fit [0, 1] where 1 = perfect reconstruction
    """

    error = compute_reconstruction_error(tensor, rank, decomposition_type)

    fit = 1.0 - error
    return fit

def compute_spectral_properties(matrix: np.ndarray) -> Dict[str, float]:
    """
    Compute spectral properties of a similarity matrix. 

    Args:
        matrix : Similarity matrix (n x n)

    Returns: 
        Properties : dict with:
            - spectral_norm : Largest eigenvalue
            - num_large_eigenvalues : Number of eigenvalues > 0.1 * max
            - eigenvalue_ratio : Ratio of largest to second largest eigenvalue 
    """

    # compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues = np.sort(eigenvalues)[::-1] # descending order

    # spectral norm (largest eigenvalue)
    spectral_norm = eigenvalues[0]

    # Count significant eigenvalues
    threshold = 0.1 * spectral_norm
    num_large_eigenvalues = np.sum(eigenvalues > threshold)

    # Eigenvalue ratio (Measures how dominant the first component is)
    if len(eigenvalues) > 1:
        eigenvalue_ratio = eigenvalues[0] / eigenvalues[1]
    else:
        eigenvalue_ratio = 1.0
    
    return {
        'spectral_norm' : float(spectral_norm),
        'num_large_eigenvalues' : int(num_large_eigenvalues),
        'eigenvalue_ratio' : float(eigenvalue_ratio)
    }


def compute_blockiness_score(
        S_semantic : np.ndarray,
        S_knowledge : np.ndarray,
        rank : int = 10,
        decomposition_type : str = "tucker"
) -> Dict[str, float] :
    """
    Compute overall "blockiness" score for uncertainty quantification.

    Combines multiple matrics:
    1. Reconstruction fit at low rank (higher = more blocky)
    2. Spectral properties (fewer large eigenvalues = more blocky)
    3. Spectral norm ratio between semantic and knowledge matrices

    Args:
        S_semantic : Semantic similarity matrix (n x n)
        S_knowledge : Knowledge similarity matrix (n x n)
        rank : Rank for tensor decomposition
        decompostion_type : 'tucker' or 'cp'

    Returns:
        metrics: Dict of blockiness metrics
    """

    # Stack matrices into 3D tensor [n, n, 2]
    tensor = np.stack([S_semantic, S_knowledge], axis=2)

    # 1. Reconstruction fit (how compressible is the tensor?)
    recon_fit = compute_reconstruction_fit(
        tensor,
        rank,
        decomposition_type
    )

    # 2. Spectral properties of each matrix
    spectral_semantic = compute_spectral_properties(S_semantic)
    spectral_knowledge = compute_spectral_properties(S_knowledge)

    # 3. Spectral norm ratio (from MD-UQ paper)
    snr = spectral_semantic['spectral_norm'] / spectral_knowledge['spectral_norm']

    # 4. Average number of large eigenvalues (fewer = more blocky)
    avg_large_eigenvalues = (
        spectral_semantic['num_large_eigenvalues'] +
        spectral_knowledge['num_large_eigenvalues']
    ) / 2.0

    return {
        'reconstruction_fit' : recon_fit,
        'spectral_norm_ratio' : snr,
        'avg_large_eigenvalues' : avg_large_eigenvalues,
        'semantic_spectral_norm' : spectral_semantic['spectral_norm'],
        'knowledge_spectral_norm' : spectral_knowledge['spectral_norm'],
        'semantic_eigenvalue_ratio' : spectral_semantic['eigenvalue_ratio'],
        'knowledge_eigenvalue_ratio' : spectral_knowledge['eigenvalue_ratio']
    }



## Not from the paper but useful for summarizing blockiness into single score

def compute_uncertainty_score(blockiness_metrics: Dict[str, float]) -> float:
    """
    Compute single uncertainty score from blockiness metrics.

    Lower blockiness = higher uncertainty. 

    Simple weighted combination: 
    - High reconstruction fit -> low uncertainty
    - More large eigenvalues -> high uncertainty

    Args:
        blockiness_metrics : Dict form compute_blockiness_score()

    Returns:
        uncertainty : Single uncertainty score (higher = more uncertain)
    """

    # Invert reconstruction fit (high fit = low uncertainty)
    uncertainty_from_fit = 1.0 - blockiness_metrics['reconstruction_fit']

    # Normalize number of eigenvalues (more = higher uncertainty)
    # assume max possible eigenvalues = 20 for normalization
    max_eigenvalues = 20.0
    uncertainty_from_eigenvalues = blockiness_metrics['avg_large_eigenvalues'] / max_eigenvalues

    # Weighted combination
    uncertainty = 0.6 * uncertainty_from_fit + 0.4 * uncertainty_from_eigenvalues

    return float(uncertainty)
    

