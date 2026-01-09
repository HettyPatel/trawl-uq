import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple, Dict
from tqdm import tqdm

# Import markers from knowledge module
from src.uncertainty.knowledge import DEGENERATE_MARKER, NO_FACTS_MARKER


def is_invalid_knowledge(knowledge: str) -> bool:
    """
    Check if a knowledge extraction result is invalid (degenerate or no facts found).

    Args:
        knowledge: The extracted knowledge string

    Returns:
        True if the knowledge is invalid and should not be used for similarity
    """
    if not knowledge or len(knowledge.strip()) == 0:
        return True
    if knowledge == DEGENERATE_MARKER:
        return True

    # Only mark as invalid if response starts with NO_FACTS_FOUND or is very short
    # Don't invalidate responses that have actual facts but mention NO_FACTS_FOUND somewhere
    stripped = knowledge.strip().upper()
    if stripped.startswith(NO_FACTS_MARKER):
        return True
    if stripped == NO_FACTS_MARKER:
        return True
    # Handle cases like "* NO_FACTS_FOUND" or "1. NO_FACTS_FOUND"
    if stripped.lstrip('*-0123456789. ').startswith(NO_FACTS_MARKER):
        return True

    return False


def count_invalid_knowledge(knowledge_responses: List[str]) -> Dict[str, int]:
    """
    Count invalid knowledge responses in a list.

    Args:
        knowledge_responses: List of extracted knowledge strings

    Returns:
        Dict with counts: degenerate_count, no_facts_count, valid_count, total_count
    """
    degenerate_count = 0
    no_facts_count = 0

    for kr in knowledge_responses:
        if kr == DEGENERATE_MARKER:
            degenerate_count += 1
        elif is_invalid_knowledge(kr) and kr != DEGENERATE_MARKER:
            # Use the same logic as is_invalid_knowledge for consistency
            no_facts_count += 1

    return {
        'degenerate_count': degenerate_count,
        'no_facts_count': no_facts_count,
        'valid_count': len(knowledge_responses) - degenerate_count - no_facts_count,
        'total_count': len(knowledge_responses)
    }


class NLISimilarityCalculator:
    """
    Compute semantic similarity between texts using NLI (Natural Language Inference)

    Use DeBERTa model to check if text A entails text B.
    High entailment score = high semantic similarity. 

    This is the approach from MD-UQ paper Section 2.2.1 (Equation 3)
    """

    def __init__(self, model_name: str = "microsoft/deberta-large-mnli", device: str = "cuda"):
        """
        Args:
            model_name: HuggingFace NLI model (default: DeBERTa-large-mnli)
            device: Device to run on
        """
        print(f"Loading NLI model: {model_name} on {device}...")

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # Find the correct index for 'entailment' label
        self.entailment_idx = None
        for idx, label in self.model.config.id2label.items():
            if label.lower() == 'entailment':
                self.entailment_idx = idx
                break
        
        if self.entailment_idx is None:
            raise ValueError(f"Could not find entailment label in {self.model.config.id2label}")
        
        print(f"NLI model loaded.")
        print(f"  Label mapping: {self.model.config.id2label}")
        print(f"  Using entailment index: {self.entailment_idx}")

    def compute_entailment_score(self, text1: str, text2: str) -> float:
        """
        Compute entailment probability: P(text1 entails text2)

        Args:
            text1: premise text
            text2: hypothesis text

        Returns:
            entailment_prob: Probability that text 1 entails text 2 [0, 1]
        """

        # Tokenize the pair
        inputs = self.tokenizer(
            text1,
            text2,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Get logits
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits # shape (1, 3)

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1) # shape (1, 3)

        # Use the CORRECT entailment index
        entailment_prob = probs[0, self.entailment_idx].item()
        return entailment_prob
    
    def compute_bidirectional_similarity(self, text1: str, text2: str) -> float:
        """
        Compute symmetric similarty using bidirectiona entailment:

        From MD-UQ paper Equation 3:
        s_ij = 0.5 * (P_entail(A_i, A_j) + P_entail(A_j, A_i))

        Args:
            text1: First text
            text2: Second text

        Returns: 
            similarity: Average of both directions [0, 1]
        """

        # Compute both directions
        score_1_to_2 = self.compute_entailment_score(text1, text2)
        score_2_to_1 = self.compute_entailment_score(text2, text1)

        # Average them (symmetric similarity)
        similarity = 0.5 * (score_1_to_2 + score_2_to_1)
        return similarity
    
def build_semantic_similarity_matrix(
        responses: List[str],
        nli_calculator: NLISimilarityCalculator = None,
        device : str = "cuda"
) -> np.ndarray:
    """
    Build semantic similarity matrix for a list of responses. 

    This creates an n x n matrix where entry (i, j) is the semantic similarity between response i and response j.

    Args:
        responses: List of generated text responses
        nli_calculator: Pre-loaded NLI calculator (optional - will create one if None)
        device: Device to run on (default: "cuda")

    Returns:
        similarity_matrix: n x n numpy array of similarity scores [0, 1]
    """

    n = len(responses)
    
    # Initialize calcualtor if not provided
    if nli_calculator is None:
        nli_calculator = NLISimilarityCalculator(device=device)

    # Initialize similarity matrix
    similarity_matrix = np.zeros((n, n))

    # Fill diagonal (response is identical to itself)
    np.fill_diagonal(similarity_matrix, 1.0)

    # Compute pairwise similarities (uppter triangle only, then mirror)
    print(f"Computing semantic similarities for {n} responses...")
    for i in tqdm(range(n), desc="Semantic similarity"):
        for j in range(i + 1, n): # only upper triangle

            # Compute similarity between response i and j
            sim = nli_calculator.compute_bidirectional_similarity(
                responses[i],
                responses[j]
            )

            # Fill both (i, j) and (j, i) since similarity is symmetric
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    return similarity_matrix

def build_similarity_matrices_for_dataset(
        results: List[dict],
        nli_calculator: NLISimilarityCalculator = None,
        device : str = "cuda"
) -> List[dict]:
    """
    Build semantic similarity matrices for all samples in a dataset.

    Args:
        results: List of dicts from generate_for_dataset(), each containing:
            {'id', 'question' 'answer', 'responses'}
        nli_calculator : Pre-loaded NLI calcualtor.
        device: Device to run on.
    """

    # Initialize calculator once for all samples
    if nli_calculator is None:
        nli_calculator = NLISimilarityCalculator(device=device)

    # Process each sample
    for result in tqdm(results, desc="Building similarity matrices"):

        # Build semantic similarity matrix for this sample's responses
        S = build_semantic_similarity_matrix(
            responses = result['responses'],
            nli_calculator = nli_calculator,
            device = device
        )

        # Add to result dict
        result['similarity_matrix'] = S

    return results

def compute_spectral_norm_ratio(S_semantic: np.ndarray, S_knowledge: np.ndarray) -> float:
    """
    Compute Spectral Norm Ratio (SNR) between two matrices.
    From MD-UQ paper Section 3.2 Equation 7:

    SNR = ||Λ_sem||_2 / ||Λ_know||_2

    Where ||·||_2 is the spectral norm (largest eigenvalue).

    SNR ≈ 1 means matrices have similar structure (high redundancy).

    Args:
        S_semantic: Semantic similarity matrix
        S_knowledge: Knowledge similarity matrix
    Returns:
        snr: Spectral Norm Ratio
    """

    ## compute largest eigenvalue (spectral norm) for both matrices
    eigenvals_semantic = np.linalg.eigvalsh(S_semantic)
    eigenvals_knowledge = np.linalg.eigvalsh(S_knowledge)

    spectral_norm_semantic = np.max(eigenvals_semantic)
    spectral_norm_knowledge = np.max(eigenvals_knowledge)

    # Compute SNR
    snr = spectral_norm_semantic / spectral_norm_knowledge

    return snr


def build_knowledge_similarity_matrix(
        knowledge_responses: List[str],
        nli_calculator: NLISimilarityCalculator = None,
        device: str = "cuda",
        invalid_similarity: float = 0.0
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Build knowledge similarity matrix, handling invalid (degenerate/no_facts) responses.

    For invalid responses, we set similarity to invalid_similarity (default 0.0),
    which means they are treated as completely dissimilar from everything.
    This prevents degenerate responses from artificially inflating similarity.

    Args:
        knowledge_responses: List of extracted knowledge strings
        nli_calculator: Pre-loaded NLI calculator (optional)
        device: Device to run on
        invalid_similarity: Similarity value to use for invalid responses (default 0.0)

    Returns:
        similarity_matrix: n x n numpy array of similarity scores
        stats: Dict with counts of invalid responses
    """
    n = len(knowledge_responses)

    # Initialize calculator if not provided
    if nli_calculator is None:
        nli_calculator = NLISimilarityCalculator(device=device)

    # Count invalid responses
    stats = count_invalid_knowledge(knowledge_responses)

    # Check if too many invalid - if > 50%, the measurement is unreliable
    if stats['valid_count'] < n * 0.5:
        stats['is_reliable'] = False
    else:
        stats['is_reliable'] = True

    # Initialize similarity matrix
    similarity_matrix = np.zeros((n, n))
    np.fill_diagonal(similarity_matrix, 1.0)

    # Track which responses are invalid
    invalid_mask = [is_invalid_knowledge(kr) for kr in knowledge_responses]

    # Compute pairwise similarities
    print(f"Computing knowledge similarities for {n} responses ({stats['valid_count']} valid)...")
    for i in tqdm(range(n), desc="Knowledge similarity"):
        for j in range(i + 1, n):
            # If either response is invalid, use invalid_similarity
            if invalid_mask[i] or invalid_mask[j]:
                sim = invalid_similarity
            else:
                # Both valid - compute actual similarity
                sim = nli_calculator.compute_bidirectional_similarity(
                    knowledge_responses[i],
                    knowledge_responses[j]
                )

            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    return similarity_matrix, stats

