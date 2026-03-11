"""
SVD Noise Removal Experiment

Adapts the CP noise removal approach (experiment 07) to SVD decomposition.
Instead of removing individual CP components (rank ~40), removes CHUNKS of
singular values (from ~4096 total) and evaluates impact using MCQ, Open QA,
or both evaluation modes.

Supports operating on single or multiple weight matrices simultaneously.
When multiple matrices are specified, the same chunk indices are dropped
from all matrices.

Experiment flow:
1. Load evaluation set(s) based on --dataset choice (mcq, open_qa, or both)
2. SVD decompose target weight matrix(es)
3. Measure baseline metrics for chosen evaluation mode(s)
4. For each chunk of singular values: zero them out, evaluate, restore
5. Classify chunks as noise/signal based on entropy and accuracy changes
6. Save results

Usage:
    # MCQ only, 50 components per chunk
    python experiments/14_svd_noise_removal.py --layer 31 --matrix mlp_out --dataset mcq --components-per-chunk 50

    # Open QA only
    python experiments/14_svd_noise_removal.py --layer 31 --matrix mlp_out --dataset open_qa --chunk-size 100

    # Both evaluations (default)
    python experiments/14_svd_noise_removal.py --layer 31 --matrix mlp_in mlp_out --chunk-size 100

    # Quick test
    python experiments/14_svd_noise_removal.py --layer 31 --matrix mlp_out --dataset mcq --test
"""

import sys
sys.path.append('.')

import json
import re
import torch
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from src.generation.generate import load_model_and_tokenizer, seed_everything
from src.evaluation.metrics import compute_mcq_entropy_and_nll
from src.decomposition.svd import (
    decompose_weight_svd,
    reconstruct_from_svd,
    update_layer_with_svd,
    restore_original_weight,
    compute_energy_retention,
    get_svd_stats,
)


# =============================================================================
# Helper functions (copied from experiments 07 and 13 for self-containment)
# =============================================================================

def load_mcq_eval_set(filepath: str):
    """Load MCQ evaluation set."""
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data['samples'])} MCQ samples from {filepath}")
    return data['samples']


def evaluate_mcq_on_samples(model, tokenizer, samples, device="cuda"):
    """Evaluate MCQ entropy and NLL on all samples."""
    results = []
    for sample in tqdm(samples, desc="Evaluating MCQ", leave=False):
        metrics = compute_mcq_entropy_and_nll(
            mcq_prompt=sample['mcq_prompt'],
            correct_letter=sample['correct_letter'],
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        results.append({
            'sample_id': sample['id'],
            'question': sample['question'],
            'correct_answer': sample['correct_answer'],
            'correct_letter': sample['correct_letter'],
            **metrics
        })
    return results


def load_qa_eval_set(dataset_name: str = "nq_open", split: str = "validation", num_samples: int = 200):
    """Load open-ended QA dataset."""
    from src.generation.datasets import get_dataset
    print(f"Loading {dataset_name} {split} set...")
    dataset = get_dataset(dataset_name, split=split, num_samples=num_samples)
    dataset.load(None)
    samples = []
    for i, item in enumerate(dataset.data):
        samples.append({
            'id': f"{dataset_name}_{i}",
            'question': item['question'],
            'answer': item['answer'],
            'all_answers': item.get('all_answers', [item['answer']])
        })
    print(f"Loaded {len(samples)} QA samples")
    return samples


def check_answer_match(generated_text: str, gold_answers: list) -> bool:
    """Check if gold answer appears in generated text with intelligent matching."""
    gen_lower = generated_text.lower().strip()

    number_words = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'twenty': '20', 'thirty': '30'
    }
    word_to_number = number_words
    number_to_word = {v: k for k, v in number_words.items()}

    for gold in gold_answers:
        gold_lower = gold.lower().strip()

        if gold_lower in gen_lower:
            return True

        gold_words = gold_lower.split()
        if len(gold_words) == 1:
            pattern = r'\b' + re.escape(gold_lower) + r'\b'
            if re.search(pattern, gen_lower):
                return True
            if gold_lower in word_to_number:
                number_pattern = r'\b' + re.escape(word_to_number[gold_lower]) + r'\b'
                if re.search(number_pattern, gen_lower):
                    return True
            if gold_lower in number_to_word:
                word_pattern = r'\b' + re.escape(number_to_word[gold_lower]) + r'\b'
                if re.search(word_pattern, gen_lower):
                    return True

        gold_numbers = re.findall(r'\b\d{4}\b|\b\d+\b', gold_lower)
        gen_numbers = re.findall(r'\b\d{4}\b|\b\d+\b', gen_lower)
        gold_years = [n for n in gold_numbers if len(n) == 4]
        gen_years = [n for n in gen_numbers if len(n) == 4]
        if gold_years and gen_years:
            if any(gy in gold_years for gy in gen_years):
                return True

    return False


def evaluate_generation_on_samples(model, tokenizer, samples, max_new_tokens=10, device="cuda"):
    """Evaluate open-ended generation on all samples with token-level entropy."""
    results = []
    for sample in tqdm(samples, desc="Evaluating generation", leave=False):
        question = sample['question']
        gold_answers = sample['all_answers']
        prompt = f"Answer this question concisely in 1-5 words.\n\nQuestion: {question}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        # Manual autoregressive generation to capture token-level entropy
        input_ids = inputs.input_ids
        token_entropies = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]  # [1, vocab_size]

                # Compute entropy of the distribution over vocab
                probs = torch.softmax(next_token_logits, dim=-1)
                log_probs = torch.log_softmax(next_token_logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).item()
                token_entropies.append(entropy)

                # Greedy select next token
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

                # Stop if EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break

                input_ids = torch.cat([input_ids, next_token], dim=-1)

        generated_ids = input_ids
        generated_text = tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)
        is_correct = check_answer_match(generated_text, gold_answers)
        gen_length = len(generated_ids[0]) - prompt_len
        mean_entropy = np.mean(token_entropies) if token_entropies else 0.0

        results.append({
            'sample_id': sample['id'],
            'question': question,
            'gold_answers': gold_answers,
            'generated_text': generated_text,
            'is_correct': is_correct,
            'generated_length': gen_length,
            'mean_token_entropy': mean_entropy,
            'token_entropies': token_entropies,
        })

    return results


def get_original_weight(model, layer_idx, matrix_type, model_type):
    """Get a clone of the original weight matrix from the model."""
    if model_type == "llama":
        if matrix_type == "mlp_in":
            return model.model.layers[layer_idx].mlp.up_proj.weight.data.clone()
        elif matrix_type == "mlp_out":
            return model.model.layers[layer_idx].mlp.down_proj.weight.data.clone()
        elif matrix_type == "gate_proj":
            return model.model.layers[layer_idx].mlp.gate_proj.weight.data.clone()
        elif matrix_type == "attn_q":
            return model.model.layers[layer_idx].self_attn.q_proj.weight.data.clone()
        elif matrix_type == "attn_k":
            return model.model.layers[layer_idx].self_attn.k_proj.weight.data.clone()
        elif matrix_type == "attn_v":
            return model.model.layers[layer_idx].self_attn.v_proj.weight.data.clone()
        elif matrix_type == "attn_o":
            return model.model.layers[layer_idx].self_attn.o_proj.weight.data.clone()
        else:
            raise ValueError(f"Unknown matrix_type: {matrix_type}")
    elif model_type == "gpt2":
        if matrix_type == "mlp_in":
            return model.transformer.h[layer_idx].mlp.c_fc.weight.data.clone()
        elif matrix_type == "mlp_out":
            return model.transformer.h[layer_idx].mlp.c_proj.weight.data.clone()
        else:
            raise ValueError(f"Unknown matrix_type: {matrix_type}")
    elif model_type == "gptj":
        if matrix_type == "mlp_in":
            return model.transformer.h[layer_idx].mlp.fc_in.weight.data.clone()
        elif matrix_type == "mlp_out":
            return model.transformer.h[layer_idx].mlp.fc_out.weight.data.clone()
        else:
            raise ValueError(f"Unknown matrix_type: {matrix_type}")
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


# =============================================================================
# Core SVD noise removal logic
# =============================================================================

def reconstruct_with_chunk_removed(U, S, Vh, chunk_start, chunk_end):
    """
    Reconstruct weight matrix with a chunk of singular values dropped.

    Drops components [chunk_start:chunk_end] and reconstructs from the
    remaining components only.

    Args:
        U: Left singular vectors [m, k]
        S: Singular values [k] (sorted descending)
        Vh: Right singular vectors transposed [k, n]
        chunk_start: Start index of chunk to remove (inclusive)
        chunk_end: End index of chunk to remove (exclusive)

    Returns:
        W_modified: Reconstructed weight without the chunk [m, n]
        energy_removed: Fraction of energy in the removed chunk
    """
    # Build indices of components to KEEP (everything except the chunk)
    keep_indices = torch.cat([
        torch.arange(0, chunk_start, device=S.device),
        torch.arange(chunk_end, len(S), device=S.device)
    ])

    U_kept = U[:, keep_indices]
    S_kept = S[keep_indices]
    Vh_kept = Vh[keep_indices, :]

    W_modified = reconstruct_from_svd(U_kept, S_kept, Vh_kept)

    total_energy = torch.sum(S ** 2).item()
    removed_energy = torch.sum(S[chunk_start:chunk_end] ** 2).item()
    energy_removed = removed_energy / total_energy if total_energy > 0 else 0.0

    return W_modified, energy_removed


def reconstruct_with_chunks_removed(U, S, Vh, chunk_ranges):
    """
    Reconstruct weight matrix with multiple chunks of singular values removed.

    Args:
        U: Left singular vectors [m, k]
        S: Singular values [k]
        Vh: Right singular vectors transposed [k, n]
        chunk_ranges: List of (start, end) tuples for chunks to remove

    Returns:
        W_modified: Reconstructed weight without the chunks [m, n]
        energy_removed: Fraction of total energy in the removed chunks
    """
    # Build set of all indices to remove
    remove_indices = set()
    for start, end in chunk_ranges:
        remove_indices.update(range(start, end))

    # Keep everything NOT in the remove set
    all_indices = set(range(len(S)))
    keep_indices = sorted(all_indices - remove_indices)
    keep_indices = torch.tensor(keep_indices, device=S.device, dtype=torch.long)

    U_kept = U[:, keep_indices]
    S_kept = S[keep_indices]
    Vh_kept = Vh[keep_indices, :]

    W_modified = reconstruct_from_svd(U_kept, S_kept, Vh_kept)

    total_energy = torch.sum(S ** 2).item()
    remove_indices_t = torch.tensor(sorted(remove_indices), device=S.device, dtype=torch.long)
    removed_energy = torch.sum(S[remove_indices_t] ** 2).item()
    energy_removed = removed_energy / total_energy if total_energy > 0 else 0.0

    return W_modified, energy_removed


# =============================================================================
# Main experiment
# =============================================================================

def run_svd_noise_removal(
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    model_type: str = "llama",
    dataset: str = "both",
    mcq_eval_set_path: str = "data/eval_sets/eval_set_mcq_nq_open_200.json",
    qa_dataset_name: str = "nq_open",
    num_qa_samples: int = 200,
    target_layer: int = 31,
    matrix_types: list = None,
    chunk_size: int = 100,
    max_new_tokens: int = 10,
    device: str = "cuda",
    checkpoint_every: int = 5
):
    """
    Run SVD noise removal experiment.

    For each chunk of singular values, zero them out and measure the impact
    using MCQ, Open QA, or both evaluation modes.

    Args:
        model_name: HuggingFace model name
        model_type: Model architecture ('llama', 'gpt2', 'gptj')
        dataset: Evaluation mode - 'mcq', 'open_qa', or 'both'
        mcq_eval_set_path: Path to MCQ evaluation set JSON
        qa_dataset_name: Open QA dataset name
        num_qa_samples: Number of QA samples for generation evaluation
        target_layer: Which layer to decompose
        matrix_types: List of matrix types to decompose (e.g., ['mlp_out'] or ['mlp_in', 'mlp_out'])
        chunk_size: Number of singular values per chunk (components per chunk)
        max_new_tokens: Max tokens to generate per QA answer
        device: Device for computation
        checkpoint_every: Save checkpoint every N chunks
    """
    seed_everything(42)

    if matrix_types is None:
        matrix_types = ["mlp_out"]

    run_mcq = dataset in ("mcq", "both")
    run_qa = dataset in ("open_qa", "both")

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1]
    matrix_str = "+".join(matrix_types)
    output_dir = Path(f"results/svd_noise_removal/{model_short}_layer{target_layer}_{matrix_str}_chunk{chunk_size}_{dataset}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SVD NOISE REMOVAL EXPERIMENT")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Layer: {target_layer}")
    print(f"Matrices: {matrix_types}")
    print(f"Chunk size: {chunk_size} components per chunk")
    print(f"Dataset mode: {dataset}")
    if run_mcq:
        print(f"MCQ eval set: {mcq_eval_set_path}")
    if run_qa:
        print(f"QA dataset: {qa_dataset_name} ({num_qa_samples} samples)")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Load evaluation sets
    mcq_samples = load_mcq_eval_set(mcq_eval_set_path) if run_mcq else None
    qa_samples = load_qa_eval_set(qa_dataset_name, num_samples=num_qa_samples) if run_qa else None

    # Save config
    config = {
        'experiment_type': 'svd_noise_removal',
        'model_name': model_name,
        'model_type': model_type,
        'dataset': dataset,
        'target_layer': target_layer,
        'matrix_types': matrix_types,
        'chunk_size': chunk_size,
        'mcq_eval_set_path': mcq_eval_set_path if run_mcq else None,
        'qa_dataset_name': qa_dataset_name if run_qa else None,
        'num_mcq_samples': len(mcq_samples) if run_mcq else 0,
        'num_qa_samples': len(qa_samples) if run_qa else 0,
        'max_new_tokens': max_new_tokens,
        'timestamp': timestamp,
    }

    # Load model
    print(f"\nLoading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    print("Model loaded.")

    # ========== Phase 2: Extract Weights and Decompose ==========
    print("\n" + "=" * 70)
    print("PHASE 2: SVD DECOMPOSITION")
    print("=" * 70)

    original_weights = {}
    decompositions = {}
    svd_stats = {}

    for mt in matrix_types:
        print(f"\n  Decomposing {mt}...")
        original_weights[mt] = get_original_weight(model, target_layer, mt, model_type)
        U, S, Vh = decompose_weight_svd(original_weights[mt], device)
        decompositions[mt] = {'U': U, 'S': S, 'Vh': Vh}
        svd_stats[mt] = get_svd_stats(original_weights[mt], device)
        print(f"    Shape: {svd_stats[mt]['shape']}, Rank: {len(S)}, Effective rank: {svd_stats[mt]['effective_rank']:.1f}")

    # Determine chunk layout
    total_components = min(len(decompositions[mt]['S']) for mt in matrix_types)
    num_chunks = (total_components + chunk_size - 1) // chunk_size

    config['total_components'] = total_components
    config['num_chunks'] = num_chunks

    print(f"\nTotal singular values: {total_components}")
    print(f"Chunk size: {chunk_size}")
    print(f"Number of chunks: {num_chunks}")
    last_chunk_size = total_components - (num_chunks - 1) * chunk_size
    if last_chunk_size != chunk_size:
        print(f"Last chunk has {last_chunk_size} components (indices {(num_chunks-1)*chunk_size}-{total_components-1})")

    # ========== Phase 3: Baseline ==========
    print("\n" + "=" * 70)
    print("PHASE 3: BASELINE EVALUATION")
    print("=" * 70)

    baseline_mcq = None
    baseline_mcq_entropy = baseline_mcq_nll = baseline_mcq_accuracy = None
    baseline_qa = None
    baseline_qa_accuracy = baseline_qa_avg_length = baseline_qa_entropy = None

    if run_mcq:
        print("\nEvaluating MCQ baseline...")
        baseline_mcq = evaluate_mcq_on_samples(model, tokenizer, mcq_samples, device)
        baseline_mcq_entropy = np.mean([r['entropy'] for r in baseline_mcq])
        baseline_mcq_nll = np.mean([r['nll'] for r in baseline_mcq])
        baseline_mcq_accuracy = np.mean([r['is_correct'] for r in baseline_mcq])

        print(f"  MCQ entropy: {baseline_mcq_entropy:.4f}")
        print(f"  MCQ NLL: {baseline_mcq_nll:.4f}")
        print(f"  MCQ accuracy: {baseline_mcq_accuracy*100:.1f}%")

    if run_qa:
        print("\nEvaluating QA baseline...")
        baseline_qa = evaluate_generation_on_samples(model, tokenizer, qa_samples, max_new_tokens, device)
        baseline_qa_accuracy = sum(r['is_correct'] for r in baseline_qa) / len(baseline_qa)
        baseline_qa_avg_length = np.mean([r['generated_length'] for r in baseline_qa])
        baseline_qa_entropy = np.mean([r['mean_token_entropy'] for r in baseline_qa])

        print(f"  QA accuracy: {baseline_qa_accuracy*100:.1f}%")
        print(f"  QA entropy: {baseline_qa_entropy:.4f}")
        print(f"  QA avg length: {baseline_qa_avg_length:.1f} tokens")

    # ========== Phase 4: Chunk Removal Loop ==========
    print("\n" + "=" * 70)
    print(f"PHASE 4: CHUNK REMOVAL ({num_chunks} chunks)")
    print("=" * 70)

    chunk_results = []

    for chunk_idx in tqdm(range(num_chunks), desc="Testing chunks"):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, total_components)
        actual_chunk_size = chunk_end - chunk_start

        # Step A: Remove chunk from ALL matrices
        energy_removed = {}
        for mt in matrix_types:
            U = decompositions[mt]['U']
            S = decompositions[mt]['S']
            Vh = decompositions[mt]['Vh']

            W_modified, e_removed = reconstruct_with_chunk_removed(
                U, S, Vh, chunk_start, chunk_end
            )
            energy_removed[mt] = e_removed
            update_layer_with_svd(model, target_layer, W_modified, mt, model_type)

        # Step B: Evaluate MCQ
        mcq_entropy = mcq_nll = mcq_accuracy = None
        mcq_entropy_change = mcq_accuracy_change = None
        mcq_eval = None
        if run_mcq:
            mcq_eval = evaluate_mcq_on_samples(model, tokenizer, mcq_samples, device)
            mcq_entropy = np.mean([r['entropy'] for r in mcq_eval])
            mcq_nll = np.mean([r['nll'] for r in mcq_eval])
            mcq_accuracy = np.mean([r['is_correct'] for r in mcq_eval])
            mcq_entropy_change = mcq_entropy - baseline_mcq_entropy
            mcq_accuracy_change = mcq_accuracy - baseline_mcq_accuracy

        # Step C: Evaluate Open QA
        qa_accuracy = qa_avg_length = qa_entropy = None
        qa_accuracy_change = qa_entropy_change = None
        qa_eval = None
        if run_qa:
            qa_eval = evaluate_generation_on_samples(model, tokenizer, qa_samples, max_new_tokens, device)
            qa_accuracy = sum(r['is_correct'] for r in qa_eval) / len(qa_eval)
            qa_avg_length = np.mean([r['generated_length'] for r in qa_eval])
            qa_entropy = np.mean([r['mean_token_entropy'] for r in qa_eval])
            qa_accuracy_change = qa_accuracy - baseline_qa_accuracy
            qa_entropy_change = qa_entropy - baseline_qa_entropy

        # Step D: Store result
        result = {
            'chunk_idx': chunk_idx,
            'chunk_start': chunk_start,
            'chunk_end': chunk_end,
            'chunk_size': actual_chunk_size,
            'energy_removed': energy_removed,
            'avg_energy_removed': np.mean(list(energy_removed.values())),
        }
        if run_mcq:
            result.update({
                'mcq_entropy': mcq_entropy,
                'mcq_nll': mcq_nll,
                'mcq_accuracy': mcq_accuracy,
                'mcq_entropy_change': mcq_entropy_change,
                'mcq_nll_change': mcq_nll - baseline_mcq_nll,
                'mcq_accuracy_change': mcq_accuracy_change,
                'mcq_per_sample': mcq_eval,
            })
        if run_qa:
            result.update({
                'qa_accuracy': qa_accuracy,
                'qa_entropy': qa_entropy,
                'qa_avg_length': qa_avg_length,
                'qa_accuracy_change': qa_accuracy_change,
                'qa_entropy_change': qa_entropy_change,
                'qa_length_change': qa_avg_length - baseline_qa_avg_length,
                'qa_per_sample': qa_eval,
            })
        chunk_results.append(result)

        # Step E: Print progress
        parts = [f"  Chunk {chunk_idx} [{chunk_start}:{chunk_end}]:"]
        if run_mcq:
            parts.append(f"MCQ ent {mcq_entropy_change:+.4f}, MCQ acc {mcq_accuracy_change*100:+.1f}pp,")
        if run_qa:
            parts.append(f"QA ent {qa_entropy_change:+.4f}, QA acc {qa_accuracy_change*100:+.1f}pp,")
        parts.append(f"energy rm {np.mean(list(energy_removed.values()))*100:.2f}%")
        print(" ".join(parts))

        # Step G: Restore ALL original weights
        for mt in matrix_types:
            restore_original_weight(model, target_layer, original_weights[mt], mt, model_type)

        # Step H: Checkpoint
        if (chunk_idx + 1) % checkpoint_every == 0:
            checkpoint = {
                'config': config,
                'svd_stats': svd_stats,
                'chunk_results': chunk_results,
                'completed_chunks': chunk_idx + 1
            }
            if run_mcq:
                checkpoint['baseline_mcq'] = {
                    'avg_entropy': baseline_mcq_entropy,
                    'avg_nll': baseline_mcq_nll,
                    'accuracy': baseline_mcq_accuracy,
                    'per_sample': baseline_mcq
                }
            if run_qa:
                checkpoint['baseline_qa'] = {
                    'accuracy': baseline_qa_accuracy,
                    'avg_entropy': baseline_qa_entropy,
                    'avg_length': baseline_qa_avg_length,
                    'per_sample': baseline_qa
                }
            with open(output_dir / 'checkpoint.pkl', 'wb') as f:
                pickle.dump(checkpoint, f)

    # ========== Phase 5: Classification ==========
    print("\n" + "=" * 70)
    print("PHASE 5: CHUNK CLASSIFICATION")
    print("=" * 70)

    # MCQ-based classification — labels describe what the COMPONENTS are
    # Removing chunk → model improves (entropy↓ accuracy↑) → components were NOISE
    # Removing chunk → model degrades (entropy↑ accuracy↓) → components were SIGNAL
    true_noise = []       # entropy↓ accuracy↑: removing helped → components are noise
    confident_wrong = []  # entropy↓ accuracy↓: more confident but wrong
    true_signal = []      # entropy↑ accuracy↓: removing hurt → components are signal
    uncertain_right = []  # entropy↑ accuracy↑: more uncertain but right

    if run_mcq:
        for c in chunk_results:
            if c['mcq_entropy_change'] < 0 and c['mcq_accuracy_change'] > 0:
                true_noise.append(c['chunk_idx'])
            elif c['mcq_entropy_change'] < 0 and c['mcq_accuracy_change'] <= 0:
                confident_wrong.append(c['chunk_idx'])
            elif c['mcq_entropy_change'] >= 0 and c['mcq_accuracy_change'] <= 0:
                true_signal.append(c['chunk_idx'])
            else:
                uncertain_right.append(c['chunk_idx'])

        print(f"MCQ-based classification:")
        print(f"  TRUE NOISE (entropy↓ accuracy↑): {len(true_noise)} chunks")
        print(f"  Confident Wrong (entropy↓ accuracy↓): {len(confident_wrong)} chunks")
        print(f"  TRUE SIGNAL (entropy↑ accuracy↓): {len(true_signal)} chunks")
        print(f"  Uncertain Right (entropy↑ accuracy↑): {len(uncertain_right)} chunks")

    # QA-based classification — labels describe what the COMPONENTS are (same logic as MCQ)
    qa_true_noise = []       # entropy↓ accuracy↑: removing helped → components are noise
    qa_confident_wrong = []  # entropy↓ accuracy↓: more confident but wrong
    qa_true_signal = []      # entropy↑ accuracy↓: removing hurt → components are signal
    qa_uncertain_right = []  # entropy↑ accuracy↑: more uncertain but right

    if run_qa:
        for c in chunk_results:
            if c['qa_entropy_change'] < 0 and c['qa_accuracy_change'] > 0:
                qa_true_noise.append(c['chunk_idx'])
            elif c['qa_entropy_change'] < 0 and c['qa_accuracy_change'] <= 0:
                qa_confident_wrong.append(c['chunk_idx'])
            elif c['qa_entropy_change'] >= 0 and c['qa_accuracy_change'] <= 0:
                qa_true_signal.append(c['chunk_idx'])
            else:
                qa_uncertain_right.append(c['chunk_idx'])

        print(f"\nQA-based classification:")
        print(f"  TRUE NOISE (entropy↓ accuracy↑): {len(qa_true_noise)} chunks")
        print(f"  Confident Wrong (entropy↓ accuracy↓): {len(qa_confident_wrong)} chunks")
        print(f"  TRUE SIGNAL (entropy↑ accuracy↓): {len(qa_true_signal)} chunks")
        print(f"  Uncertain Right (entropy↑ accuracy↑): {len(qa_uncertain_right)} chunks")

    # ========== Phase 5b: Cumulative Noise Removal ==========
    # Stack TRUE NOISE chunks one by one and measure cumulative effect
    # These are the junk components — removing more should keep improving the model
    noise_chunk_indices = []
    if run_mcq and true_noise:
        noise_chunk_indices = true_noise
        noise_source = "mcq"
    if run_qa and qa_true_noise:
        # If both, prefer QA since that's the one just run; if only QA, use it
        noise_chunk_indices = qa_true_noise
        noise_source = "qa"
    if not run_qa and run_mcq and true_noise:
        noise_chunk_indices = true_noise
        noise_source = "mcq"

    cumulative_results = []
    if noise_chunk_indices:
        print("\n" + "=" * 70)
        print(f"PHASE 5b: CUMULATIVE NOISE REMOVAL ({len(noise_chunk_indices)} noise chunks from {noise_source})")
        print("=" * 70)

        # Order noise chunks by their index (low → high singular values first)
        noise_chunk_indices_sorted = sorted(noise_chunk_indices)

        # Map chunk index → (start, end) range
        chunk_range_map = {c['chunk_idx']: (c['chunk_start'], c['chunk_end']) for c in chunk_results}

        stacked_ranges = []
        for i, ci in enumerate(tqdm(noise_chunk_indices_sorted, desc="Cumulative noise removal")):
            stacked_ranges.append(chunk_range_map[ci])

            # Remove all stacked chunks from all matrices
            energy_removed = {}
            for mt in matrix_types:
                U = decompositions[mt]['U']
                S = decompositions[mt]['S']
                Vh = decompositions[mt]['Vh']

                W_modified, e_removed = reconstruct_with_chunks_removed(
                    U, S, Vh, stacked_ranges
                )
                energy_removed[mt] = e_removed
                update_layer_with_svd(model, target_layer, W_modified, mt, model_type)

            # Evaluate
            cum_result = {
                'step': i,
                'num_chunks_removed': i + 1,
                'chunk_indices_removed': list(noise_chunk_indices_sorted[:i+1]),
                'ranges_removed': list(stacked_ranges),
                'energy_removed': energy_removed,
                'avg_energy_removed': np.mean(list(energy_removed.values())),
            }

            if run_mcq:
                mcq_eval = evaluate_mcq_on_samples(model, tokenizer, mcq_samples, device)
                cum_mcq_entropy = np.mean([r['entropy'] for r in mcq_eval])
                cum_mcq_accuracy = np.mean([r['is_correct'] for r in mcq_eval])
                cum_result.update({
                    'mcq_entropy': cum_mcq_entropy,
                    'mcq_accuracy': cum_mcq_accuracy,
                    'mcq_entropy_change': cum_mcq_entropy - baseline_mcq_entropy,
                    'mcq_accuracy_change': cum_mcq_accuracy - baseline_mcq_accuracy,
                })

            if run_qa:
                qa_eval = evaluate_generation_on_samples(model, tokenizer, qa_samples, max_new_tokens, device)
                cum_qa_accuracy = sum(r['is_correct'] for r in qa_eval) / len(qa_eval)
                cum_qa_entropy = np.mean([r['mean_token_entropy'] for r in qa_eval])
                cum_result.update({
                    'qa_accuracy': cum_qa_accuracy,
                    'qa_entropy': cum_qa_entropy,
                    'qa_accuracy_change': cum_qa_accuracy - baseline_qa_accuracy,
                    'qa_entropy_change': cum_qa_entropy - baseline_qa_entropy,
                })

            cumulative_results.append(cum_result)

            # Print progress
            parts = [f"  Stack {i+1} (removed {i+1} chunks):"]
            if run_mcq:
                parts.append(f"MCQ ent {cum_result['mcq_entropy_change']:+.4f}, MCQ acc {cum_result['mcq_accuracy_change']*100:+.1f}pp,")
            if run_qa:
                parts.append(f"QA ent {cum_result['qa_entropy_change']:+.4f}, QA acc {cum_result['qa_accuracy_change']*100:+.1f}pp,")
            parts.append(f"energy rm {cum_result['avg_energy_removed']*100:.2f}%")
            print(" ".join(parts))

            # Restore original weights
            for mt in matrix_types:
                restore_original_weight(model, target_layer, original_weights[mt], mt, model_type)
    else:
        print("\nNo TRUE NOISE chunks found — skipping cumulative noise removal.")

    # ========== Phase 6: Save Results ==========
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results = {
        'config': config,
        'svd_stats': svd_stats,
        'chunk_results': chunk_results,
        'cumulative_results': cumulative_results,
        'rankings': {
            'by_energy_removed': sorted(
                [(c['chunk_idx'], c['avg_energy_removed']) for c in chunk_results],
                key=lambda x: x[1], reverse=True
            ),
        }
    }

    if run_mcq:
        results['baseline_mcq'] = {
            'avg_entropy': baseline_mcq_entropy,
            'avg_nll': baseline_mcq_nll,
            'accuracy': baseline_mcq_accuracy,
            'per_sample': baseline_mcq
        }
        results['rankings'].update({
            'by_mcq_entropy': sorted(
                [(c['chunk_idx'], c['mcq_entropy_change']) for c in chunk_results],
                key=lambda x: x[1]
            ),
            'by_mcq_accuracy': sorted(
                [(c['chunk_idx'], c['mcq_accuracy_change']) for c in chunk_results],
                key=lambda x: x[1], reverse=True
            ),
            'true_signal': true_signal,
            'confident_wrong': confident_wrong,
            'true_noise': true_noise,
            'uncertain_right': uncertain_right,
        })

    if run_qa:
        results['baseline_qa'] = {
            'accuracy': baseline_qa_accuracy,
            'avg_entropy': baseline_qa_entropy,
            'avg_length': baseline_qa_avg_length,
            'per_sample': baseline_qa
        }
        results['rankings'].update({
            'by_qa_entropy': sorted(
                [(c['chunk_idx'], c['qa_entropy_change']) for c in chunk_results],
                key=lambda x: x[1]
            ),
            'by_qa_accuracy': sorted(
                [(c['chunk_idx'], c['qa_accuracy_change']) for c in chunk_results],
                key=lambda x: x[1], reverse=True
            ),
            'qa_true_signal': qa_true_signal,
            'qa_confident_wrong': qa_confident_wrong,
            'qa_true_noise': qa_true_noise,
            'qa_uncertain_right': qa_uncertain_right,
        })

    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to: {output_dir / 'results.pkl'}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Build header
    header_parts = [f"{'Chunk':>6}", f"{'Range':>12}"]
    if run_mcq:
        header_parts += [f"{'MCQ Ent':>10}", f"{'MCQ Acc':>10}"]
    if run_qa:
        header_parts += [f"{'QA Ent':>10}", f"{'QA Acc':>10}"]
    header_parts += [f"{'Energy Rm':>10}"]
    if run_mcq:
        header_parts += [f"{'MCQ Class':>16}"]
    if run_qa:
        header_parts += [f"{'QA Class':>16}"]
    print("\n" + " ".join(header_parts))
    print("-" * len(" ".join(header_parts)))

    for c in chunk_results:
        mcq_classification = ""
        if run_mcq:
            ci = c['chunk_idx']
            if ci in true_signal:
                mcq_classification = "TRUE SIGNAL"
            elif ci in confident_wrong:
                mcq_classification = "Confident Wrong"
            elif ci in true_noise:
                mcq_classification = "TRUE NOISE"
            elif ci in uncertain_right:
                mcq_classification = "Uncertain Right"

        qa_classification = ""
        if run_qa:
            ci = c['chunk_idx']
            if ci in qa_true_signal:
                qa_classification = "TRUE SIGNAL"
            elif ci in qa_confident_wrong:
                qa_classification = "Confident Wrong"
            elif ci in qa_true_noise:
                qa_classification = "TRUE NOISE"
            elif ci in qa_uncertain_right:
                qa_classification = "Uncertain Right"

        row = f"{c['chunk_idx']:>6} [{c['chunk_start']:>4}:{c['chunk_end']:>4}]"
        if run_mcq:
            row += f" {c['mcq_entropy_change']:>+9.4f} {c['mcq_accuracy_change']*100:>+9.1f}pp"
        if run_qa:
            row += f" {c['qa_entropy_change']:>+9.4f} {c['qa_accuracy_change']*100:>+9.1f}pp"
        row += f" {c['avg_energy_removed']*100:>9.2f}%"
        if run_mcq:
            row += f" {mcq_classification:>16}"
        if run_qa:
            row += f" {qa_classification:>16}"
        print(row)

    if run_mcq:
        print(f"\nBaseline MCQ: {baseline_mcq_accuracy*100:.1f}% accuracy, {baseline_mcq_entropy:.4f} entropy")
    if run_qa:
        print(f"Baseline QA: {baseline_qa_accuracy*100:.1f}% accuracy, {baseline_qa_entropy:.4f} entropy")

    # Cumulative stacking summary
    if cumulative_results:
        print(f"\n{'='*70}")
        print("CUMULATIVE NOISE REMOVAL SUMMARY")
        print(f"{'='*70}")

        cum_header = [f"{'Step':>5}", f"{'#Chunks':>8}"]
        if run_mcq:
            cum_header += [f"{'MCQ Ent':>10}", f"{'MCQ Acc':>10}"]
        if run_qa:
            cum_header += [f"{'QA Ent':>10}", f"{'QA Acc':>10}"]
        cum_header += [f"{'Energy Rm':>10}"]
        print(" ".join(cum_header))
        print("-" * len(" ".join(cum_header)))

        for cr in cumulative_results:
            row = f"{cr['step']+1:>5} {cr['num_chunks_removed']:>8}"
            if run_mcq:
                row += f" {cr['mcq_entropy_change']:>+9.4f} {cr['mcq_accuracy_change']*100:>+9.1f}pp"
            if run_qa:
                row += f" {cr['qa_entropy_change']:>+9.4f} {cr['qa_accuracy_change']*100:>+9.1f}pp"
            row += f" {cr['avg_energy_removed']*100:>9.2f}%"
            print(row)

    print(f"\nOutput directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SVD Noise Removal - Remove chunks of singular values and evaluate impact"
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--model-type", type=str, default="llama",
                        choices=["llama", "gpt2", "gptj"])
    parser.add_argument("--dataset", type=str, default="both",
                        choices=["mcq", "open_qa", "both"],
                        help="Which evaluation to run: 'mcq', 'open_qa', or 'both'")
    parser.add_argument("--layer", type=int, default=31,
                        help="Target layer for SVD noise removal")
    parser.add_argument("--matrix", type=str, nargs="+", default=["mlp_out"],
                        choices=["mlp_in", "mlp_out", "gate_proj", "attn_q", "attn_k", "attn_v", "attn_o"],
                        help="Which weight matrix(es) to decompose. Multiple = drop same chunk from all.")
    parser.add_argument("--chunk-size", "--components-per-chunk", type=int, default=100,
                        dest="chunk_size",
                        help="Number of singular values (components) per chunk")
    parser.add_argument("--eval-set", type=str,
                        default="data/eval_sets/eval_set_mcq_nq_open_200.json",
                        help="Path to MCQ evaluation set JSON")
    parser.add_argument("--qa-dataset", type=str, default="nq_open",
                        choices=["nq_open", "hotpotqa", "coqa"],
                        help="Open QA dataset for generation evaluation")
    parser.add_argument("--num-qa-samples", type=int, default=200,
                        help="Number of QA samples for generation evaluation")
    parser.add_argument("--max-new-tokens", type=int, default=10,
                        help="Max tokens to generate per QA answer")
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="Save checkpoint every N chunks")
    parser.add_argument("--test", action="store_true",
                        help="Quick test mode: chunk-size=1024, 20 QA samples")

    args = parser.parse_args()

    if args.test:
        print("Running quick test mode...")
        run_svd_noise_removal(
            model_name=args.model,
            model_type=args.model_type,
            dataset=args.dataset,
            target_layer=args.layer,
            matrix_types=args.matrix,
            chunk_size=1024,
            mcq_eval_set_path=args.eval_set,
            qa_dataset_name=args.qa_dataset,
            num_qa_samples=20,
            max_new_tokens=args.max_new_tokens,
            checkpoint_every=1
        )
    else:
        run_svd_noise_removal(
            model_name=args.model,
            model_type=args.model_type,
            dataset=args.dataset,
            target_layer=args.layer,
            matrix_types=args.matrix,
            chunk_size=args.chunk_size,
            mcq_eval_set_path=args.eval_set,
            qa_dataset_name=args.qa_dataset,
            num_qa_samples=args.num_qa_samples,
            max_new_tokens=args.max_new_tokens,
            checkpoint_every=args.checkpoint_every
        )
