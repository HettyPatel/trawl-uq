"""
Residual Stream Analysis Experiment

Measures the contribution of MLP and Attention blocks to the residual stream
across all layers using forward hooks. This experiment helps explain why SVD
truncation on MLP weights produces similar results regardless of which
components are kept (top, bottom, or random).

For each LlamaDecoderLayer:
    Attention: x_after_attn = x_before_attn + self_attn(layernorm(x_before_attn))
    MLP:       x_after_mlp  = x_before_mlp  + mlp(layernorm(x_before_mlp))

Metrics per layer:
    - Cosine similarity: cos(x_before, x_after) — directional change
    - Norm ratio: ||block_output|| / ||x_before|| — relative magnitude
    - Block norm: ||block_output|| — absolute magnitude
    - Residual norm: ||x_before|| — residual stream scale

Analysis:
    - Layer-wise profiles across all layers
    - Last token (prediction position) vs all-token average
    - Correct vs incorrect MCQ predictions

Usage:
    python experiments/11_residual_stream_analysis.py --model meta-llama/Llama-2-7b-chat-hf
    python experiments/11_residual_stream_analysis.py --model meta-llama/Meta-Llama-3-8B --test
"""

import sys
sys.path.append('.')

import json
import torch
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

from src.generation.generate import load_model_and_tokenizer, seed_everything


def load_mcq_eval_set(filepath: str):
    """Load MCQ evaluation set."""
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data['samples'])} MCQ samples from {filepath}")
    return data['samples']


class ResidualStreamHooks:
    """
    Registers forward hooks on all transformer layers to capture
    residual stream components before/after attention and MLP blocks.

    For Llama, each decoder layer does:
        # Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, ...)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

    We hook:
        - input_layernorm (input)    -> x_before_attn
        - self_attn (output[0])      -> attn_output
        - post_attention_layernorm (input) -> x_before_mlp
        - mlp (output)               -> mlp_output
    """

    def __init__(self, model, model_type="llama"):
        self.model = model
        self.model_type = model_type
        self.hooks = []
        self.activations = {}

    def _get_layers(self):
        if self.model_type == "llama":
            return self.model.model.layers
        elif self.model_type in ("gpt2", "gptj"):
            return self.model.transformer.h
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def register_hooks(self):
        """Register forward hooks on all layers."""
        layers = self._get_layers()

        for layer_idx, layer in enumerate(layers):
            if self.model_type == "llama":
                # input_layernorm input = x_before_attn
                self.hooks.append(
                    layer.input_layernorm.register_forward_hook(
                        self._make_hook(layer_idx, 'x_before_attn', capture='input')
                    )
                )
                # self_attn output[0] = attn_output
                self.hooks.append(
                    layer.self_attn.register_forward_hook(
                        self._make_hook(layer_idx, 'attn_output', capture='output_tuple')
                    )
                )
                # post_attention_layernorm input = x_before_mlp
                self.hooks.append(
                    layer.post_attention_layernorm.register_forward_hook(
                        self._make_hook(layer_idx, 'x_before_mlp', capture='input')
                    )
                )
                # mlp output = mlp_output
                self.hooks.append(
                    layer.mlp.register_forward_hook(
                        self._make_hook(layer_idx, 'mlp_output', capture='output')
                    )
                )

    def _make_hook(self, layer_idx, key, capture='input'):
        """Create a hook function that stores activations."""
        def hook_fn(module, input, output):
            if capture == 'input':
                self.activations.setdefault(layer_idx, {})[key] = input[0].detach()
            elif capture == 'output':
                self.activations.setdefault(layer_idx, {})[key] = output.detach()
            elif capture == 'output_tuple':
                self.activations.setdefault(layer_idx, {})[key] = output[0].detach()
        return hook_fn

    def clear(self):
        """Clear stored activations to free memory."""
        self.activations = {}

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def compute_residual_metrics(x_before, block_output, token_idx=None):
    """
    Compute residual stream metrics for a single layer and sample.

    Args:
        x_before: Residual stream before the block [1, seq_len, hidden_dim]
        block_output: Output of the block (attn or mlp) [1, seq_len, hidden_dim]
        token_idx: If provided, compute metrics only at this token position.
                   If None, average metrics over all positions.

    Returns:
        Dict with cosine_sim, norm_ratio, block_norm, residual_norm
    """
    x_after = x_before + block_output

    if token_idx is not None:
        # Single token position
        xb = x_before[0, token_idx, :]    # [hidden_dim]
        xa = x_after[0, token_idx, :]     # [hidden_dim]
        bo = block_output[0, token_idx, :] # [hidden_dim]

        cos_sim = torch.nn.functional.cosine_similarity(
            xb.unsqueeze(0), xa.unsqueeze(0)
        ).item()

        residual_norm = torch.norm(xb).item()
        block_norm = torch.norm(bo).item()
        norm_ratio = block_norm / (residual_norm + 1e-10)
    else:
        # Compute per-position then average
        xb = x_before[0]    # [seq_len, hidden_dim]
        xa = x_after[0]     # [seq_len, hidden_dim]
        bo = block_output[0] # [seq_len, hidden_dim]

        cos_sims = torch.nn.functional.cosine_similarity(xb, xa, dim=-1)
        cos_sim = cos_sims.mean().item()

        residual_norms = torch.norm(xb, dim=-1)
        block_norms = torch.norm(bo, dim=-1)
        norm_ratios = block_norms / (residual_norms + 1e-10)

        residual_norm = residual_norms.mean().item()
        block_norm = block_norms.mean().item()
        norm_ratio = norm_ratios.mean().item()

    return {
        'cosine_sim': cos_sim,
        'norm_ratio': norm_ratio,
        'block_norm': block_norm,
        'residual_norm': residual_norm
    }


def aggregate_layer_metrics(results_list, num_layers):
    """
    Aggregate per-layer metrics across samples.

    Args:
        results_list: List of sample result dicts
        num_layers: Number of model layers

    Returns:
        Dict mapping layer_idx -> block_key -> metric_key -> {mean, std, min, max}
    """
    agg = {}
    block_keys = ['attn_last_token', 'attn_all_tokens', 'mlp_last_token', 'mlp_all_tokens']

    for layer_idx in range(num_layers):
        layer_data = {k: defaultdict(list) for k in block_keys}

        for r in results_list:
            if layer_idx in r['layer_metrics']:
                lm = r['layer_metrics'][layer_idx]
                for block_key in block_keys:
                    for metric_key, value in lm[block_key].items():
                        layer_data[block_key][metric_key].append(value)

        agg[layer_idx] = {}
        for block_key in block_keys:
            agg[layer_idx][block_key] = {}
            for metric_key, values in layer_data[block_key].items():
                if values:
                    agg[layer_idx][block_key][metric_key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }
    return agg


def run_residual_stream_analysis(
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    model_type: str = "llama",
    eval_set_path: str = "data/eval_sets/eval_set_mcq_nq_open_200.json",
    device: str = "cuda",
    checkpoint_every: int = 25,
    max_samples: int = None
):
    """
    Run residual stream analysis experiment.

    Args:
        model_name: HuggingFace model name
        model_type: Model architecture ('llama', 'gpt2', 'gptj')
        eval_set_path: Path to MCQ evaluation set JSON
        device: Device for computation
        checkpoint_every: Save checkpoint every N samples
        max_samples: Limit number of samples (for testing)
    """
    seed_everything(42)

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1]
    output_dir = Path(f"results/residual_stream/{model_short}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RESIDUAL STREAM ANALYSIS EXPERIMENT")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Eval set: {eval_set_path}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Load eval set
    samples = load_mcq_eval_set(eval_set_path)
    if max_samples is not None:
        samples = samples[:max_samples]
        print(f"Limited to {max_samples} samples (test mode)")

    # Save config
    config = {
        'experiment_type': 'residual_stream_analysis',
        'model_name': model_name,
        'model_type': model_type,
        'eval_set_path': eval_set_path,
        'num_samples': len(samples),
        'timestamp': timestamp
    }

    # Load model
    print(f"\nLoading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)

    # Get number of layers
    if model_type == "llama":
        num_layers = len(model.model.layers)
    elif model_type in ("gpt2", "gptj"):
        num_layers = len(model.transformer.h)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    print(f"Number of layers: {num_layers}")
    config['num_layers'] = num_layers

    # Register hooks
    hooks = ResidualStreamHooks(model, model_type)
    hooks.register_hooks()
    print(f"Registered {len(hooks.hooks)} forward hooks ({len(hooks.hooks) // num_layers} per layer)")

    # ========== Run Analysis ==========
    print("\n" + "=" * 70)
    print("RUNNING ANALYSIS")
    print("=" * 70)

    all_sample_results = []

    for sample_idx, sample in enumerate(tqdm(samples, desc="Analyzing samples")):
        mcq_prompt = sample['mcq_prompt']
        correct_letter = sample['correct_letter']

        # Tokenize
        inputs = tokenizer(mcq_prompt, return_tensors="pt").to(device)
        seq_len = inputs.input_ids.shape[1]
        last_token_idx = seq_len - 1

        # Forward pass (hooks capture activations automatically)
        hooks.clear()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]

        # Get MCQ prediction
        options = ['A', 'B', 'C', 'D']
        option_token_ids = {}
        for letter in options:
            token_id = tokenizer.encode(f" {letter}", add_special_tokens=False)
            if len(token_id) == 1:
                option_token_ids[letter] = token_id[0]
            else:
                token_id = tokenizer.encode(letter, add_special_tokens=False)
                option_token_ids[letter] = token_id[0]

        option_logits = torch.tensor([logits[option_token_ids[l]].item() for l in options])
        option_probs = torch.nn.functional.softmax(option_logits, dim=0)
        predicted_idx = torch.argmax(option_probs).item()
        predicted_letter = options[predicted_idx]
        is_correct = predicted_letter == correct_letter

        entropy = -torch.sum(option_probs * torch.log(option_probs + 1e-10)).item()

        # Compute metrics for each layer
        layer_metrics = {}
        for layer_idx in range(num_layers):
            acts = hooks.activations.get(layer_idx, {})

            required_keys = ['x_before_attn', 'attn_output', 'x_before_mlp', 'mlp_output']
            if not all(k in acts for k in required_keys):
                continue

            # Attention metrics (last token and all tokens)
            attn_last = compute_residual_metrics(
                acts['x_before_attn'], acts['attn_output'], token_idx=last_token_idx
            )
            attn_all = compute_residual_metrics(
                acts['x_before_attn'], acts['attn_output'], token_idx=None
            )

            # MLP metrics (last token and all tokens)
            mlp_last = compute_residual_metrics(
                acts['x_before_mlp'], acts['mlp_output'], token_idx=last_token_idx
            )
            mlp_all = compute_residual_metrics(
                acts['x_before_mlp'], acts['mlp_output'], token_idx=None
            )

            layer_metrics[layer_idx] = {
                'attn_last_token': attn_last,
                'attn_all_tokens': attn_all,
                'mlp_last_token': mlp_last,
                'mlp_all_tokens': mlp_all
            }

        sample_result = {
            'sample_id': sample['id'],
            'question': sample['question'],
            'correct_letter': correct_letter,
            'predicted_letter': predicted_letter,
            'is_correct': is_correct,
            'entropy': entropy,
            'seq_len': seq_len,
            'option_probs': {l: option_probs[i].item() for i, l in enumerate(options)},
            'layer_metrics': layer_metrics
        }

        all_sample_results.append(sample_result)
        hooks.clear()

        # Checkpoint
        if (sample_idx + 1) % checkpoint_every == 0:
            checkpoint = {
                'config': config,
                'sample_results': all_sample_results,
                'completed_samples': sample_idx + 1
            }
            with open(output_dir / 'checkpoint.pkl', 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"\n  Checkpoint saved ({sample_idx + 1}/{len(samples)} samples)")

    # Remove hooks
    hooks.remove_hooks()

    # ========== Aggregate Results ==========
    print("\n" + "=" * 70)
    print("AGGREGATING RESULTS")
    print("=" * 70)

    correct_results = [r for r in all_sample_results if r['is_correct']]
    incorrect_results = [r for r in all_sample_results if not r['is_correct']]

    accuracy = len(correct_results) / len(all_sample_results)
    print(f"Total samples: {len(all_sample_results)}")
    print(f"Correct: {len(correct_results)} ({accuracy*100:.1f}%)")
    print(f"Incorrect: {len(incorrect_results)} ({(1-accuracy)*100:.1f}%)")

    aggregated_all = aggregate_layer_metrics(all_sample_results, num_layers)
    aggregated_correct = aggregate_layer_metrics(correct_results, num_layers) if correct_results else None
    aggregated_incorrect = aggregate_layer_metrics(incorrect_results, num_layers) if incorrect_results else None

    # Print summary tables
    print("\n--- MLP Contribution (last token, mean across samples) ---")
    print(f"{'Layer':>6} {'Cos Sim':>10} {'Norm Ratio':>12} {'MLP Norm':>10} {'Resid Norm':>12}")
    for layer_idx in range(num_layers):
        m = aggregated_all[layer_idx].get('mlp_last_token', {})
        if m:
            print(f"{layer_idx:>6} "
                  f"{m['cosine_sim']['mean']:>10.6f} "
                  f"{m['norm_ratio']['mean']:>12.6f} "
                  f"{m['block_norm']['mean']:>10.2f} "
                  f"{m['residual_norm']['mean']:>12.2f}")

    print("\n--- Attention Contribution (last token, mean across samples) ---")
    print(f"{'Layer':>6} {'Cos Sim':>10} {'Norm Ratio':>12} {'Attn Norm':>10} {'Resid Norm':>12}")
    for layer_idx in range(num_layers):
        m = aggregated_all[layer_idx].get('attn_last_token', {})
        if m:
            print(f"{layer_idx:>6} "
                  f"{m['cosine_sim']['mean']:>10.6f} "
                  f"{m['norm_ratio']['mean']:>12.6f} "
                  f"{m['block_norm']['mean']:>10.2f} "
                  f"{m['residual_norm']['mean']:>12.2f}")

    # ========== Save Results ==========
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results = {
        'config': config,
        'sample_results': all_sample_results,
        'aggregated_all': aggregated_all,
        'aggregated_correct': aggregated_correct,
        'aggregated_incorrect': aggregated_incorrect,
        'summary': {
            'num_samples': len(all_sample_results),
            'num_correct': len(correct_results),
            'num_incorrect': len(incorrect_results),
            'accuracy': accuracy
        }
    }

    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to: {output_dir / 'results.pkl'}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Residual stream MLP/attention contribution analysis"
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--model-type", type=str, default="llama",
                        choices=["llama", "gpt2", "gptj"])
    parser.add_argument("--eval-set", type=str,
                        default="data/eval_sets/eval_set_mcq_nq_open_200.json")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--test", action="store_true",
                        help="Quick test with 5 samples")

    args = parser.parse_args()

    kwargs = dict(
        model_name=args.model,
        model_type=args.model_type,
        eval_set_path=args.eval_set,
        checkpoint_every=args.checkpoint_every
    )

    if args.test:
        print("Running quick test with 5 samples...")
        kwargs['max_samples'] = 5
        kwargs['checkpoint_every'] = 2

    run_residual_stream_analysis(**kwargs)
