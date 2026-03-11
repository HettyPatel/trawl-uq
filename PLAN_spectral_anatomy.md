# Plan: Spectral Anatomy of Factual Noise in Transformers

## Context

The project investigates where "factual noise" lives in transformer weight matrices using SVD chunk removal. Partial results on Llama-2-7b (13 of 32 layers, MLP only) show a clear pattern: Layer 31 is pure noise (every chunk decreases entropy when removed), mid-layers (10-15) are signal-dominant, and Layer 1 is critical signal. The goal is to complete this into a mechanistic interpretability paper by extending to all layers, attention matrices, and a second model.

## Code Changes Required

### 1. Add attention matrix support to `experiments/14_svd_noise_removal.py`

**`get_original_weight()` (line 208-234)**: Add `elif` branches for `"attn_q"`, `"attn_k"`, `"attn_v"`, `"attn_o"` inside the `model_type == "llama"` block:
- `attn_q` → `model.model.layers[layer_idx].self_attn.q_proj.weight.data.clone()`
- `attn_k` → `model.model.layers[layer_idx].self_attn.k_proj.weight.data.clone()`
- `attn_v` → `model.model.layers[layer_idx].self_attn.v_proj.weight.data.clone()`
- `attn_o` → `model.model.layers[layer_idx].self_attn.o_proj.weight.data.clone()`

**Argparse `--matrix` choices (line 908-909)**: Expand from `["mlp_in", "mlp_out", "gate_proj"]` to also include `["attn_q", "attn_k", "attn_v", "attn_o"]`.

### 2. Add attention matrix support to `src/decomposition/svd.py`

**`update_layer_with_svd()` (line 364-372)**: Add same `elif` branches for `attn_q/k/v/o` pointing to `model.model.layers[layer_idx].self_attn.{q,k,v,o}_proj.weight`.

`restore_original_weight()` needs no changes (it delegates to `update_layer_with_svd`).

### 3. Create batch runner scripts

Shell scripts to run the full sweep (1 GPU at a time).

### 4. Create a new multi-layer noise removal experiment

A variant of experiment 14 that loads saved results, identifies TRUE NOISE chunks per layer, removes them from multiple layers simultaneously, and evaluates.

### 5. Create cross-layer heatmap plotting script

New plotting function that loads results from all 32 layers and produces the hero figure (layer x chunk heatmap colored by classification).

### 6. Create per-question analysis script

CPU-only script that loads all saved results and analyzes per-question sensitivity patterns.

---

## Experiments

### Experiment 1: Complete MLP Layer Sweep (Llama-2-7b) — PRIORITY 1

**Run**: Layers 5-9, 17-30 (19 remaining layers), `mlp_in+mlp_out`, chunk=100, MCQ only
**Time**: ~7 min/layer → ~2.2 hrs sequential on 1 GPU
**Produces**: Figure 1 (MLP noise heatmap, 32 layers x 41 chunks) + Table 1 (per-layer summary)
**Tests**: Whether noise accumulates gradually through later layers or spikes at layer 31

### Experiment 2: Attention Layer Sweep (Llama-2-7b) — PRIORITY 1

**Run**: All 32 layers, `attn_v` + `attn_o` separately (value pathway), chunk=100, MCQ only
**Time**: ~3.75 hrs sequential on 1 GPU
**Produces**: Figure 2 (Attention noise heatmap) + Figure 3 (MLP vs Attention noise fraction line plot)
**Tests**: Whether attention matrices have a different noise profile than MLPs

### Experiment 3: Cross-Model Validation (Llama-3-8B) — PRIORITY 2

**Run**: 10 representative layers [0,1,4,8,12,16,20,24,28,31], MLP and attention
**Note**: Llama-3 uses GQA — k_proj/v_proj are 1024x4096 (only 1024 SVs), so chunk=25 for those
**Time**: ~3.5 hrs sequential on 1 GPU
**Produces**: Figure 4 (cross-model comparison)
**Tests**: Whether the noise landscape transfers across architectures

### Experiment 4: Multi-Layer Simultaneous Noise Removal — PRIORITY 2

**Run**: Using results from Exp 1-2, remove TRUE NOISE from multiple layers at once
**3 configs**: (a) all layers, (b) only noise-dominant layers, (c) top-5 noisiest layers
**Time**: ~1.5 hr sequential
**Produces**: Figure 5 (practical accuracy improvement)
**Tests**: Whether noise removal is additive across layers

### Experiment 5: Per-Question Analysis — PRIORITY 1 (no GPU)

**Run**: CPU-only analysis of saved per-sample results
**Produces**: Figure 6 (per-question sensitivity heatmap)
**Tests**: Whether noise affects the same questions across layers or different questions

### Experiment 6: Gate Projection Sweep — PRIORITY 3 (supplementary)

**Run**: All 32 layers, `gate_proj` only, chunk=100, MCQ
**Produces**: Supplementary figure

### Experiment 7: Fine-Grained Key Layers — PRIORITY 3 (supplementary)

**Run**: Layers 1, 12, 31 with chunk=10 (410 chunks each)
**Produces**: Supplementary figure showing spectral fine structure

---

## GPU Schedule (1 GPU sequential, ~12 hours total for all, ~6 hours for minimum viable)

| Phase | Time | What |
|-------|------|------|
| 1 (MLP gap fill) | ~2.2 hr | Layers 5-9, 17-30, mlp_in+mlp_out |
| 2 (Attention sweep) | ~3.75 hr | All 32 layers, attn_v+attn_o |
| 3 (Llama-3) | ~3.5 hr | 10 representative layers, MLP+attention |
| 4 (Multi-layer removal) | ~1.5 hr | 3 configs |
| 5 (Per-question) | ~5 min | CPU only |

**Minimum viable**: Phases 1 + 2 + 5 = ~6 hours → full MLP + attention comparison + per-question analysis

---

## Paper Figures

| Figure | Content | Source |
|--------|---------|--------|
| Fig 1 | MLP Noise Landscape heatmap (32 layers x 41 chunks) | Exp 1 |
| Fig 2 | Attention Noise Landscape heatmap (32 layers x 41 chunks) | Exp 2 |
| Fig 3 | MLP vs Attention noise fraction by layer (line plot) | Exp 1+2 |
| Fig 4 | Cross-model comparison (Llama-2 vs Llama-3) | Exp 3 |
| Fig 5 | Multi-layer noise removal accuracy gain | Exp 4 |
| Fig 6 | Per-question sensitivity analysis | Exp 5 |

---

## Meeting Pitch Slides

### Slide 1: "SVD Chunk Removal Classifies Weight Components as Noise or Signal"

**Visual**: Side-by-side 4-quadrant scatter plots — Layer 31 (pure noise) vs Layer 1 (pure signal)

**Say**:
- We SVD decompose a weight matrix, remove chunks of 100 singular values one at a time, measure MCQ entropy and accuracy change
- This gives a 2x2 classification: TRUE NOISE (entropy down, accuracy up), TRUE SIGNAL (entropy up, accuracy down), and two mixed categories
- Layer 31: every single chunk is noise. Layer 1: removing chunk 0 alone causes +1.0 entropy and -28pp accuracy drop

### Slide 2: "Noise Distribution Across Layers Is Not Uniform"

**Visual**: Bar chart of net noise score by layer (layers 0-4, 10-16, 31 — the data you have). Gaps for missing layers marked as "to be completed"

**Say**:
- Early layers (0): slight noise. Layer 1: critical signal. Mid-layers (10-15): signal-dominant, peak at layer 12. Layer 31: pure noise
- This goes beyond LASER (which just truncates bottom SVs from late layers) — we map the full spectral structure at every layer
- The obvious next question: what about attention matrices? And does this generalize across models?

### Slide 3: "Proposed Paper: Spectral Anatomy of Factual Noise"

**Visual**: Transformer layer diagram with MLP + Attention blocks, each weight matrix labeled, color-coded as done/planned

**Say**:
- Complete the 32-layer sweep for MLP (19 layers remaining), then run all 32 layers for attention value pathway (v_proj + o_proj)
- This produces a full "noise map" — a heatmap showing exactly where factual noise concentrates in the model
- Second model (Llama-3) validates the pattern generalizes. Multi-layer removal shows the map is actionable (can improve accuracy)
- Connects to LASER, MLP-as-memory (Geva et al.), and knowledge localization (Meng et al.) — explains WHY LASER works and extends the picture to attention

### Slide 4: "Timeline and What We'll Show"

**Visual**: Table of experiments with estimated time (~12 hrs GPU total) and which figure each produces

**Say**:
- Minimum viable paper needs ~6 hours of GPU time: complete MLP sweep + attention sweep + per-question analysis
- The central contribution: first complete spectral map of factual noise across all weight matrices and layers in a transformer
- Practical payoff: principled noise removal across multiple layers to improve factual accuracy without retraining

---

## Verification

After code changes:
1. Run `--test` mode on layer 31 with `attn_v` to verify attention matrices work
2. Compare baseline MCQ accuracy (should be 55.0% for ARC Challenge, matching existing results)
3. Spot-check that `attn_v` chunk results at layer 31 produce a different noise profile from `mlp_in`
4. Run one full layer (e.g., layer 0) with `attn_v attn_o` to verify multi-matrix attention works
