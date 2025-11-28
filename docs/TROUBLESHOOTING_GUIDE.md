# Troubleshooting & Problem Resolution Guide

This document maps the specific failure modes encountered during the development of the Semantic LLM Compressor to their root causes, evidence logs, and implemented solutions in the codebase.

---

## 1. Problem: Semantic Collapse ("Gibberish")

### Description
The model generates completely random or nonsensical tokens, losing all semantic coherence.
- **Example Output**: `"isedayedayangeredarrettilynangered..."`

### Root Cause
**Compression of MLP (Feed Forward) Layers.**
The MLP layers in GPT-Neo project the state into a high-dimensional space (4x hidden size) to access dense associative memories. Low-rank SVD approximation (Rank 128) destroys the sparsity required to retrieve these specific associations, causing the state vector to fall off the valid semantic manifold.

### Evidence & Logs
- **File**: [`docs/archive/BENCHMARK_REPORT_GPT_NEO.md`](docs/archive/BENCHMARK_REPORT_GPT_NEO.md)
  - See section "Generated Text Quality" showing the gibberish output.
  - See "Cosine Similarity: 0.102" (random/orthogonal).

### Solution
**Exclude MLP layers from compression.**
We implemented a filter to strictly keep all `mlp.c_fc` and `mlp.c_proj` layers in their original dense format.

### Code Reference
- **File**: [`semantic_llm_compressor/algorithms/factorization_policy.py`](semantic_llm_compressor/algorithms/factorization_policy.py)
- **Lines**: 65-66
  ```python
  "mlp.c_fc",            # MLP first layer - 78% SVD error at rank 128
  "mlp.c_proj",          # MLP second layer - equally sensitive
  ```

---

## 2. Problem: Repetition Loops ("Repetition Collapse")

### Description
The model generates coherent text initially but quickly falls into an infinite loop of repeating the same token or phrase.
- **Example Output**: `"The meaning of life is to to to to to..."`

### Root Cause
**Compression of Layer 4 (`transformer.h.4.attn.attention.k_proj`).**
This specific layer appears to act as a critical control mechanism (likely an "Induction Head" or inhibition-of-return mechanism). Even slight approximation errors in this layer break the model's ability to advance the context window, causing it to collapse to the most recent local attractor.

### Evidence & Logs
- **File**: [`docs/WAVES_NOTES.md`](docs/WAVES_NOTES.md)
  - See "Fase 3" and "Fase 4" detailing the discovery of the repetition loop and the isolation of Layer 4.
- **File**: [`docs/archive/DEBUG_SUMMARY.md`](docs/archive/DEBUG_SUMMARY.md)
  - See "Generated Text Examples" showing the repetitive behavior.

### Solution
**Surgically exclude Layer 4 from compression.**
We added an explicit exception for this specific layer in the compression policy.

### Code Reference
- **File**: [`semantic_llm_compressor/algorithms/factorization_policy.py`](semantic_llm_compressor/algorithms/factorization_policy.py)
- **Line**: 67
  ```python
  "h.4.attn.attention.k_proj", # Toxic layer causing repetition
  ```
- **File**: [`run_pipeline.py`](run_pipeline.py)
  - **Line**: 45
  ```python
  "exclude_layers": [4] # Critical fix from v2
  ```

---

## 3. Theoretical Implications

These findings confirm recent theories in Mechanistic Interpretability:
1.  **Manifold Collapse**: SVD on MLPs pushes activations off the semantic manifold.
2.  **Induction Head Fragility**: Layer 4 likely contains an Induction Head circuit that requires high precision to function, validating the "Attention Sink" and "Induction Head" theories (Olsson et al.).

For a deeper theoretical analysis, see [`docs/RELATORIO_IMPLICACOES_TEORICAS.md`](docs/RELATORIO_IMPLICACOES_TEORICAS.md) (in Portuguese).
