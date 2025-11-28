# Design Notes: Adapting Compression Policy to LLaMA/Mistral

**Status:** Draft / Proposal  
**Target:** LLaMA-2-7B / Mistral-7B  

---

## 1. Architectural Differences (GPT-Neo vs LLaMA)

| Feature | GPT-Neo 1.3B | LLaMA 7B | Impact on Compression |
| :--- | :--- | :--- | :--- |
| **Attention** | Standard Multi-Head | **GQA (Grouped Query Attention)** | K/V heads are already compressed (shared). Compressing them further via SVD might be redundant or destructive. |
| **Activation** | GeLU | **SiLU (SwiGLU)** | SwiGLU adds a gate projection (`w1`, `w2`, `w3`). More MLP parameters to potentially compress. |
| **Positional Emb** | Learned | **RoPE (Rotary)** | RoPE is applied to Q/K. SVD on Q/K might interfere with rotary semantics if not careful. |
| **Normalization** | LayerNorm | **RMSNorm** | Less sensitive to scaling, might be more robust to quantization noise. |

---

## 2. Proposed Compression Policy

Based on our findings with GPT-Neo (where `k_proj` was safe but `mlp` was not), we propose the following adaptation for LLaMA:

### 2.1 The "Safe" Baseline
*   **Target**: `q_proj`, `o_proj` (Attention Output).
*   **Avoid**: `k_proj`, `v_proj` (Due to GQA already reducing their dimensionality).
*   **Avoid**: `gate_proj`, `up_proj`, `down_proj` (MLP). Start with these dense.

### 2.2 The "Aggressive" Experiment
*   **Target**: MLP Layers (`gate_proj`, `up_proj`).
*   **Strategy**: Since LLaMA MLPs are huge (SwiGLU has 3 matrices), they are the biggest memory prize.
*   **Risk Mitigation**: Use **Higher Rank** (e.g., Rank 512 or 1024) instead of 128. LLaMA's 11000 intermediate dim is too rich for Rank 128.

---

## 3. Experiment Plan

### Phase 1: Sensitivity Scanning
Run the `layer_sensitivity.py` script on LLaMA-7B to identify "Pillar Layers".
*   **Hypothesis**: LLaMA might also have specific layers (like Layer 4 in GPT-Neo) that are "load-bearing" for Induction Heads.
*   **Metric**: Perplexity on WikiText2 after compressing *only* one layer.

### Phase 2: RoPE Compatibility
Verify if SVD on `q_proj` / `k_proj` destroys Rotary Embedding properties.
*   **Test**: Compare attention scores before/after compression.
*   **Fallback**: If RoPE breaks, only compress `o_proj` and MLP layers.

### Phase 3: Calibration
LLaMA is more sensitive to activation outliers.
*   **Action**: Implement **activation-aware scaling** (smoothquant-style) before SVD decomposition to handle outlier features.

---

## 4. Expected Results
*   **Goal**: Compress LLaMA-7B (13GB FP16) to < 8GB (to fit in consumer 8GB VRAM cards).
*   **Method**: 4-bit quantization (GPTQ/AWQ) is standard, but **SVD+INT8** offers a different trade-off: faster CPU inference (matrix-vector vs dequant-vector) and potential for fine-tuning the low-rank adapters.

---

## 5. Open Questions
1.  Does GQA make K/V projections incompressible?
2.  Is SwiGLU more or less robust to low-rank approximation than standard GeLU MLP?
