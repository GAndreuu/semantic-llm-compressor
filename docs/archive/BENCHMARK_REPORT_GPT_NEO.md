# GPT-Neo 1.3B Benchmark Report: Original vs Compressed

**Date:** November 25, 2025  
**Model:** EleutherAI/gpt-neo-1.3B  
**Compression:** Rank 128, INT8 Quantization, min_dim=1024  
**Device:** CPU  

---

## Executive Summary

This benchmark compares the original GPT-Neo 1.3B model against a compressed version using SVD (rank 128) + INT8 quantization. The results reveal **critical issues** that require investigation:

| Metric | Original | Compressed | Change |
|:---|---:|---:|---:|
| **Memory Usage** | 139.8 MB | 468.1 MB | **+234.8%** ⚠️ |
| **Inference Speed** | 3.98 tok/s | 18.02 tok/s | **+352.6%** ✅ |
| **Logits MSE** | - | 223.0 | - |
| **Cosine Similarity** | - | 0.102 | **Poor** ⚠️ |

### Key Findings

✅ **Speed Improvement**: The compressed model is **4.5x faster** (18.02 vs 3.98 tokens/sec)  
⚠️ **Memory Regression**: Compressed model uses **3.3x MORE memory** than original  
⚠️ **Quality Degradation**: Severe output quality issues (repetitive tokens, gibberish)

---

## Detailed Results

### 1. Memory Usage

```
Original Model:   139.83 MB
Compressed Model: 468.08 MB
Difference:      +328.25 MB (+234.8%)
```

**Analysis**: The compressed model is using significantly MORE memory than the original. This is unexpected and indicates:
- Possible issue with factor storage (U, S, Vh tensors not being freed)
- Original weights may not be properly released after patching
- Quantized factors might be loaded in FP32 instead of INT8

**Layers Patched**: 144/145 linear layers successfully replaced with `CompressedLinear`

### 2. Inference Speed

```
Original Model:
  - Avg Time: 12.56 sec (50 tokens)
  - Throughput: 3.98 tokens/sec
  - Std Dev: 0.072 sec

Compressed Model:
  - Avg Time: 2.77 sec (50 tokens)
  - Throughput: 18.02 tokens/sec
  - Std Dev: 0.028 sec
```

**Analysis**: The compressed model is **4.5x faster**, which aligns with the theoretical $O(r)$ complexity reduction. The lower standard deviation also suggests more consistent performance.

### 3. Quality Metrics

#### Logits Comparison

| Prompt | MSE | Max Diff | Cosine Sim |
|:---|---:|---:|---:|
| "The meaning of life is" | 302.84 | 47.95 | -0.530 |
| "Artificial intelligence will" | 221.97 | 47.31 | 0.233 |
| "In the future, technology" | 144.18 | 43.43 | 0.604 |
| **Average** | **222.99** | **46.23** | **0.102** |

**Analysis**: 
- **High MSE** (223.0): Logits are significantly different from the original
- **Low Cosine Similarity** (0.102): Output distributions are nearly orthogonal
- **Negative similarity** on first prompt indicates completely inverted predictions

#### Generated Text Quality

**Prompt**: "The meaning of life is"

**Original Output**:
> "The meaning of life is not a matter of life and death. It is a matter of life and love.
> 
> The meaning of life is not a matter of life and death. It is a matter of life and love.
> 
> The meaning of life is not a"

**Compressed Output**:
> "The meaning of life isedayedayangeredarrettilynangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangeredangered"

**Analysis**: The compressed model output is **completely degraded** - producing repetitive nonsense tokens. This indicates a fundamental issue with the compression or patching process.

---

## Root Cause Analysis

### Hypothesis 1: Conv1D Factor Transposition Issue
The `patcher.py` transposes factors for `Conv1D` layers:
```python
if is_conv1d:
    new_U = factors.Vh.t()
    new_Vh = factors.U.t()
```

**Potential Issue**: The transposition logic may be incorrect, causing weight reconstruction errors.

### Hypothesis 2: Quantization/Dequantization Error
The factors are quantized to INT8 during compression but must be dequantized during inference. Errors in scale factors or zero-points could cause severe degradation.

### Hypothesis 3: Rank Too Low for 1.3B Model
Rank 128 may be insufficient for a 1.3B parameter model, especially for critical layers like attention projections.

### Hypothesis 4: Memory Leak
The original weights are not being freed after patching, causing both models to coexist in memory.

---

## Recommendations

### Immediate Actions
1. **Fix Memory Leak**: Ensure original weights are freed after `CompressedLinear` replacement
2. **Verify Conv1D Logic**: Test the transposition logic with a simple unit test
3. **Check Dequantization**: Verify scale factors are correctly applied during `load_compressed_factors_for_weight`

### Future Improvements
1. **Increase Rank**: Test with rank 256 or 512 for better quality
2. **Selective Compression**: Skip critical layers (e.g., first/last layers, attention)
3. **Per-Layer Rank**: Use adaptive rank based on layer importance
4. **FP16 Quantization**: Try 16-bit quantization instead of INT8 for better quality

---

## Conclusion

While the compressed model achieves a **4.5x speedup**, it suffers from:
- **3.3x memory increase** (regression)
- **Severe quality degradation** (unusable output)

These issues must be resolved before the compression system can be considered production-ready. The speed improvement demonstrates the theoretical potential, but practical usability requires addressing the memory and quality problems.

---

## Appendix: Raw Data

Full benchmark results: [`benchmark_results_gpt_neo.json`](file:///c:/Users/G/Desktop/Compress%20LLM/semantic-llm-compressor/benchmark_results_gpt_neo.json)
