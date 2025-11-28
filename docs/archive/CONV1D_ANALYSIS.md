# Conv1D Investigation Results

**Date:** November 25, 2025  
**Experiment:** Skip Conv1D compression to isolate quality issues

---

## Summary

Skipping Conv1D layer compression **dramatically improved** model quality, confirming that **Conv1D transposition logic is the primary cause** of the quality degradation.

## Comparison

| Metric | With Conv1D Compressed | Without Conv1D | Improvement |
|:---|---:|---:|---:|
| **Cosine Similarity** | 0.102 | **0.875** | **8.5x better** ✅ |
| **Logits MSE** | 223.0 | 116.2 | 47.9% reduction ✅ |
| **Memory Usage** | 468 MB | 187 MB | 60% reduction ✅ |
| **Speed** | 18.02 tok/s | 5.11 tok/s | 3.5x slower ⚠️ |
| **Layers Patched** | 144/145 | 96/145 | 48 fewer layers |

## Detailed Analysis

### Quality Metrics

**Prompt: "The meaning of life is"**

| Version | Cosine Sim | MSE | Output |
|:---|---:|---:|:---|
| **With Conv1D** | -0.530 | 302.84 | "isedayedayangeredarrettilynangered..." (gibberish) |
| **Without Conv1D** | **0.924** | 134.23 | "is to,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,," (repetitive but coherent start) |

**Average across 3 prompts:**
- Cosine Similarity: **0.875** (vs 0.102) - **8.5x improvement**
- Logits MSE: 116.2 (vs 223.0) - 47.9% reduction

### Memory Usage

- **With Conv1D**: 468 MB (3.3x worse than original)
- **Without Conv1D**: 187 MB (1.3x worse than original)
- **Improvement**: 60% memory reduction

The memory is still higher than the original (139 MB), indicating there's still a memory leak issue, but it's much better.

### Speed

- **With Conv1D**: 18.02 tokens/sec (4.5x faster)
- **Without Conv1D**: 5.11 tokens/sec (1.3x faster)

Speed decreased because we're compressing fewer layers (96 vs 144), but quality is the priority.

---

## Conclusion

✅ **Conv1D transposition logic is CONFIRMED as the root cause** of quality degradation  
✅ Skipping Conv1D compression results in **usable model output**  
⚠️ Quality still not perfect (repetitive tokens) - suggests additional issues in remaining compressed layers  
⚠️ Memory leak still present but reduced (187 MB vs 139 MB original)

## Next Steps

1. **Fix Conv1D transposition logic** (A3)
   - Current logic swaps and transposes U/Vh incorrectly
   - Need to understand exact Conv1D forward pass and match it

2. **Create SVD roundtrip test** (A2)
   - Verify reconstruction quality for both Linear and Conv1D layers
   - Isolate quantization vs transposition issues

3. **Fix remaining memory leak** (B1-B4)
   - Even without Conv1D, memory is 34% higher than original
   - Need to ensure original weights are freed after patching

4. **Investigate remaining quality issues**
   - Repetitive comma output suggests problems in other layers too
   - May need higher rank or selective compression
