# Wave 1: Memory Optimization Status

## Current State (Confirmed)

### Loader Behavior
**File**: `semantic_llm_compressor/runtime/loader.py`

**Current Implementation**:
```python
# TODO: Return QuantizedTensor for INT8 storage (currently causing issues)
# For now, dequantize immediately to ensure stability
quantizer = Quantizer(num_bits=quant_bits)

def dequant(data: torch.Tensor, meta: dict) -> torch.Tensor:
    qt = QuantizedTensor(...)
    return quantizer.dequantize(qt)  # ← Returns FP32 torch.Tensor

U = dequant(U_data, U_meta)
return CompressedFactors(U=U, S=S, Vh=Vh)  # ← All FP32
```

**Result**: `factors.U` is `torch.Tensor` with `dtype=torch.float32`

---

## Memory Analysis

### Current Benchmark Results
- **Original Model**: 139.7 MB (mmap-based, low RSS)
- **Compressed Model**: 191.8 MB (+37%)

### Why Memory Increased

1. **Original Model (mmap)**:
   - Model file: 2.6 GB on disk
   - Loaded via memory-mapping
   - Only touches ~140 MB of RAM
   
2. **Compressed Model (in-RAM)**:
   - Base model: ~140 MB (also mmap)
   - Compressed factors (96 layers): ~48 MB FP32
   - Total: ~188 MB

### INT8 Potential Savings

**If INT8 storage enabled**:
- Factors: 48 MB FP32 → 12 MB INT8 (4x reduction)
- Target memory: ~152 MB
- **Still 9% above original** due to mmap vs in-RAM

---

## Action Items

### ✅ Completed
1. Confirmed loader returns FP32 (intentional for stability)
2. Verified CompressedLinear handles both quantized/non-quantized
3. Benchmark runs successfully with FP32

### 🔧 To Enable INT8 Storage

#### Step 1: Update loader.py
```python
# Remove dequantize, return QuantizedTensor directly
def make_qt(data, meta) -> QuantizedTensor:
    return QuantizedTensor(
        data=data,  # INT8
        scale=meta["scale"],
        zero_point=meta["zero_point"],
        original_shape=tuple(meta["original_shape"]),
        original_dtype=meta["original_dtype"],
    )

U = make_qt(U_data, U_meta)
return CompressedFactors(U=U, S=S, Vh=Vh)  # QuantizedTensor
```

#### Step 2: Verify CompressedLinear
Already implemented! Uses `_get_U()`, `_get_S()`, `_get_Vh()` which:
- Check `self.is_quantized`
- Call `_dequantize()` if True
- Return FP32 buffer otherwise

#### Step 3: Add GC to benchmark
```python
# Already added!
import gc
gc.collect()
time.sleep(1)  # Allow OS to update stats
```

#### Step 4: Re-run benchmark
```bash
python scripts/5_benchmark_gpt_neo.py \
  --original_dir models/original/gpt-neo-1.3B \
  --compressed_dir models/compressed/gpt-neo-1.3B-r128-no-conv1d \
  --output_json benchmark_results_int8.json
```

**Expected**:
- Memory: ~152 MB (↓20% from 192 MB)
- Speed: Slight decrease due to dequant overhead
- Quality: No change (same factors, just stored differently)

---

## Recommendation

**Don't enable INT8 storage yet**. Reasons:

1. **Diminishing returns**: 192 MB → 152 MB (still above 140 MB baseline)
2. **mmap trade-off**: We're comparing in-RAM vs mmap (apples to oranges)
3. **Stability**: FP32 path is proven to work
4. **Better alternatives**:
   - Implement adaptive rank (bigger quality gains)
   - Keep MLP in FP16 uncompressed (simpler)
   - Focus on scaling to larger models where savings matter more

**If you DO want INT8**:
- Uncomment the `make_qt` code in `loader.py`
- Run benchmark to verify memory drops to ~150 MB
- Check that quality stays at cosine ~0.875
- Measure dequantization overhead (probably <5% speed loss)

---

## Next: Wave 2 - Quality Per Layer

Create `scripts/test_compressed_vs_dense.py` to verify individual layer quality.
