# scripts/debug_storage_path.py

from pathlib import Path
from semantic_llm_compressor.runtime.loader import load_compressed_factors_for_weight

base_dir = Path("models/compressed/gpt-neo-1.3B-r128-no-conv1d")
weight_name = "transformer.h.0.attn.attention.out_proj.weight"

print(f"Loading factors for: {weight_name}")
print(f"From directory: {base_dir}")
print()

factors = load_compressed_factors_for_weight(base_dir, weight_name)

print("=" * 60)
print("FACTOR STORAGE ANALYSIS")
print("=" * 60)

print(f"\nU:")
print(f"  Type: {type(factors.U)}")
print(f"  Dtype: {getattr(factors.U, 'dtype', 'N/A')}")
if hasattr(factors.U, 'data'):
    print(f"  ✓ Has .data attribute (QuantizedTensor)")
    print(f"  Data dtype: {factors.U.data.dtype}")
    print(f"  Data shape: {factors.U.data.shape}")
    print(f"  Scale (type): {type(factors.U.scale)} = {factors.U.scale}")
    print(f"  Zero point: {factors.U.zero_point}")
else:
    print(f"  ✗ No .data attribute (regular Tensor)")
    print(f"  Shape: {factors.U.shape}")

print(f"\nS:")
print(f"  Type: {type(factors.S)}")
print(f"  Dtype: {getattr(factors.S, 'dtype', 'N/A')}")
if hasattr(factors.S, 'data'):
    print(f"  ✓ QuantizedTensor: Data dtype={factors.S.data.dtype}, shape={factors.S.data.shape}")
else:
    print(f"  ✗ Regular Tensor: Shape={factors.S.shape}")

print(f"\nVh:")
print(f"  Type: {type(factors.Vh)}")
print(f"  Dtype: {getattr(factors.Vh, 'dtype', 'N/A')}")
if hasattr(factors.Vh, 'data'):
    print(f"  ✓ QuantizedTensor: Data dtype={factors.Vh.data.dtype}, shape={factors.Vh.data.shape}")
else:
    print(f"  ✗ Regular Tensor: Shape={factors.Vh.shape}")

print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)

from semantic_llm_compressor.algorithms import QuantizedTensor
import torch

if isinstance(factors.U, QuantizedTensor):
    if factors.U.data.dtype == torch.int8:
        print("✅ CORRECT: Factors are QuantizedTensor with INT8 data")
    else:
        print(f"⚠️  WARNING: QuantizedTensor but data is {factors.U.data.dtype} (expected INT8)")
elif isinstance(factors.U, torch.Tensor):
    print(f"❌ WRONG: Factors are torch.Tensor (dtype={factors.U.dtype})")
    print("   → Loader is dequantizing! Should return QuantizedTensor.")
else:
    print(f"❌ UNKNOWN: Factors are {type(factors.U)}")
