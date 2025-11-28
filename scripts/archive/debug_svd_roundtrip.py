# scripts/debug_svd_roundtrip.py

"""
Test SVD roundtrip quality on individual layers.
This isolates compression/decompression issues before involving the patcher.
"""

import torch
from transformers import AutoModelForCausalLM
from semantic_llm_compressor.algorithms import SVDDecomposer, Quantizer

def test_weight_roundtrip(name, W, rank=128, verbose=True):
    """
    Test compress -> decompress roundtrip on a single weight tensor.
    
    Returns:
        dict with mse, rel_error, and shape info
    """
    original_shape = W.shape
    
    # 1) SVD decomposition
    svd = SVDDecomposer(rank=rank)
    U, S, Vh = svd.decompose(W.detach().cpu())
    
    # 2) Quantize each factor
    quantizer = Quantizer(num_bits=8)
    
    U_q = quantizer.quantize(U)
    S_q = quantizer.quantize(S)
    Vh_q = quantizer.quantize(Vh)
    
    # 3) Dequantize
    U_d = quantizer.dequantize(U_q)
    S_d = quantizer.dequantize(S_q)
    Vh_d = quantizer.dequantize(Vh_q)
    
    # 4) Reconstruct
    W_hat = U_d @ torch.diag(S_d) @ Vh_d
    
    # 5) Metrics
    diff = W_hat - W
    mse = torch.mean(diff**2).item()
    rel_error = (torch.linalg.norm(diff) / torch.linalg.norm(W)).item()
    
    result = {
        "name": name,
        "shape": original_shape,
        "rank": rank,
        "mse": mse,
        "rel_error": rel_error,
        "rel_error_percent": rel_error * 100
    }
    
    if verbose:
        print(f"\n{name}")
        print(f"  Shape: {original_shape}")
        print(f"  MSE: {mse:.6e}")
        print(f"  Rel Error: {rel_error:.4%}")
        
        # Quality assessment
        if rel_error < 0.005:
            quality = "✅ EXCELLENT"
        elif rel_error < 0.02:
            quality = "✅ GOOD"
        elif rel_error < 0.05:
            quality = "⚠️  ACCEPTABLE"
        else:
            quality = "❌ POOR"
        print(f"  Quality: {quality}")
    
    return result

def main():
    print("=" * 80)
    print("SVD Roundtrip Test: GPT-Neo 1.3B")
    print("=" * 80)
    print("\nLoading model...")
    
    model = AutoModelForCausalLM.from_pretrained("models/original/gpt-neo-1.3B")
    
    results = []
    
    # Test different layer types
    print("\n" + "=" * 80)
    print("Testing MLP Layers (should be standard Linear)")
    print("=" * 80)
    
    with torch.no_grad():
        # MLP layers - these should work well
        results.append(test_weight_roundtrip(
            "h[0].mlp.c_fc",
            model.transformer.h[0].mlp.c_fc.weight
        ))
        
        results.append(test_weight_roundtrip(
            "h[0].mlp.c_proj",
            model.transformer.h[0].mlp.c_proj.weight
        ))
        
        results.append(test_weight_roundtrip(
            "h[5].mlp.c_fc",
            model.transformer.h[5].mlp.c_fc.weight
        ))
    
    print("\n" + "=" * 80)
    print("Testing Attention Conv1D Layers (problematic)")
    print("=" * 80)
    
    with torch.no_grad():
        # Attention Conv1D layers - these are the problematic ones
        # GPT-Neo uses: transformer.h[i].attn.attention.{k_proj, v_proj, q_proj, out_proj}
        # But weights are stored as c_attn (combined QKV) and c_proj (output)
        
        # Check if model has the expected structure
        try:
            results.append(test_weight_roundtrip(
                "h[0].attn.attention.k_proj",
                model.transformer.h[0].attn.attention.k_proj.weight
            ))
        except AttributeError:
            # Try alternative structure
            print("  Note: Using alternative layer structure")
            
        results.append(test_weight_roundtrip(
            "h[0].attn.attention.out_proj",
            model.transformer.h[0].attn.attention.out_proj.weight
        ))
        
        results.append(test_weight_roundtrip(
            "h[5].attn.attention.out_proj",
            model.transformer.h[5].attn.attention.out_proj.weight
        ))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    mlp_results = [r for r in results if "mlp" in r["name"]]
    attn_results = [r for r in results if "attn" in r["name"]]
    
    if mlp_results:
        avg_mlp_error = sum(r["rel_error"] for r in mlp_results) / len(mlp_results)
        print(f"\nMLP Layers (Linear):")
        print(f"  Avg Rel Error: {avg_mlp_error:.4%}")
        print(f"  Status: {'✅ GOOD' if avg_mlp_error < 0.02 else '⚠️ NEEDS ATTENTION'}")
    
    if attn_results:
        avg_attn_error = sum(r["rel_error"] for r in attn_results) / len(attn_results)
        print(f"\nAttention Layers (Conv1D):")
        print(f"  Avg Rel Error: {avg_attn_error:.4%}")
        print(f"  Status: {'✅ GOOD' if avg_attn_error < 0.02 else '⚠️ NEEDS ATTENTION'}")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    if mlp_results and attn_results:
        if avg_mlp_error < 0.02 and avg_attn_error > 0.05:
            print("\n❌ Conv1D layers have MUCH worse reconstruction than Linear layers")
            print("   → This confirms the transposition/shape issue in Conv1D handling")
            print("   → Fix: Remove U/Vh swapping in patcher, treat Conv1D like Linear")
        elif avg_mlp_error < 0.02 and avg_attn_error < 0.02:
            print("\n✅ Both layer types reconstruct well!")
            print("   → SVD + quantization are working correctly")
            print("   → Problem must be in the patcher or runtime forward pass")
        elif avg_mlp_error > 0.05:
            print("\n❌ Even Linear layers have poor reconstruction")
            print("   → Problem is in SVD decomposition or quantization")
            print("   → Check rank (may be too low) or quantization precision")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
