import torch
import torch.nn as nn
from pathlib import Path
from semantic_llm_compressor.runtime import load_compressed_factors_for_weight, CompressedLinear

def test_refined_runtime():
    print("--- Test Refined Runtime ---")
    
    # Path to the compressed model we created earlier
    compressed_dir = Path("models/compressed/tiny-gpt2-r16")
    weight_name = "transformer.h.0.mlp.c_fc.weight"
    
    print(f"Loading factors for {weight_name} from {compressed_dir}...")
    
    try:
        factors = load_compressed_factors_for_weight(compressed_dir, weight_name)
        print("✅ Factors loaded successfully")
        print(f"U shape: {factors.U.shape}")
        print(f"S shape: {factors.S.shape}")
        print(f"Vh shape: {factors.Vh.shape}")
        print(f"Rank: {factors.rank}")
        print(f"In Features: {factors.in_features}")
        print(f"Out Features: {factors.out_features}")
        
        # Create CompressedLinear
        print("Creating CompressedLinear layer...")
        # Note: We don't have the bias handy here easily without loading it separately, 
        # so we'll test without bias or just pass None for now to test the matmul.
        layer = CompressedLinear(factors, bias=None)
        
        # Test Forward Pass
        batch_size = 2
        seq_len = 5
        x = torch.randn(batch_size, seq_len, factors.in_features)
        
        print(f"Input shape: {x.shape}")
        y = layer(x)
        print(f"Output shape: {y.shape}")
        
        expected_shape = (batch_size, seq_len, factors.out_features)
        assert y.shape == expected_shape
        print("✅ Forward pass shape correct")
        
        # Test Materialize
        W_hat = layer.materialize_weight()
        print(f"Materialized Weight shape: {W_hat.shape}")
        assert W_hat.shape == (factors.out_features, factors.in_features)
        print("✅ Materialize weight shape correct")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    test_refined_runtime()
