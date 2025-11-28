import torch
import torch.nn as nn
from semantic_llm_compressor.algorithms import SVDDecomposer, Quantizer, FactorizationPolicy
from semantic_llm_compressor.config import CompressionConfig
from semantic_llm_compressor.runtime import CompressedTensor, CompressedLinear

def test_svd_quantization_reconstruction():
    print("--- Test 1: SVD + Quantization Reconstruction ---")
    torch.manual_seed(42)
    
    # Create a random 2D tensor (simulating a weight matrix)
    out_dim, in_dim = 128, 128
    W = torch.randn(out_dim, in_dim)
    
    # Config
    rank = 32
    svd = SVDDecomposer(rank=rank)
    quantizer = Quantizer(num_bits=8)
    
    # Decompose
    U, S, Vh = svd.decompose(W)
    print(f"Original shape: {W.shape}")
    print(f"Decomposed shapes: U={U.shape}, S={S.shape}, Vh={Vh.shape}")
    
    # Quantize
    qU = quantizer.quantize(U)
    qS = quantizer.quantize(S)
    qVh = quantizer.quantize(Vh)
    
    # Dequantize
    U_hat = quantizer.dequantize(qU)
    S_hat = quantizer.dequantize(qS)
    Vh_hat = quantizer.dequantize(qVh)
    
    # Reconstruct W_hat = U @ diag(S) @ Vh
    W_hat = U_hat @ (torch.diag(S_hat) @ Vh_hat)
    
    # Measure Error
    mse = torch.nn.functional.mse_loss(W, W_hat).item()
    rel_error = torch.norm(W - W_hat) / torch.norm(W)
    
    print(f"MSE: {mse:.6f}")
    print(f"Relative Error: {rel_error:.6f}")
    
    assert U.shape == (out_dim, rank)
    assert S.shape == (rank,)
    assert Vh.shape == (rank, in_dim)
    print("✅ SVD shapes correct")
    print("✅ Reconstruction successful (error expected to be non-zero but low)")
    print()

def test_factorization_policy():
    print("--- Test 2: Factorization Policy ---")
    config = CompressionConfig(min_dim_for_compression=32, compress_only_2d=True)
    policy = FactorizationPolicy(config)
    
    t1 = torch.randn(64, 64) # Should compress
    t2 = torch.randn(16, 64) # Should NOT compress (dim < 32)
    t3 = torch.randn(64)     # Should NOT compress (1D)
    t4 = torch.randn(10, 10, 10) # Should NOT compress (3D)
    
    assert policy.should_compress("t1", t1) == True
    assert policy.should_compress("t2", t2) == False
    assert policy.should_compress("t3", t3) == False
    assert policy.should_compress("t4", t4) == False
    
    print("✅ Policy logic verified")
    print()

def test_runtime_components():
    print("--- Test 3: Runtime Components (CompressedLinear) ---")
    torch.manual_seed(42)
    
    out_dim, in_dim = 64, 64
    rank = 16
    
    # Original Linear Layer
    linear = nn.Linear(in_dim, out_dim)
    W = linear.weight.data
    b = linear.bias.data
    
    # Compress manually
    svd = SVDDecomposer(rank=rank)
    quantizer = Quantizer(num_bits=8)
    
    U, S, Vh = svd.decompose(W)
    qU = quantizer.quantize(U)
    qS = quantizer.quantize(S)
    qVh = quantizer.quantize(Vh)
    
    # Create CompressedTensor
    comp_tensor = CompressedTensor(qU, qS, qVh, quantizer)
    
    # Test Materialization
    U_m, S_m, Vh_m = comp_tensor.materialize_factors(dtype=torch.float32)
    assert U_m.shape == U.shape
    print("✅ Materialization shapes correct")
    
    # Create CompressedLinear
    comp_linear = CompressedLinear(comp_tensor, bias=b, dtype=torch.float32)
    
    # Test Forward Pass
    x = torch.randn(10, in_dim) # Batch of 10
    
    y_orig = linear(x)
    y_comp = comp_linear(x)
    
    mse = torch.nn.functional.mse_loss(y_orig, y_comp).item()
    print(f"Forward Pass MSE: {mse:.6f}")
    
    assert y_comp.shape == (10, out_dim)
    print("✅ Forward pass shape correct")
    print()

if __name__ == "__main__":
    test_svd_quantization_reconstruction()
    test_factorization_policy()
    test_runtime_components()
