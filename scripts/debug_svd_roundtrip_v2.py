# scripts/debug_svd_roundtrip_v2.py

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from semantic_llm_compressor.algorithms import SVDDecomposer, Quantizer

MODEL_ID = "models/original/gpt-neo-1.3B"
RANK = 128
DEVICE = "cpu"

def frobenius_rel_error(W, W_hat):
    diff = W_hat - W
    num = torch.linalg.norm(diff)
    den = torch.linalg.norm(W)
    return (num / (den + 1e-12)).item()

def test_weight(name: str, W: torch.Tensor, svd: SVDDecomposer, quant: Quantizer):
    print(f"\n=== TESTANDO {name} | shape={tuple(W.shape)} ===")
    W = W.to(torch.float32).to(DEVICE)

    # 1) SVD truncado
    # SVDDecomposer returns a tuple (U, S, Vh)
    svd_result = svd.decompose(W) 
    U, S, Vh = svd_result

    # 2a) Reconstrução SEM quantização (pra isolar SVD)
    W_hat_svd = U @ torch.diag(S) @ Vh
    mse_svd = F.mse_loss(W_hat_svd, W).item()
    rel_svd = frobenius_rel_error(W, W_hat_svd)
    print(f"[SVD puro]    MSE={mse_svd:.4e}  rel={rel_svd:.4%}")

    # 2b) Quantização + dequantização (usa a MESMA lógica do compressor)
    # Quantizer.quantize returns a QuantizedTensor object
    U_q_obj = quant.quantize(U)
    S_q_obj = quant.quantize(S)
    Vh_q_obj = quant.quantize(Vh)

    # Quantizer.dequantize takes the QuantizedTensor object
    U_d = quant.dequantize(U_q_obj)
    S_d = quant.dequantize(S_q_obj)
    Vh_d = quant.dequantize(Vh_q_obj)

    W_hat_q = U_d @ torch.diag(S_d) @ Vh_d
    mse_q = F.mse_loss(W_hat_q, W).item()
    rel_q = frobenius_rel_error(W, W_hat_q)
    
    norm_w = torch.linalg.norm(W).item()
    print(f"[SVD+INT8]    MSE={mse_q:.4e}  rel={rel_q:.4%}  norm_W={norm_w:.4e}")
    
    return rel_q

def main():
    print(f"Carregando GPT-Neo 1.3B de {MODEL_ID}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return
        
    model.eval()

    svd = SVDDecomposer(rank=RANK)
    quant = Quantizer(num_bits=8)

    print("\nIniciando testes de roundtrip...")
    
    with torch.no_grad():
        # 1) MLP Layers (Suspected culprits)
        try:
            print("\n--- Testing MLP Layers (Rank 512) ---")
            lin = model.transformer.h[0].mlp.c_fc
            
            # Create a new SVD with higher rank
            svd_high = SVDDecomposer(rank=512)
            test_weight("h0.mlp.c_fc", lin.weight, svd_high, quant)
            
            # proj = model.transformer.h[0].mlp.c_proj
            # test_weight("h0.mlp.c_proj", proj.weight, svd, quant)
        except AttributeError:
            print("Could not find mlp layers")

        # 2) Attention Layers
        # try:
        #     print("\n--- Testing Attention Layers ---")
        #     attn_out = model.transformer.h[0].attn.attention.out_proj
        #     test_weight("h0.attn.out_proj", attn_out.weight, svd, quant)
            
        #     # if hasattr(model.transformer.h[0].attn.attention, "k_proj"):
        #     #     k_proj = model.transformer.h[0].attn.attention.k_proj
        #     #     test_weight("h0.attn.k_proj", k_proj.weight, svd, quant)
                
        # except AttributeError as e:
        #     print(f"Could not find attention layer: {e}")

if __name__ == "__main__":
    main()
