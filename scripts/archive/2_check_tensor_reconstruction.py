# scripts/2_check_tensor_reconstruction.py

from pathlib import Path
import json

import torch
from safetensors.torch import load_file
from safetensors import safe_open

from semantic_llm_compressor.algorithms import Quantizer


ORIG_DIR = Path("models/original/tiny-gpt2")
COMP_DIR = Path("models/compressed/tiny-gpt2-r16")

# escolha um nome de weight que você sabe que é grande 2D
# você pode listar depois olhando o shard, mas esse é típico em GPT2-like:
TARGET_WEIGHT_NAME = "transformer.h.0.mlp.c_fc.weight"


def load_original_weight() -> torch.Tensor:
    # descobrir shard original
    index_path = ORIG_DIR / "model.safetensors.index.json"
    index_data = json.loads(index_path.read_text())
    shard_rel = index_data["weight_map"][TARGET_WEIGHT_NAME]
    shard_path = ORIG_DIR / shard_rel

    shard_tensors = load_file(str(shard_path))
    W = shard_tensors[TARGET_WEIGHT_NAME]
    return W


def load_compressed_factors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index_path = COMP_DIR / "model.safetensors.index.json"
    index_data = json.loads(index_path.read_text())
    shard_rel = index_data["weight_map"][TARGET_WEIGHT_NAME]
    shard_path = COMP_DIR / shard_rel

    # Carrega shard comprimido (tem .U.quant, .S.quant, .Vh.quant)
    shard_tensors = load_file(str(shard_path))
    
    # FIX: Use safe_open to get metadata
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        
    compression_meta = json.loads(metadata.get("compression_meta", "{}"))

    # nomes dos tensores quantizados
    U_name = f"{TARGET_WEIGHT_NAME}.U.quant"
    S_name = f"{TARGET_WEIGHT_NAME}.S.quant"
    Vh_name = f"{TARGET_WEIGHT_NAME}.Vh.quant"

    qU_meta = compression_meta[f"{TARGET_WEIGHT_NAME}.U"]
    qS_meta = compression_meta[f"{TARGET_WEIGHT_NAME}.S"]
    qVh_meta = compression_meta[f"{TARGET_WEIGHT_NAME}.Vh"]

    quantizer = Quantizer(num_bits=8)

    # reconstrói QuantizedTensor e dequantiza
    from semantic_llm_compressor.algorithms.quantization import QuantizedTensor

    def deq(name, meta):
        data = shard_tensors[name]  # int8 data
        qt = QuantizedTensor(
            data=data,
            scale=meta["scale"],
            zero_point=meta["zero_point"],
            original_shape=tuple(meta["original_shape"]),
            original_dtype=meta["original_dtype"],
        )
        return quantizer.dequantize(qt)

    U = deq(U_name, qU_meta)
    S = deq(S_name, qS_meta)
    Vh = deq(Vh_name, qVh_meta)
    return U, S, Vh


def reconstruct_weight(U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor) -> torch.Tensor:
    # W_hat ≈ U @ diag(S) @ Vh
    # Implementação eficiente: (U * S) @ Vh, usando broadcasting
    # U: [out, r], S: [r], Vh: [r, in]
    U_scaled = U * S.unsqueeze(0)  # [out, r]
    W_hat = U_scaled @ Vh  # [out, in]
    return W_hat


def main():
    print("Carregando peso original...")
    W = load_original_weight()
    print("Original shape:", tuple(W.shape))

    print("Carregando fatores comprimidos...")
    try:
        U, S, Vh = load_compressed_factors()
        print("U shape:", tuple(U.shape), "S shape:", tuple(S.shape), "Vh shape:", tuple(Vh.shape))

        W_hat = reconstruct_weight(U, S, Vh)

        # converter para float32 para calcular erro com mais precisão
        W_f32 = W.to(torch.float32)
        W_hat_f32 = W_hat.to(torch.float32)

        mse = torch.mean((W_f32 - W_hat_f32) ** 2).item()
        frob = torch.norm(W_f32).item()
        frob_err = torch.norm(W_f32 - W_hat_f32).item()
        rel_err = frob_err / (frob + 1e-8)

        print(f"MSE(W, W_hat)      = {mse:.6e}")
        print(f"||W||_F            = {frob:.6e}")
        print(f"||W - W_hat||_F    = {frob_err:.6e}")
        print(f"Erro relativo Frobenius = {rel_err:.6%}")
    except KeyError as e:
        print(f"Erro: Tensor não encontrado ou não comprimido. Verifique se {TARGET_WEIGHT_NAME} atende aos critérios de compressão.")
        print(f"Detalhe: {e}")


if __name__ == "__main__":
    main()
