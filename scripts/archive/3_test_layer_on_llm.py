# scripts/3_test_layer_on_llm.py

from pathlib import Path
import json

import torch
from safetensors.torch import load_file
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from semantic_llm_compressor.algorithms import Quantizer
from semantic_llm_compressor.algorithms.quantization import QuantizedTensor


ORIG_DIR = Path("models/original/tiny-gpt2")
COMP_DIR = Path("models/compressed/tiny-gpt2-r16")
MODEL_NAME = "sshleifer/tiny-gpt2"

TARGET_WEIGHT_NAME = "transformer.h.0.mlp.c_fc.weight"


def load_reconstructed_weight() -> torch.Tensor:
    # praticamente igual ao script anterior
    index_path = COMP_DIR / "model.safetensors.index.json"
    index_data = json.loads(index_path.read_text())
    shard_rel = index_data["weight_map"][TARGET_WEIGHT_NAME]
    shard_path = COMP_DIR / shard_rel

    shard_tensors = load_file(str(shard_path))
    
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    
    compression_meta = json.loads(metadata.get("compression_meta", "{}"))

    U_name = f"{TARGET_WEIGHT_NAME}.U.quant"
    S_name = f"{TARGET_WEIGHT_NAME}.S.quant"
    Vh_name = f"{TARGET_WEIGHT_NAME}.Vh.quant"

    qU_meta = compression_meta[f"{TARGET_WEIGHT_NAME}.U"]
    qS_meta = compression_meta[f"{TARGET_WEIGHT_NAME}.S"]
    qVh_meta = compression_meta[f"{TARGET_WEIGHT_NAME}.Vh"]

    quantizer = Quantizer(num_bits=8)

    def deq(name, meta):
        data = shard_tensors[name]
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

    # W_hat ≈ U diag(S) Vh
    U_scaled = U * S.unsqueeze(0)
    W_hat = U_scaled @ Vh
    return W_hat


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Carregando modelo original e tokenizer...")
    model_orig = AutoModelForCausalLM.from_pretrained(ORIG_DIR)
    tokenizer = AutoTokenizer.from_pretrained(ORIG_DIR)

    model_orig.to(device)
    model_orig.eval()

    # Copiamos o modelo para aplicar peso aproximado
    model_approx = AutoModelForCausalLM.from_pretrained(ORIG_DIR)
    model_approx.to(device)
    model_approx.eval()

    print(f"Reconstruindo peso aproximado de {TARGET_WEIGHT_NAME}...")
    W_hat = load_reconstructed_weight().to(model_approx.device)

    # Substitui o peso de UMA camada no modelo aproximado
    # tiny-gpt2 usa estrutura GPT2-like: model.transformer.h[0].mlp.c_fc.weight
    # Note: GPT-2 uses Conv1D, so weight is [in_features, out_features] (transposed compared to Linear)
    # But safetensors stores it as is.
    # Let's check shape match.
    target_module = model_approx.transformer.h[0].mlp.c_fc
    with torch.no_grad():
        print("Peso original shape:", tuple(target_module.weight.shape))
        print("Peso reconstruído shape:", tuple(W_hat.shape))
        
        # If shapes mismatch (transposed), transpose W_hat
        if target_module.weight.shape != W_hat.shape and target_module.weight.shape == W_hat.T.shape:
             print("Transpondo W_hat para casar com Conv1D...")
             W_hat = W_hat.T

        target_module.weight.copy_(W_hat.to(target_module.weight.dtype))
    print("Peso aproximado copiado para modelo_approx.")

    # Prompt de teste
    prompt = "The meaning of life is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out_orig = model_orig(**inputs)
        out_approx = model_approx(**inputs)

    logits_orig = out_orig.logits[:, -1, :]   # último token
    logits_approx = out_approx.logits[:, -1, :]

    # MSE entre logits
    mse_logits = torch.mean((logits_orig - logits_approx) ** 2).item()
    max_diff = torch.max(torch.abs(logits_orig - logits_approx)).item()

    print(f"MSE entre logits finais: {mse_logits:.6e}")
    print(f"Máx |Δlogit|: {max_diff:.6e}")
    
    with open("metrics_output.json", "w") as f:
        json.dump({"mse_logits": mse_logits, "max_diff": max_diff}, f)

    # Gerar algumas continuações pra ver qualitativamente
    print("\n=== Geração com modelo original ===")
    gen_orig = model_orig.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
    )
    print(tokenizer.decode(gen_orig[0], skip_special_tokens=True))

    print("\n=== Geração com modelo aproximado (uma camada comprimida) ===")
    gen_approx = model_approx.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
    )
    print(tokenizer.decode(gen_approx[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
