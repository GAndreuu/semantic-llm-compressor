# scripts/test_compressed_vs_dense.py

import argparse
import torch
import torch.nn as nn
from pathlib import Path

from transformers import AutoModelForCausalLM

from semantic_llm_compressor.runtime.loader import load_compressed_factors_for_weight
from semantic_llm_compressor.runtime.compressed_layers import CompressedLinear


def test_single_layer(
    model_dir: str,
    compressed_dir: str,
    weight_name: str,
    batch_size: int = 8,
):
    """
    Compara nn.Linear original com CompressedLinear, usando input aleatório.
    """

    # 1) Carrega modelo original
    print(f"[INFO] Carregando modelo de {model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    # 2) Localiza a camada pelo nome do weight
    #    Ex: "transformer.h.0.attn.attention.out_proj.weight"
    print(f"[INFO] Procurando camada para weight: {weight_name}")
    module_name = weight_name.rsplit(".weight", 1)[0]

    target_module = None
    for name, module in model.named_modules():
        if name == module_name:
            target_module = module
            break

    if target_module is None:
        raise ValueError(f"Não encontrei módulo com nome '{module_name}'")

    if not isinstance(target_module, nn.Linear):
        print(f"[WARN] Módulo {module_name} não é nn.Linear (é {type(target_module)})")

    W = target_module.weight.detach().clone()
    b = (
        target_module.bias.detach().clone()
        if target_module.bias is not None
        else None
    )

    in_features = W.shape[1]
    out_features = W.shape[0]

    print(f"[INFO] Camada encontrada: {module_name}")
    print(f"       shape weight: {tuple(W.shape)} (out={out_features}, in={in_features})")

    # 3) Carrega fatores comprimidos
    print(f"[INFO] Carregando fatores comprimidos de {compressed_dir}...")
    factors = load_compressed_factors_for_weight(Path(compressed_dir), weight_name)

    # 4) Cria CompressedLinear isolado
    comp = CompressedLinear(
        factors=factors,
        bias=b,
    )
    comp.eval()

    # 5) Gera input aleatório
    x = torch.randn(batch_size, in_features)

    # 6) Forward na camada densa original
    with torch.no_grad():
        y_dense = nn.functional.linear(x, W, b)

    # 7) Forward na camada comprimida
    with torch.no_grad():
        y_comp = comp(x)

    # 8) Métricas
    mse = torch.mean((y_dense - y_comp) ** 2).item()
    mae = torch.mean(torch.abs(y_dense - y_comp)).item()
    max_diff = torch.max(torch.abs(y_dense - y_comp)).item()

    # cosine similarity entre vetores achatados
    y_dense_flat = y_dense.flatten()
    y_comp_flat = y_comp.flatten()
    cos = nn.functional.cosine_similarity(
        y_dense_flat.unsqueeze(0), y_comp_flat.unsqueeze(0), dim=1
    ).item()

    # Relative error
    rel_error = (torch.norm(y_dense - y_comp) / torch.norm(y_dense)).item()

    print("\n" + "=" * 60)
    print("RESULTADOS - CAMADA ISOLADA")
    print("=" * 60)
    print(f"Camada: {module_name}")
    print(f"Shape: {tuple(W.shape)}")
    print(f"\nMétricas:")
    print(f"  MSE:              {mse:.6e}")
    print(f"  MAE:              {mae:.6e}")
    print(f"  Max Diff:         {max_diff:.6e}")
    print(f"  Cosine Similarity: {cos:.6f}")
    print(f"  Relative Error:   {rel_error:.2%}")
    
    # Verdict
    print(f"\nVeredicto:")
    if cos > 0.99:
        print("  ✅ EXCELENTE - Camada praticamente idêntica")
    elif cos > 0.95:
        print("  ✅ BOM - Pequena degradação aceitável")
    elif cos > 0.90:
        print("  ⚠️  MODERADO - Degradação visível mas usável")
    elif cos > 0.80:
        print("  ⚠️  RUIM - Degradação significativa")
    else:
        print("  ❌ CRÍTICO - Camada muito degradada")
    
    print("=" * 60)

    return {
        "layer": module_name,
        "shape": tuple(W.shape),
        "mse": mse,
        "mae": mae,
        "max_diff": max_diff,
        "cosine": cos,
        "rel_error": rel_error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Diretório do modelo original (Ex: models/original/gpt-neo-1.3B)",
    )
    parser.add_argument(
        "--compressed_dir",
        type=str,
        required=True,
        help="Diretório com os shards comprimidos",
    )
    parser.add_argument(
        "--weight_name",
        type=str,
        required=True,
        help="Nome exato do peso (Ex: transformer.h.0.attn.attention.out_proj.weight)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Tamanho do batch para o teste",
    )
    args = parser.parse_args()

    test_single_layer(
        model_dir=args.model_dir,
        compressed_dir=args.compressed_dir,
        weight_name=args.weight_name,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
