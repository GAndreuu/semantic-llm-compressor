# semantic_llm_compressor/runtime/patcher.py

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from semantic_llm_compressor.runtime.compressed_layers import (
    CompressedLinear,
    CompressedFactors,
)
from semantic_llm_compressor.runtime.loader import (
    load_compressed_factors_for_weight,
)


def _replace_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    """
    Seta root.<module_name> = new_module, navegando pelos submódulos.

    Ex:
        module_name="transformer.h.0.mlp.c_fc"
        => root.transformer.h[0].mlp.c_fc = new_module
    """
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        # lida com listas/sequentials (h.0, etc.)
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def patch_model_with_compressed_linears(
    model: nn.Module,
    compressed_dir: Path,
    quant_bits: int = 8,
    verbose: bool = True,
) -> Tuple[int, int]:
    """
    Percorre o modelo e tenta substituir cada nn.Linear por CompressedLinear,
    usando fatores U/S/Vh do diretório comprimido.

    Estratégia:
      - Para cada nn.Linear com nome 'X',
      - tenta carregar fatores para 'X.weight' a partir de compressed_dir.
      - Se conseguir: substitui pela versão comprimida.
      - Se não: deixa a camada original intacta.

    Retorna:
      (num_patched, num_linear_total)
    """
    compressed_dir = Path(compressed_dir)

    num_linear = 0
    num_patched = 0

    # Tenta importar Conv1D (comum em GPT-2)
    try:
        from transformers.pytorch_utils import Conv1D
    except ImportError:
        Conv1D = None

    # Usamos list(...) porque vamos modificar a árvore enquanto iteramos
    for name, module in list(model.named_modules()):
        is_linear = isinstance(module, nn.Linear)
        is_conv1d = Conv1D is not None and isinstance(module, Conv1D)

        if is_linear or is_conv1d:
            num_linear += 1
            weight_name = f"{name}.weight"

            try:
                factors: CompressedFactors = load_compressed_factors_for_weight(
                    compressed_dir=compressed_dir,
                    weight_name=weight_name,
                    quant_bits=quant_bits,
                )
            except Exception:
                # Não foi comprimido / não encontrou fatores -> pula
                continue

            if verbose:
                print(f"[PATCH] Substituindo {weight_name} por CompressedLinear")

            # Bias vem do modelo original (não foi comprimido)
            bias = module.bias.data if module.bias is not None else None

            # Se for Conv1D, os pesos são [in, out]. O compressor decompôs W = U S Vh.
            # Forward Conv1D: x @ W + b = x @ (U S Vh) + b.
            # CompressedLinear faz: x @ (U_lin S Vh_lin)^T + b = x @ (Vh_lin^T S U_lin^T) + b.
            # Para equivaler: Vh_lin^T = U  => Vh_lin = U.T
            #                 U_lin^T = Vh => U_lin = Vh.T
            if is_conv1d:
                # Troca U e Vh e transpoe
                new_U = factors.Vh.t()
                new_Vh = factors.U.t()
                factors = CompressedFactors(U=new_U, S=factors.S, Vh=new_Vh)

            comp_linear = CompressedLinear(
                factors=factors,
                bias=bias,
            ).to(module.weight.device)

            _replace_module(model, name, comp_linear)
            num_patched += 1

    if verbose:
        print(
            f"[PATCH] Linear layers: {num_patched}/{num_linear} substituídas "
            f"por CompressedLinear."
        )

    return num_patched, num_linear

# Mantendo compatibilidade com código antigo se necessário, ou podemos remover CompressedModelLoader
# Se CompressedModelLoader não for mais usado, podemos removê-lo ou adaptá-lo.
# Por enquanto, vou sobrescrever o arquivo com a nova implementação limpa.
