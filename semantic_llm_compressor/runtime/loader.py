# semantic_llm_compressor/runtime/loader.py

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import json

import torch
from safetensors.torch import load_file
from safetensors import safe_open

from semantic_llm_compressor.algorithms import Quantizer, QuantizedTensor
from semantic_llm_compressor.runtime.compressed_layers import CompressedFactors


def load_compressed_factors_for_weight(
    compressed_dir: Path,
    weight_name: str,
    quant_bits: int = 8,
) -> CompressedFactors:
    """
    Carrega U, S, Vh dequantizados para um dado weight_name
    a partir de um diretório comprimido.

    Assumimos:
      - Existe model.safetensors.index.json em compressed_dir.
      - Shard correspondente contém:
          <weight_name>.U.quant
          <weight_name>.S.quant
          <weight_name>.Vh.quant
      - Metadata 'compression_meta' com params de quantização.
    """
    compressed_dir = Path(compressed_dir)

    # 1) Carrega index.json
    index_path = compressed_dir / "model.safetensors.index.json"
    index_data = json.loads(index_path.read_text())

    weight_map = index_data["weight_map"]
    
    # O weight_map aponta para o shard do peso ORIGINAL.
    # Mas como o compressor mantém a estrutura de shards, deve estar no mesmo shard.
    # Porém, o compressor pode ter salvo os fatores com nomes diferentes no mesmo shard.
    # Vamos assumir que se "weight_name" está no map, é lá que estão os fatores.
    # Ou, se o compressor substituiu a entrada no map para apontar para U.quant, precisamos achar.
    
    # No nosso compressor atual, nós salvamos:
    # writer.add_tensor(f"{name}.U.quant", ...)
    # E o index.json tem "weight_map" original + compression_config.
    # O weight_map original aponta "layer.weight" -> "shard-001".
    # O shard-001 comprimido tem "layer.weight.U.quant".
    
    if weight_name not in weight_map:
        # Tenta achar via sufixo se o index foi atualizado (nosso compressor atual mantém o nome original no map?)
        # O compressor atual:
        # index_dict["weight_map"] = index.weight_map (copia do original)
        # Então weight_name deve estar lá.
        raise KeyError(f"Weight {weight_name!r} não encontrado no weight_map.")

    shard_rel = weight_map[weight_name]
    shard_path = compressed_dir / shard_rel

    # 2) Carrega shard e metadata
    # load_file carrega todos os tensores. Para eficiência em modelos grandes, 
    # deveríamos usar safe_open e get_tensor apenas para o que precisamos.
    # Mas load_file é mais simples para implementação inicial.
    
    # Precisamos do metadata primeiro
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        # Para carregar tensores específicos sem carregar tudo:
        U_q_name = f"{weight_name}.U.quant"
        S_q_name = f"{weight_name}.S.quant"
        Vh_q_name = f"{weight_name}.Vh.quant"
        
        try:
            U_data = f.get_tensor(U_q_name)
            S_data = f.get_tensor(S_q_name)
            Vh_data = f.get_tensor(Vh_q_name)
        except Exception:
             raise KeyError(f"Fatores comprimidos não encontrados no shard {shard_rel} para {weight_name}")

    compression_meta = json.loads(metadata.get("compression_meta", "{}"))

    # 4) Metadados dos fatores
    try:
        U_meta = compression_meta[f"{weight_name}.U"]
        S_meta = compression_meta[f"{weight_name}.S"]
        Vh_meta = compression_meta[f"{weight_name}.Vh"]
    except KeyError as e:
        raise KeyError(f"Metadados de compressão faltando para {weight_name}: {e}")

    # TODO: Return QuantizedTensor for INT8 storage (currently causing issues)
    # For now, dequantize immediately to ensure stability
    quantizer = Quantizer(num_bits=quant_bits)

    def dequant(data: torch.Tensor, meta: dict) -> torch.Tensor:
        qt = QuantizedTensor(
            data=data,
            scale=meta["scale"],
            zero_point=meta["zero_point"],
            original_shape=tuple(meta["original_shape"]),
            original_dtype=meta["original_dtype"],
        )
        return quantizer.dequantize(qt)

    U = dequant(U_data, U_meta)
    S = dequant(S_data, S_meta)
    Vh = dequant(Vh_data, Vh_meta)

    return CompressedFactors(U=U, S=S, Vh=Vh)
