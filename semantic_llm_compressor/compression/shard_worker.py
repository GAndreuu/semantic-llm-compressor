# semantic_llm_compressor/compression/shard_worker.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import json

import torch

from ..algorithms import SVDDecomposer, Quantizer, FactorizationPolicy
from ..io import SafeTensorReader, SafeTensorWriter
from ..logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class ShardWorker:
    """
    Responsável por comprimir um ÚNICO shard safetensors.

    Fluxo:
    - Abrir shard original.
    - Para cada tensor:
        - decidir se comprime ou copia.
        - se comprime: SVD → quantizar U/S/Vh → salvar com sufixos.
    - Escrever novo shard comprimido + metadata de compressão.
    """

    shard_path_in: Path
    shard_path_out: Path
    svd: SVDDecomposer
    quantizer: Quantizer
    policy: FactorizationPolicy

    def run(self) -> None:
        logger.info(f"Comprimindo shard: {self.shard_path_in.name}")

        reader = SafeTensorReader.open(self.shard_path_in)
        writer = SafeTensorWriter(self.shard_path_out)

        # Metadados de compressão por tensor/fator
        # Estrutura:
        # {
        #   "layer.weight.U": {...},
        #   "layer.weight.S": {...},
        #   "layer.weight.Vh": {...},
        #   ...
        # }
        compression_meta: Dict[str, Dict[str, Any]] = {}

        for name, tensor in reader.tensors.items():
            if self.policy.should_compress(name, tensor):
                logger.debug(f"  [COMPRESS] {name} shape={tuple(tensor.shape)}")

                # Adaptive rank selection based on layer type
                rank = self.policy.choose_rank(name, tensor)
                svd_local = SVDDecomposer(rank=rank)
                U, S, Vh = svd_local.decompose(tensor)
                
                logger.debug(f"    → Using adaptive rank={rank} for {name}")

                qU = self.quantizer.quantize(U)
                qS = self.quantizer.quantize(S)
                qVh = self.quantizer.quantize(Vh)

                # Adiciona tensores quantizados ao writer
                writer.add_tensor(f"{name}.U.quant", qU.data)
                writer.add_tensor(f"{name}.S.quant", qS.data)
                writer.add_tensor(f"{name}.Vh.quant", qVh.data)

                # Registra metadados necessários para dequantizar depois
                compression_meta[f"{name}.U"] = {
                    "scale": qU.scale,
                    "zero_point": qU.zero_point,
                    "original_shape": qU.original_shape,
                    "original_dtype": qU.original_dtype,
                }
                compression_meta[f"{name}.S"] = {
                    "scale": qS.scale,
                    "zero_point": qS.zero_point,
                    "original_shape": qS.original_shape,
                    "original_dtype": qS.original_dtype,
                }
                compression_meta[f"{name}.Vh"] = {
                    "scale": qVh.scale,
                    "zero_point": qVh.zero_point,
                    "original_shape": qVh.original_shape,
                    "original_dtype": qVh.original_dtype,
                }

            else:
                # Não comprime -> copia tensor original
                logger.debug(f"  [COPY] {name} shape={tuple(tensor.shape)}")
                writer.add_tensor(name, tensor)

        # Serializa metadados de compressão como uma única entrada JSON
        if compression_meta:
            writer.set_metadata("compression_meta", json.dumps(compression_meta))

        # Salva shard comprimido
        writer.save()
        logger.info(f"Shard comprimido salvo em: {self.shard_path_out}")
