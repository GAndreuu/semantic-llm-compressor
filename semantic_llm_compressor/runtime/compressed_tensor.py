# semantic_llm_compressor/runtime/compressed_tensor.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from ..algorithms import QuantizedTensor, Quantizer


@dataclass
class CompressedTensor:
    """
    Representa um tensor comprimido em forma de fatores (U, S, Vh) quantizados.

    No runtime, podemos:
    - dequantizar uma vez (FP16) e manter em memória (cache),
    - ou dequantizar sob demanda se quisermos economizar RAM.
    """

    U_q: QuantizedTensor
    S_q: QuantizedTensor
    Vh_q: QuantizedTensor
    quantizer: Quantizer

    # Cache opcional para fatores dequantizados
    U_cache: Optional[torch.Tensor] = None
    S_cache: Optional[torch.Tensor] = None
    Vh_cache: Optional[torch.Tensor] = None

    def materialize_factors(self, dtype: torch.dtype = torch.float16) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Garante que U, S, Vh estejam dequantizados e em dtype desejado.
        Usa cache se disponível.
        """
        if self.U_cache is not None and self.S_cache is not None and self.Vh_cache is not None:
            # Verifica se dtype bate (opcional, assumindo consistência)
            if self.U_cache.dtype == dtype:
                return self.U_cache, self.S_cache, self.Vh_cache

        # Dequantiza
        U = self.quantizer.dequantize(self.U_q).to(dtype)
        S = self.quantizer.dequantize(self.S_q).to(dtype)
        Vh = self.quantizer.dequantize(self.Vh_q).to(dtype)

        # Salva no cache
        self.U_cache = U
        self.S_cache = S
        self.Vh_cache = Vh

        return U, S, Vh

    def clear_cache(self) -> None:
        """Libera memória dos tensores dequantizados."""
        self.U_cache = None
        self.S_cache = None
        self.Vh_cache = None
