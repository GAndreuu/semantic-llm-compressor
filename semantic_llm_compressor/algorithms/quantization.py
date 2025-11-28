# semantic_llm_compressor/algorithms/quantization.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class QuantizedTensor:
    """
    Representa um tensor quantizado (por ex. INT8) + metadata para dequantizar.
    """
    data: torch.Tensor        # tipicamente int8
    scale: float
    zero_point: int
    original_shape: Tuple[int, ...]
    original_dtype: str       # e.g. "torch.float16", "torch.float32"


@dataclass
class Quantizer:
    """
    Implementa quantização escalar simétrica signed INT8 (por padrão).

    Estratégia:
    - Usa faixa simétrica em torno de zero.
    - Para num_bits=8, usa int8 em [-128, 127].
    - zero_point = 0.
    """

    num_bits: int = 8

    def _get_qmax(self) -> int:
        """
        Retorna o valor máximo representável (lado positivo) para o número de bits.
        Para signed:
            qmax = 2^(bits-1) - 1
        Ex: 8 bits -> 127
        """
        if self.num_bits < 2:
            raise ValueError("Quantizer.num_bits deve ser >= 2")
        return (1 << (self.num_bits - 1)) - 1

    def quantize(self, tensor: torch.Tensor) -> QuantizedTensor:
        """
        Converte tensor float -> QuantizedTensor (data INT8 + metadata).

        Quantização simétrica por tensor:
        - encontra max_abs = max(|min|, |max|)
        - scale = max_abs / qmax
        - q = round(tensor / scale), clamp em [-qmax-1, qmax]
        """
        if not tensor.is_floating_point():
            raise TypeError(f"Quantizer.quantize espera tensor float, recebeu {tensor.dtype}")

        original_shape = tuple(tensor.shape)
        original_dtype = str(tensor.dtype)

        # Trabalha em float32 para estabilidade numérica
        x = tensor.to(torch.float32)

        max_val = x.max()
        min_val = x.min()
        max_abs = torch.max(max_val.abs(), min_val.abs())

        # Caso degenerado: tensor todo zero
        if max_abs == 0:
            q_data = torch.zeros_like(x, dtype=torch.int8)
            scale = 1.0  # qualquer valor não-zero serve; não será usado na prática
            zero_point = 0
            return QuantizedTensor(
                data=q_data,
                scale=float(scale),
                zero_point=int(zero_point),
                original_shape=original_shape,
                original_dtype=original_dtype,
            )

        qmax = self._get_qmax()
        scale = max_abs / qmax
        zero_point = 0

        # tensor / scale -> quantização
        q = torch.round(x / scale)

        # Faixa de int8: [-128, 127]
        q_clamped = torch.clamp(q, -qmax - 1, qmax).to(torch.int8)

        return QuantizedTensor(
            data=q_clamped,
            scale=float(scale),
            zero_point=int(zero_point),
            original_shape=original_shape,
            original_dtype=original_dtype,
        )

    def dequantize(self, q: QuantizedTensor) -> torch.Tensor:
        """
        Converte QuantizedTensor -> tensor float aproximado.

        x_hat = (q.data.float() - zero_point) * scale
        """
        if q.data.dtype != torch.int8:
            raise TypeError(f"QuantizedTensor.data esperado int8, recebido {q.data.dtype}")

        x = (q.data.to(torch.float32) - q.zero_point) * q.scale
        x = x.view(q.original_shape)

        # Interpretar original_dtype de volta para torch.dtype
        try:
            # q.original_dtype é algo tipo 'torch.float16'
            dtype = getattr(torch, q.original_dtype.split(".")[-1])
        except AttributeError:
            # Fallback: manter float32
            dtype = torch.float32

        return x.to(dtype)
