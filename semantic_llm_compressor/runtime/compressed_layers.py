# semantic_llm_compressor/runtime/compressed_layers.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from semantic_llm_compressor.algorithms import QuantizedTensor


@dataclass
class CompressedFactors:
    """
    Contém os fatores decompostos de uma camada linear, possivelmente quantizados.
    
    Agora suporta QuantizedTensor para economia de memória.
    """
    U: Union[torch.Tensor, QuantizedTensor]
    S: Union[torch.Tensor, QuantizedTensor]
    Vh: Union[torch.Tensor, QuantizedTensor]

    @property
    def rank(self) -> int:
        if isinstance(self.S, QuantizedTensor):
            return self.S.data.shape[0]
        return self.S.shape[0]

    @property
    def in_features(self) -> int:
        if isinstance(self.Vh, QuantizedTensor):
            return self.Vh.data.shape[1]
        return self.Vh.shape[1]

    @property
    def out_features(self) -> int:
        if isinstance(self.U, QuantizedTensor):
            return self.U.data.shape[0]
        return self.U.shape[0]


class CompressedLinear(nn.Module):
    """
    Camada Linear baseada em fatores comprimidos U, S, Vh.
    Suporta armazenamento em INT8 com dequantização on-the-fly.
    """

    def __init__(
        self,
        factors: CompressedFactors,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.is_quantized = isinstance(factors.U, QuantizedTensor)
        
        # Store dimensions
        if self.is_quantized:
            self._in_features = factors.Vh.data.shape[1]
            self._out_features = factors.U.data.shape[0]
        else:
            self._in_features = factors.Vh.shape[1]
            self._out_features = factors.U.shape[0]
        
        if self.is_quantized:
            # Register quantized buffers
            self._register_quantized_tensor("U", factors.U)
            self._register_quantized_tensor("S", factors.S)
            self._register_quantized_tensor("Vh", factors.Vh)
        else:
            # Register float buffers
            self.register_buffer("_U", factors.U)
            self.register_buffer("_S", factors.S)
            self.register_buffer("_Vh", factors.Vh)

        if bias is not None:
            self.bias = nn.Parameter(bias.clone().detach())
        else:
            self.register_parameter('bias', None)

    def _register_quantized_tensor(self, name: str, qt: QuantizedTensor):
        self.register_buffer(f"_{name}_data", qt.data)
        self.register_buffer(f"_{name}_scale", torch.tensor(qt.scale))
        self.register_buffer(f"_{name}_zero_point", torch.tensor(qt.zero_point))

    def _dequantize(self, name: str) -> torch.Tensor:
        data = getattr(self, f"_{name}_data")
        scale = getattr(self, f"_{name}_scale")
        zero_point = getattr(self, f"_{name}_zero_point")
        return (data.to(torch.float32) - zero_point.to(torch.float32)) * scale.to(torch.float32)

    def _get_U(self) -> torch.Tensor:
        if self.is_quantized:
            return self._dequantize("U")
        else:
            return self._U

    def _get_S(self) -> torch.Tensor:
        if self.is_quantized:
            return self._dequantize("S")
        else:
            return self._S

    def _get_Vh(self) -> torch.Tensor:
        if self.is_quantized:
            return self._dequantize("Vh")
        else:
            return self._Vh

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def out_features(self) -> int:
        return self._out_features

    @torch.no_grad()
    def materialize_weight(self) -> torch.Tensor:
        U = self._get_U()
        S = self._get_S()
        Vh = self._get_Vh()
        U_scaled = U * S.unsqueeze(0)
        W_hat = U_scaled @ Vh
        return W_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize on the fly
        U = self._get_U()
        S = self._get_S()
        Vh = self._get_Vh()

        orig_shape = x.shape
        last_dim = orig_shape[-1]

        if last_dim != self.in_features:
            raise ValueError(
                f"CompressedLinear expected input last dim={self.in_features}, got {last_dim}"
            )

        x_flat = x.view(-1, last_dim)
        
        V = Vh.t()
        h = x_flat @ V
        h = h * S.unsqueeze(0)
        y = h @ U.t()

        if self.bias is not None:
            y = y + self.bias

        new_shape = orig_shape[:-1] + (self.out_features,)
        y = y.view(*new_shape)

        return y
