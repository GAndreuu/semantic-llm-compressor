# semantic_llm_compressor/runtime/compressed_linear.py

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .compressed_tensor import CompressedTensor


class CompressedLinear(nn.Module):
    """
    Versão comprimida de nn.Linear baseada em uma decomposição SVD:

        W ≈ U Σ V^T

    Onde W tem shape [out_features, in_features].
    
    Forward pass eficiente:
        y = x W^T + b
        y = x (U Σ V^T)^T + b
        y = x (V Σ U^T) + b

    Ordem de operações para economizar FLOPs (assumindo rank r pequeno):
        1. h = x @ V      ( [batch, in] @ [in, r] -> [batch, r] )  (Note: V aqui é Vh.T)
           Mas temos Vh [r, in]. Então x @ Vh.T
        2. h = h * S      ( broadcast element-wise )
        3. y = h @ U.T    ( [batch, r] @ [r, out] -> [batch, out] )
        4. y = y + b
    """

    def __init__(self, compressed: CompressedTensor, bias: Optional[torch.Tensor] = None, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.compressed = compressed
        
        # Bias é mantido em full precision (ou fp16), geralmente não é comprimido
        if bias is not None:
            self.bias = nn.Parameter(bias.to(dtype), requires_grad=False)
        else:
            self.register_parameter('bias', None)
            
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Garante input no dtype correto
        x = x.to(self.dtype)

        # 1) Materializar fatores (U, S, Vh)
        # U: [out, r], S: [r], Vh: [r, in]
        U, S, Vh = self.compressed.materialize_factors(dtype=self.dtype)

        # 2) Computação fatorada: y = x @ Vh.T @ diag(S) @ U.T + b
        
        # Passo A: x @ Vh.T
        # x: [batch, ..., in_dim]
        # Vh.T: [in_dim, r]
        # Result: [batch, ..., r]
        # Usamos F.linear(x, Vh) -> x @ Vh.T
        h = F.linear(x, Vh)

        # Passo B: Escalar por S
        # h: [..., r], S: [r]
        h = h * S

        # Passo C: Projetar de volta para out_dim
        # h: [..., r]
        # U: [out_dim, r]
        # Queremos h @ U.T -> F.linear(h, U)
        y = F.linear(h, U)

        # Passo D: Bias
        if self.bias is not None:
            y = y + self.bias

        return y
