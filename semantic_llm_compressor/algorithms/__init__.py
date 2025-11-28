"""
Algoritmos de compressão de baixo nível:

- Decomposição SVD / low-rank.
- Quantização (ex: INT8).
- Políticas de fatoração (como escolher rank / quais tensores comprimir).
"""

from .svd_compressor import SVDDecomposer
from .quantization import Quantizer, QuantizedTensor
from .factorization_policy import FactorizationPolicy

__all__ = [
    "SVDDecomposer",
    "Quantizer",
    "QuantizedTensor",
    "FactorizationPolicy",
]
