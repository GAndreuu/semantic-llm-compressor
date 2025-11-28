"""
Pacote principal do Semantic LLM Compressor.

Ideia central:
- Comprimir pesos de LLMs em um estado semântico compacto (fatores low-rank + quantização).
- Oferecer um runtime que consegue inferir diretamente a partir desse estado comprimido.

Módulos principais:
- io: leitura/escrita de modelos (safetensors, index.json).
- algorithms: SVD, quantização, políticas de fatoração.
- compression: orquestração da compressão.
- runtime: leitores comprimidos para inferência.
- eval: métricas/pipelines de avaliação.
- cli: ponto de entrada por linha de comando.
"""

from .compression.compressor import ModelCompressor
from .runtime.patcher import patch_model_with_compressed_linears

__all__ = [
    "ModelCompressor",
    "patch_model_with_compressed_linears",
]
