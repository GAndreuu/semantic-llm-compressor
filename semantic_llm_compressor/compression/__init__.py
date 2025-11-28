"""
Orquestração de compressão:

- ModelCompressor: coordena leitura de shards, aplicação de SVD/quantização
  e escrita dos shards comprimidos.
- ShardWorker: lógica de compressão de um único shard (thread-safe).
"""

from .compressor import ModelCompressor

__all__ = ["ModelCompressor"]
