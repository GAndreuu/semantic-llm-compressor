"""
Módulo de IO: leitura/escrita de modelos e tensores.

Responsabilidades:
- Ler arquivos `model.safetensors.index.json`.
- Abrir shards safetensors individuais.
- Salvar tensores comprimidos e seus metadados.
"""

from .model_index import ModelIndex
from .safetensor_reader import SafeTensorReader
from .safetensor_writer import SafeTensorWriter

__all__ = [
    "ModelIndex",
    "SafeTensorReader",
    "SafeTensorWriter",
]
