# semantic_llm_compressor/io/safetensor_reader.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from safetensors import safe_open


@dataclass
class SafeTensorReader:
    """
    Leitor simples de arquivos safetensors.

    Responsabilidades:
    - Carregar um shard para um dict[str, Tensor].
    - Permitir acesso individual por nome.
    - Ler metadados do arquivo.
    """
    path: Path
    _file_handle: Optional[object] = None

    def __post_init__(self):
        # Mantém o arquivo aberto se necessário, ou abre sob demanda
        pass

    @classmethod
    def open(cls, path: Path) -> "SafeTensorReader":
        return cls(path=path)

    def get_tensors(self) -> Dict[str, torch.Tensor]:
        """Carrega todos os tensores de uma vez (cuidado com memória)."""
        tensors = {}
        with safe_open(self.path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        return tensors

    @property
    def tensors(self) -> Dict[str, torch.Tensor]:
        """Compatibilidade com código anterior que acessava .tensors diretamente."""
        return self.get_tensors()

    def get_metadata(self) -> Dict[str, str]:
        """Lê o bloco de metadados do arquivo."""
        with safe_open(self.path, framework="pt", device="cpu") as f:
            return f.metadata() or {}

    def get_tensor(self, name: str) -> torch.Tensor:
        with safe_open(self.path, framework="pt", device="cpu") as f:
            return f.get_tensor(name)
