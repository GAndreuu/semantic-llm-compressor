from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any

import torch
from safetensors.torch import save_file


@dataclass
class SafeTensorWriter:
    """
    Writer para criar shards safetensors de tensores comprimidos.

    Pode ser usado para:
    - salvar fatores U/S/Vh quantizados
    - salvar tensores não comprimidos (fallback)
    """
    path: Path
    tensors: Dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_tensor(self, name: str, tensor: torch.Tensor) -> None:
        self.tensors[name] = tensor

    def set_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        save_file(self.tensors, str(self.path), metadata=self.metadata)
