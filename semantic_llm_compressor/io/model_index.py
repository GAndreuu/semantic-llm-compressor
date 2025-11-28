from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
import json


@dataclass
class ModelIndex:
    """
    Representa o conteúdo de `model.safetensors.index.json`.

    Exemplo de campos:
    - weight_map: dict[str, str]
    - metadata: dict[str, Any]
    """
    weight_map: Dict[str, str]
    metadata: Dict[str, Any] = None

    @classmethod
    def load(cls, model_dir: Path) -> "ModelIndex":
        """
        Carrega o index.json e constrói um ModelIndex.
        Se o index não existir, tenta inferir de model.safetensors (single shard).
        """
        index_path = model_dir / "model.safetensors.index.json"
        
        try:
            # Tenta ler diretamente
            data = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = data.get("weight_map", {})
            metadata = {k: v for k, v in data.items() if k != "weight_map"}
            return cls(weight_map=weight_map, metadata=metadata)
        except FileNotFoundError:
            # Fallback: se não existe index, mas existe model.safetensors,
            # assumimos que é um modelo single-shard.
            single_shard_path = model_dir / "model.safetensors"
            if single_shard_path.exists():
                # Precisamos listar as chaves para criar o mapa.
                from safetensors import safe_open
                weight_map = {}
                try:
                    with safe_open(single_shard_path, framework="pt", device="cpu") as f:
                        keys = f.keys()
                    for k in keys:
                        weight_map[k] = "model.safetensors"
                    return cls(weight_map=weight_map, metadata={})
                except Exception as e:
                    raise RuntimeError(f"Falha ao ler {single_shard_path} para criar índice implícito: {e}")
            
            # Se chegou aqui, realmente não achou nada
            raise FileNotFoundError(f"Arquivo de índice não encontrado: {index_path}")

    def get_shard_for_weight(self, weight_name: str) -> Path:
        """
        Retorna o caminho do shard que contém o tensor `weight_name`.
        """
        shard_rel = self.weight_map[weight_name]
        # Chamador deve resolver com base no diretório raiz do modelo.
        return Path(shard_rel)
