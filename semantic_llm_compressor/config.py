from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CompressionConfig:
    """
    Configurações de compressão de alto nível.

    Essa classe deve ser simples de serializar (e.g. para JSON/YAML),
    pois representa o "contrato" de como um modelo foi comprimido.
    """
    rank: int = 16
    quant_bits: int = 8
    max_workers: int = 4

    # Política de compressão
    min_dim_for_compression: int = 64
    compress_only_2d: bool = True

    # Caminhos
    input_dir: Path = Path(".")
    output_dir: Path = Path("./compressed_model")
    overwrite: bool = False

    # Dtype alvo no runtime
    runtime_dtype: str = "float16"

    # Futuro: flags para MoE, atentar para gating, etc.
    moe_aware: bool = True

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "quant_bits": self.quant_bits,
            "max_workers": self.max_workers,
            "min_dim_for_compression": self.min_dim_for_compression,
            "compress_only_2d": self.compress_only_2d,
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "overwrite": self.overwrite,
            "runtime_dtype": self.runtime_dtype,
            "moe_aware": self.moe_aware,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CompressionConfig":
        return cls(
            rank=data.get("rank", 16),
            quant_bits=data.get("quant_bits", 8),
            max_workers=data.get("max_workers", 4),
            min_dim_for_compression=data.get("min_dim_for_compression", 64),
            compress_only_2d=data.get("compress_only_2d", True),
            input_dir=Path(data.get("input_dir", ".")),
            output_dir=Path(data.get("output_dir", "./compressed_model")),
            overwrite=data.get("overwrite", False),
            runtime_dtype=data.get("runtime_dtype", "float16"),
            moe_aware=data.get("moe_aware", True),
        )
