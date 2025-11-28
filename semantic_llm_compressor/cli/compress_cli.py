import argparse
from pathlib import Path

from ..config import CompressionConfig
from ..compression import ModelCompressor
from ..logging_utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress LLM weights semantically.")
    parser.add_argument("--input_dir", type=str, required=True, help="Diretório do modelo original (HF safetensors).")
    parser.add_argument("--output_dir", type=str, required=True, help="Diretório de saída para modelo comprimido.")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quant_bits", type=int, default=8)
    parser.add_argument("--min_dim_for_compression", type=int, default=32)
    parser.add_argument("--compress_only_2d", action="store_true", default=True)
    args = parser.parse_args()

    setup_logging()

    config = CompressionConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        rank=args.rank,
        quant_bits=args.quant_bits,
        min_dim_for_compression=args.min_dim_for_compression,
        compress_only_2d=args.compress_only_2d,
        max_workers=args.workers,
        overwrite=args.overwrite,
    )

    compressor = ModelCompressor(config=config)
    compressor.compress()


if __name__ == "__main__":
    main()
