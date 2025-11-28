# scripts/1_compress_model.py

from pathlib import Path

from semantic_llm_compressor.config import CompressionConfig
from semantic_llm_compressor.compression import ModelCompressor
from semantic_llm_compressor.logging_utils import setup_logging

def main():
    input_dir = Path("models/original/tiny-gpt2")
    output_dir = Path("models/compressed/tiny-gpt2-r16")

    config = CompressionConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        rank=16,               # rank > hidden_size (2) means no compression gain, but SVD works
        quant_bits=8,
        max_workers=4,
        overwrite=True,
        min_dim_for_compression=2, # FIX: tiny-gpt2 hidden size is 2!
        compress_only_2d=True,
    )

    setup_logging()
    compressor = ModelCompressor(config=config)
    compressor.compress()

    print("Compressão concluída. Saída em:", output_dir)

if __name__ == "__main__":
    main()
