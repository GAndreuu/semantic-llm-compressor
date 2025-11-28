# Running GPT-Neo 1.3B Compression

This guide walks you through compressing the `EleutherAI/gpt-neo-1.3B` model using the "K-proj only" strategy.

## Prerequisites

- 16GB+ RAM (for loading the original model)
- Python 3.10+
- Dependencies installed (`pip install -r requirements.txt`)

## Steps

### 1. Download the Original Model

```bash
# Using huggingface-cli
huggingface-cli download EleutherAI/gpt-neo-1.3B --local-dir models/original/gpt-neo-1.3B --local-dir-use-symlinks False
```

### 2. Run Compression

We use the CLI to compress the model. The default policy in `factorization_policy.py` handles the layer selection (K-proj only).

```bash
python -m semantic_llm_compressor.cli.compress_cli \
  --input_dir models/original/gpt-neo-1.3B \
  --output_dir models/compressed/gpt-neo-1.3B-kproj-only \
  --rank 128 \
  --quant_bits 8 \
  --min_dim_for_compression 1024
```

**Note:** The rank argument sets the *base* rank. The adaptive policy may increase this for boundary layers (e.g., to 384).

### 3. Verify Results

Run the benchmark script to compare the original and compressed models.

```bash
python scripts/benchmark_gpt_neo.py \
  --original_dir models/original/gpt-neo-1.3B \
  --compressed_dir models/compressed/gpt-neo-1.3B-kproj-only \
  --output_json results_kproj_only.json
```

## Expected Output

You should see output similar to `examples/sample_results_kproj_only.json`:

- **Memory**: ~98 MB (vs ~138 MB original)
- **Speed**: ~4.0 tokens/sec
- **Cosine Similarity**: > 0.99
- **Text Generation**: Coherent English text (no gibberish or infinite loops).
