import torch
import json
import shutil
from pathlib import Path
from safetensors.torch import save_file, load_file
from semantic_llm_compressor.config import CompressionConfig
from semantic_llm_compressor.compression import ModelCompressor
from semantic_llm_compressor.io import ModelIndex

def setup_dummy_model(base_dir: Path):
    input_dir = base_dir / "input_model"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy tensors
    tensors = {
        "layer1.weight": torch.randn(128, 128), # Should compress
        "layer1.bias": torch.randn(128),        # Should NOT compress
        "layer2.weight": torch.randn(64, 64),   # Should compress (if min_dim <= 64)
        "emb.weight": torch.randn(100, 64),     # Should compress
    }
    
    # Save to safetensors
    shard_name = "model.safetensors"
    save_file(tensors, input_dir / shard_name)
    
    # Create index.json
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {k: shard_name for k in tensors.keys()}
    }
    
    with open(input_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)
        
    return input_dir

def verify_compression(base_dir: Path):
    input_dir = base_dir / "input_model"
    output_dir = base_dir / "compressed_model"
    
    print("--- Setup Dummy Model ---")
    setup_dummy_model(base_dir)
    
    print("--- Run Compression ---")
    config = CompressionConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        rank=16,
        quant_bits=8,
        min_dim_for_compression=32,
        overwrite=True
    )
    
    compressor = ModelCompressor(config)
    compressor.compress()
    
    print("--- Verify Output ---")
    
    # 1. Check index.json
    new_index_path = output_dir / "model.safetensors.index.json"
    assert new_index_path.exists()
    
    with open(new_index_path, "r") as f:
        new_index = json.load(f)
        
    assert "compression_config" in new_index
    print("✅ index.json created with compression_config")
    
    # 2. Check Shard Content
    shard_path = output_dir / "model.safetensors"
    assert shard_path.exists()
    
    new_tensors = load_file(shard_path)
    
    # layer1.weight should be compressed -> replaced by U, S, Vh
    assert "layer1.weight" not in new_tensors
    assert "layer1.weight.U.quant" in new_tensors
    assert "layer1.weight.S.quant" in new_tensors
    assert "layer1.weight.Vh.quant" in new_tensors
    print("✅ layer1.weight compressed successfully")
    
    # layer1.bias should be intact
    assert "layer1.bias" in new_tensors
    print("✅ layer1.bias preserved intact")
    
    # 3. Check Metadata
    # We need to use safe_open to read metadata
    from safetensors import safe_open
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        
    assert "compression_meta" in metadata
    comp_meta = json.loads(metadata["compression_meta"])
    
    assert "layer1.weight.U" in comp_meta
    assert "scale" in comp_meta["layer1.weight.U"]
    print("✅ Compression metadata present and valid")
    
    print("\n🎉 Full Compression Flow Verified Successfully!")

if __name__ == "__main__":
    # Use a local temp dir
    base_path = Path("tests/temp_test_env")
    if base_path.exists():
        shutil.rmtree(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    try:
        verify_compression(base_path)
    finally:
        # Cleanup
        if base_path.exists():
            shutil.rmtree(base_path)
