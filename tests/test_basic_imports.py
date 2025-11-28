import unittest
import torch
import semantic_llm_compressor
from semantic_llm_compressor.algorithms.svd_compressor import SVDCompressor
from semantic_llm_compressor.algorithms.quantization import Quantizer
from semantic_llm_compressor.runtime.compressed_layers import CompressedLinear

class TestBasicImports(unittest.TestCase):
    def test_imports(self):
        """Test if core modules can be imported successfully."""
        self.assertIsNotNone(semantic_llm_compressor)
        self.assertIsNotNone(SVDCompressor)
        self.assertIsNotNone(Quantizer)
        self.assertIsNotNone(CompressedLinear)

    def test_torch_version(self):
        """Ensure torch is available and version is sufficient."""
        print(f"Torch version: {torch.__version__}")
        self.assertTrue(torch.__version__.startswith("2."))

if __name__ == '__main__':
    unittest.main()
