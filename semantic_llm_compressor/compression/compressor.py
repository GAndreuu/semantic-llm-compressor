# semantic_llm_compressor/compression/compressor.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Dict, Any
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import CompressionConfig
from ..io import ModelIndex
from ..logging_utils import get_logger
from ..algorithms import SVDDecomposer, Quantizer, FactorizationPolicy
from .shard_worker import ShardWorker


logger = get_logger(__name__)


@dataclass
class ModelCompressor:
    """
    Classe principal para comprimir um modelo LLM baseado em safetensors.

    Responsabilidades:
    - Carregar o index.json.
    - Percorrer shards e pesos, delegando a compressão para ShardWorker.
    - Gerar um novo diretório com shards comprimidos + novo index.json + metadata.
    """

    config: CompressionConfig

    def _load_index(self) -> ModelIndex:
        logger.info(f"Carregando index do diretório: {self.config.input_dir}")
        return ModelIndex.load(self.config.input_dir)

    def _get_unique_shards(self, index: ModelIndex) -> Set[Path]:
        """
        Retorna conjunto de paths relativos de shards a partir do weight_map.
        """
        rel_paths = set(index.weight_map.values())
        return {Path(p) for p in rel_paths}

    def compress(self) -> None:
        """
        Ponto de entrada principal.

        Fluxo:
        1. Carrega index.json do modelo original.
        2. Descobre lista de shards.
        3. Para cada shard, cria um ShardWorker e aplica compressão
           (possivelmente em paralelo).
        4. Gera um novo index.json apontando para shards comprimidos + metadata
           de compressão global.
        """
        # Garante que diretório de saída exista
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # 1) Carrega index
        index = self._load_index()

        # 2) Descobre shards únicos
        shard_rel_paths = self._get_unique_shards(index)
        logger.info(f"Encontrados {len(shard_rel_paths)} shards para compressão.")

        # 3) Instancia componentes de compressão (reusados entre shards)
        svd = SVDDecomposer(rank=self.config.rank)
        quantizer = Quantizer(num_bits=self.config.quant_bits)
        policy = FactorizationPolicy(config=self.config)

        # 4) Cria e executa ShardWorkers (paralelo com ThreadPoolExecutor)
        futures = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            for rel_path in shard_rel_paths:
                shard_in = self.config.input_dir / rel_path
                shard_out = self.config.output_dir / rel_path

                # Se já existe e não pode sobrescrever, pula
                if shard_out.exists() and not self.config.overwrite:
                    logger.warning(f"Shard de saída já existe e overwrite=False, pulando: {shard_out}")
                    continue

                shard_out.parent.mkdir(parents=True, exist_ok=True)

                worker = ShardWorker(
                    shard_path_in=shard_in,
                    shard_path_out=shard_out,
                    svd=svd,
                    quantizer=quantizer,
                    policy=policy,
                )

                futures.append(executor.submit(worker.run))

            # Espera conclusão e loga erros se houver
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    logger.exception(f"Erro durante compressão de shard: {e}")
                    raise

        # 5) Escreve novo index.json no diretório de saída
        new_index_path = self.config.output_dir / "model.safetensors.index.json"

        logger.info(f"Escrevendo novo index.json em: {new_index_path}")

        # Mantém mesmos weight_map e metadata, adicionando info de compressão
        index_dict: Dict[str, Any] = dict(index.metadata)
        index_dict["weight_map"] = index.weight_map
        index_dict["compression_config"] = self.config.to_dict()

        with new_index_path.open("w", encoding="utf-8") as f:
            json.dump(index_dict, f, indent=2)

        logger.info("Compressão concluída com sucesso.")
