import logging
from typing import Optional


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configura logging global com um formato padrão.

    Chamar isso no início de CLIs ou scripts principais.
    """
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Retorna um logger com o nome desejado, usando o config global.
    """
    return logging.getLogger(name or "semantic_llm_compressor")
