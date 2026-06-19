from __future__ import annotations

import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str, cache_dir: Path, local_files_only: bool) -> None:
        logger.info(
            "Loading embedding model model=%s cache_dir=%s local_files_only=%s",
            model_name,
            cache_dir,
            local_files_only,
        )
        self.model = SentenceTransformer(
            model_name,
            cache_folder=str(cache_dir),
            local_files_only=local_files_only,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        logger.info("Embedded texts count=%s", len(texts))
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
