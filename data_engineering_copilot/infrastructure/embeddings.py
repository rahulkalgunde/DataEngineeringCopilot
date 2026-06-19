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
        # transformers deprecates the `cache_dir`/`cache_folder` argument; pass via kwargs
        try:
            self.model = SentenceTransformer(
                model_name,
                model_kwargs={"cache_dir": str(cache_dir)},
                config_kwargs={"cache_dir": str(cache_dir)},
                processor_kwargs={"cache_dir": str(cache_dir)},
                local_files_only=local_files_only,
            )
        except Exception as exc:  # pragma: no cover - runtime HF errors depend on environment
            msg = str(exc)
            if local_files_only and (
                "outgoing traffic has been disabled" in msg
                or "Cannot find the requested files in the disk cache" in msg
                or "LocalEntryNotFoundError" in msg
            ):
                raise RuntimeError(
                    "Embedding model not found in local cache and downloads are disabled. "
                    "Either run `python scripts/download_embedding_model.py` to cache the model locally, "
                    "or set `embedding_local_files_only=False` in your AppSettings to allow downloads."
                ) from exc
            raise

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
