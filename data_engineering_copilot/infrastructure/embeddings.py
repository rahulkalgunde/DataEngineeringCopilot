# embeddings.py – simplified to Ollama only

import logging
import json
import urllib.request
from pathlib import Path
from data_engineering_copilot.config.settings import settings
from pathlib import Path

from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str, cache_dir: Path, local_files_only: bool) -> None:
        """Initialize the embedding provider.

        If ``model_name`` is ``"nomic-embed-text"`` we will use Ollama's
        ``/api/embeddings`` endpoint. Otherwise we fall back to a local
        ``SentenceTransformer`` model.
        """
        self.model_name = model_name
        if model_name == "nomic-embed-text":
            self.is_ollama = True
            self.ollama_base_url = settings.ollama_base_url.rstrip('/')
            logger.info(
                "Using Ollama embedding model %s at %s",
                model_name,
                self.ollama_base_url,
            )
        else:
            self.is_ollama = False
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

    def _ollama_embed(self, texts: list[str]) -> list[list[float]]:
        """Call Ollama embeddings endpoint.

        Returns a list of vectors for each input text.
        """
        payload = json.dumps({"model": self.model_name, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.ollama_base_url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:
                resp_data = json.load(response)
        except Exception as exc:
            raise RuntimeError(f"Failed to get embeddings from Ollama: {exc}") from exc
        if "embeddings" not in resp_data:
            raise RuntimeError(f"Unexpected Ollama response: {resp_data}")
        return resp_data["embeddings"]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using the selected backend.
        """
        if self.is_ollama:
            vectors = self._ollama_embed(texts)
        else:
            vectors = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vectors = vectors.tolist()
        logger.info("Embedded texts count=%s", len(texts))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.
        """
        return self.embed_texts([text])[0]
