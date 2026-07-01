# embeddings.py – Ollama-only embedding provider

import logging
import json
import urllib.request
from pathlib import Path
from data_engineering_copilot.config.settings import settings


logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddings:
    """Ollama embedding provider using the /api/embed endpoint.
    
    This class provides embeddings via Ollama's embedding API. It does not
    use local sentence-transformers models.
    """

    def __init__(self, model_name: str, cache_dir: Path, local_files_only: bool) -> None:
        """Initialize the embedding provider.
        
        Args:
            model_name: The embedding model name (must be "nomic-embed-text")
            cache_dir: Directory for caching (unused, kept for compatibility)
            local_files_only: Whether to use local files only (unused, kept for compatibility)
        """
        self.model_name = model_name
        self.ollama_base_url = settings.ollama_base_url.rstrip('/')
        logger.info(
            "Using Ollama embedding model %s at %s",
            model_name,
            self.ollama_base_url,
        )

    def _slice_texts_into_batches(self, texts: list[str], batch_size: int) -> list[list[str]]:
        """Slice texts into manageable batches to prevent Ollama OOM errors.
        
        On resource-constrained machines (e.g., 16GB RAM), sending hundreds of texts
        simultaneously to Ollama causes it to build an enormous evaluation matrix,
        exhausting memory and crashing the process. This method slices the input
        into smaller batches (default 32 items) to keep Ollama's working memory stable.
        
        Args:
            texts: List of text strings to embed
            batch_size: Maximum number of texts per batch
            
        Returns:
            List of text batches, each with at most batch_size items
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        batches = []
        for i in range(0, len(texts), batch_size):
            batches.append(texts[i : i + batch_size])
        return batches

    def _ollama_embed_single_batch(self, texts: list[str]) -> list[list[float]]:
        """Call Ollama embeddings endpoint for a single batch of texts.

        Returns a list of vectors for each input text.
        """
        payload = json.dumps({"model": self.model_name, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.ollama_base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:
                resp_data = json.load(response)
        except Exception as exc:
            raise RuntimeError(f"Failed to get embeddings from Ollama: {exc}") from exc
        
        # Log response structure for debugging
        logger.debug(
            "Ollama embeddings response keys=%s",
            sorted(resp_data.keys()) if isinstance(resp_data, dict) else "not a dict",
        )
        
        # Ollama /api/embed endpoint returns {"embeddings": [[...], [...], ...]}
        if "embeddings" not in resp_data:
            raise RuntimeError(
                f"Ollama embeddings response missing 'embeddings' key. "
                f"Response keys: {sorted(resp_data.keys()) if isinstance(resp_data, dict) else 'invalid response'}. "
                f"Response: {json.dumps(resp_data)[:500]}"
            )
        
        embeddings = resp_data["embeddings"]
        
        # Validate embeddings is a list
        if not isinstance(embeddings, list):
            raise RuntimeError(
                f"Ollama 'embeddings' value is not a list. Got type {type(embeddings).__name__}. "
                f"Response: {json.dumps(resp_data)[:500]}"
            )
        
        # Validate that we got embeddings for all input texts
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Ollama returned {len(embeddings)} embeddings for {len(texts)} input texts. "
                f"Expected one embedding per input text. "
                f"Check that the Ollama embedding model is properly configured and responding. "
                f"Response keys: {sorted(resp_data.keys())}"
            )
        
        # Validate embedding dimensions
        self._validate_embedding_dimensions(embeddings, texts)
        
        return embeddings

    def _ollama_embed(self, texts: list[str]) -> list[list[float]]:
        """Call Ollama embeddings endpoint with automatic batch slicing.

        Splits large text arrays into manageable batches to prevent OOM errors
        on resource-constrained machines. Results are concatenated in order.
        
        Returns a list of vectors for each input text.
        """
        batch_size = settings.embedding_batch_size
        batches = self._slice_texts_into_batches(texts, batch_size)
        
        # If only one batch, process directly
        if len(batches) == 1:
            return self._ollama_embed_single_batch(texts)
        
        # Process multiple batches and concatenate results
        logger.info(
            "Processing %d texts in %d batches (batch_size=%d)",
            len(texts),
            len(batches),
            batch_size,
        )
        
        all_embeddings: list[list[float]] = []
        for batch_idx, batch_texts in enumerate(batches, start=1):
            logger.debug(
                "Processing batch %d/%d with %d texts",
                batch_idx,
                len(batches),
                len(batch_texts),
            )
            batch_embeddings = self._ollama_embed_single_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        logger.info("Successfully embedded all %d texts in %d batches", len(texts), len(batches))
        return all_embeddings
    
    def _validate_embedding_dimensions(self, embeddings: list[list[float]], texts: list[str]) -> None:
        """Validate that all embeddings have the expected dimension.
        
        Raises RuntimeError if any embedding has incorrect dimension.
        """
        expected_dim = settings.embedding_dimension
        for i, emb in enumerate(embeddings):
            if not isinstance(emb, list):
                raise RuntimeError(
                    f"Embedding {i} is not a list. Got type {type(emb).__name__}. "
                    f"Text: {texts[i][:100]!r}"
                )
            if len(emb) == 0:
                raise RuntimeError(
                    f"Embedding {i} is empty (dimension 0). Expected dimension {expected_dim}. "
                    f"Text: {texts[i][:100]!r}. "
                    f"Check that the Ollama embedding model is properly configured."
                )
            if len(emb) != expected_dim:
                raise RuntimeError(
                    f"Embedding {i} has dimension {len(emb)}, expected {expected_dim}. "
                    f"Text: {texts[i][:100]!r}. "
                    f"Verify that the embedding model matches the configured embedding_dimension in settings."
                )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using Ollama."""
        vectors = self._ollama_embed(texts)
        logger.info("Embedded texts count=%s", len(texts))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        results = self.embed_texts([text])
        if not results or results[0] is None:
            raise RuntimeError(
                f"Embedding returned empty result for query: {text[:80]!r} "
                "Check the Ollama embedding model configuration."
            )
        return results[0]