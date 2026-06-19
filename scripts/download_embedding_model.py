from __future__ import annotations

import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_engineering_copilot.config.settings import settings


def main() -> None:
    settings.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    SentenceTransformer(
        settings.embedding_model_name,
        cache_folder=str(settings.embedding_cache_dir),
        local_files_only=False,
    )
    print(f"Cached embedding model: {settings.embedding_model_name}")
    print(f"Cache directory: {settings.embedding_cache_dir}")


if __name__ == "__main__":
    main()
