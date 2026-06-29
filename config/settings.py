class AppSettings:
    # Existing fields
    chroma_dir: str = "chroma_db"
    collection_name: str = "data_engineering_docs"
    embedding_model_name: str = "nomic-embed-text"
    embedding_local_files_only: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:9b"
    
    # New field for Qdrant
    qdrant_url: str = "http://localhost:6333"
    