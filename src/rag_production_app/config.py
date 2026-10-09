from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3.5:2b"
    embed_model: str = "nomic-embed-text:latest"
    embed_dim: int = 768

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "docs"

    chunk_size: int = 1000
    chunk_overlap: int = 100
    embed_batch_size: int = 32
    min_score: float = 0.3  # tune this later with your eval harness


settings = Settings()
