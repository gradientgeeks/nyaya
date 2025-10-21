"""
Configuration management for Nyaya backend.

Loads settings from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from dotenv import load_dotenv
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    app_name: str = "Nyaya Legal RAG API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Pinecone Settings
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="nyaya-legal-rag", env="PINECONE_INDEX_NAME")
    pinecone_environment: str = Field(default="us-east-1", env="PINECONE_ENVIRONMENT")
    pinecone_namespace_user: str = "user_documents"
    pinecone_namespace_training: str = "training_data"
    
    # Embedding Settings
    embedding_model: str = "google/embeddinggemma-300M"
    embedding_dimension: int = 384  # Truncated with MRL
    hf_token: str = Field(..., env="HF_TOKEN")
    
    # LLM Settings (Google VertexAI Gemini)
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    llm_model_name: str = "gemini-1.5-pro"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024
    
    # Role Classifier Settings
    classifier_model_path: str = Field(default="./models/inlegalbert_classifier.pt", env="CLASSIFIER_MODEL_PATH")
    classifier_confidence_threshold: float = 0.7
    
    # RAG Settings
    rag_top_k: int = 5
    rag_score_threshold: float = 0.7
    
    # Similarity Search Settings
    similarity_top_k: int = 10
    similarity_role_weights: dict = {
        "Facts": 0.25,
        "Issue": 0.25,
        "Reasoning": 0.30,
        "Decision": 0.20
    }
    
    # Context Management
    max_conversation_history: int = 10
    session_timeout_minutes: int = 30
    
    # File Upload Settings
    max_upload_size_mb: int = 10
    allowed_file_extensions: list = [".pdf", ".txt"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export for easy import
settings = get_settings()
