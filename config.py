from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # HuggingFace — token is optional; LiteRT community models are publicly accessible
    hf_token: Optional[str] = Field(None, description="HuggingFace API token (optional for public models)")
    model_id: str = Field(..., description="HuggingFace repo ID (e.g. litert-community/gemma-4-E2B-it-litert-lm)")
    model_filename: str = Field(..., description="Model filename inside the repo (e.g. gemma-4-E2B-it.litertlm)")
    hf_cache_dir: str = Field("./model_cache", description="Local directory for cached model file")

    # Server
    host: str = Field("0.0.0.0")
    port: int = Field(8000)
    log_level: str = Field("info")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
