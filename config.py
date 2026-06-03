from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    hf_token: str = Field(..., description="HuggingFace API token")
    model_id: str = Field(..., description="HuggingFace model ID (e.g. google/gemma-4-E2B-it)")
    hf_cache_dir: str = Field("./model_cache", description="Local directory for cached model weights")

    host: str = Field("0.0.0.0")
    port: int = Field(8000)
    log_level: str = Field("info")

    max_new_tokens: int = Field(512)
    default_temperature: float = Field(0.7)
    default_top_p: float = Field(0.9)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
