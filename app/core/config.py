from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    TRITON_HOST: str = "triton"
    TRITON_HTTP_PORT: int = 8000
    TRITON_GRPC_PORT: int = 8001
    MODEL_NAME: str = "resnet50"
    MODEL_VERSION: str = "1"
    INPUT_SIZE: int = 224
    MAX_BATCH_SIZE: int = 8
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
