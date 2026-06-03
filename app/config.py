from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_OPENROUTER_MODEL = "mistralai/mistral-7b-instruct:free"


class Settings(BaseSettings):
    openrouter_api_key: str
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    supabase_url: str
    supabase_service_key: str
    frontend_url: str = "http://localhost:5173"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8-sig", extra="ignore")

    @field_validator("openrouter_model", mode="before")
    @classmethod
    def default_openrouter_model(cls, value: str | None) -> str:
        if value is None or not str(value).strip():
            return DEFAULT_OPENROUTER_MODEL
        return str(value).strip()


settings = Settings()
