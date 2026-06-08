from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_OPENROUTER_MODEL = "mistralai/mistral-7b-instruct:free"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_AI_PROVIDER_ORDER = "gemini,openrouter"


class Settings(BaseSettings):
    openrouter_api_key: str = ""
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    gemini_api_key: str = ""
    gemini_model: str = DEFAULT_GEMINI_MODEL
    ai_provider_order: str = DEFAULT_AI_PROVIDER_ORDER
    supabase_url: str
    supabase_service_key: str
    frontend_url: str = "http://localhost:5173"
    cors_origins: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8-sig", extra="ignore")

    @field_validator("openrouter_model", mode="before")
    @classmethod
    def default_openrouter_model(cls, value: str | None) -> str:
        if value is None or not str(value).strip():
            return DEFAULT_OPENROUTER_MODEL
        return str(value).strip()

    @field_validator("gemini_model", mode="before")
    @classmethod
    def default_gemini_model(cls, value: str | None) -> str:
        if value is None or not str(value).strip():
            return DEFAULT_GEMINI_MODEL
        return str(value).strip()

    @field_validator("ai_provider_order", mode="before")
    @classmethod
    def default_ai_provider_order(cls, value: str | None) -> str:
        if value is None or not str(value).strip():
            return DEFAULT_AI_PROVIDER_ORDER
        return str(value).strip()

    @property
    def ai_providers(self) -> list[str]:
        providers = []
        for provider in self.ai_provider_order.split(","):
            name = provider.strip().lower()
            if name in {"gemini", "openrouter"} and name not in providers:
                providers.append(name)
        return providers or ["gemini", "openrouter"]

    @property
    def allowed_origins(self) -> list[str]:
        origins = ["http://localhost:5173", self.frontend_url]
        origins.extend(origin.strip() for origin in self.cors_origins.split(",") if origin.strip())
        return sorted({origin.rstrip("/") for origin in origins if origin.strip()})


settings = Settings()
