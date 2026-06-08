from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_OPENROUTER_MODEL = "openrouter/free"
DEFAULT_OPENROUTER_MODELS = "openrouter/free,mistralai/mistral-7b-instruct:free,meta-llama/llama-3.2-3b-instruct:free"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
DEFAULT_GEMINI_MODELS = "gemini-3.1-flash-lite,gemini-2.5-flash-lite,gemini-2.5-flash"
DEFAULT_AI_PROVIDER_ORDER = "gemini,openrouter"


class Settings(BaseSettings):
    openrouter_api_key: str = ""
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    openrouter_models: str = DEFAULT_OPENROUTER_MODELS
    gemini_api_key: str = ""
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_models: str = DEFAULT_GEMINI_MODELS
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

    @field_validator("openrouter_models", mode="before")
    @classmethod
    def default_openrouter_models(cls, value: str | None) -> str:
        if value is None or not str(value).strip():
            return DEFAULT_OPENROUTER_MODELS
        return str(value).strip()

    @field_validator("gemini_model", mode="before")
    @classmethod
    def default_gemini_model(cls, value: str | None) -> str:
        if value is None or not str(value).strip():
            return DEFAULT_GEMINI_MODEL
        return str(value).strip()

    @field_validator("gemini_models", mode="before")
    @classmethod
    def default_gemini_models(cls, value: str | None) -> str:
        if value is None or not str(value).strip():
            return DEFAULT_GEMINI_MODELS
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
    def gemini_model_list(self) -> list[str]:
        return self._model_list(self.gemini_models, self.gemini_model, DEFAULT_GEMINI_MODEL)

    @property
    def openrouter_model_list(self) -> list[str]:
        return self._model_list(self.openrouter_models, self.openrouter_model, DEFAULT_OPENROUTER_MODEL)

    @staticmethod
    def _model_list(models: str, preferred_model: str, default_model: str) -> list[str]:
        ordered_models = []
        for model in [*models.split(","), preferred_model, default_model]:
            name = model.strip()
            if name and name not in ordered_models:
                ordered_models.append(name)
        return ordered_models[:3]

    @property
    def allowed_origins(self) -> list[str]:
        origins = ["http://localhost:5173", self.frontend_url]
        origins.extend(origin.strip() for origin in self.cors_origins.split(",") if origin.strip())
        return sorted({origin.rstrip("/") for origin in origins if origin.strip()})


settings = Settings()
