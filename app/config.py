from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_OPENROUTER_MODEL = "openrouter/free"
DEFAULT_OPENROUTER_MODELS = "openrouter/free,mistralai/mistral-7b-instruct:free,meta-llama/llama-3.2-3b-instruct:free"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
DEFAULT_GEMINI_MODELS = "gemini-3.1-flash-lite,gemini-2.5-flash-lite,gemini-2.5-flash"
DEFAULT_AI_PROVIDER_ORDER = "gemini,openrouter"
DEFAULT_MEMORY_SIMILARITY_THRESHOLD = 0.35
DEFAULT_MAX_REQUEST_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_UPLOAD_BYTES = 5 * 1024 * 1024
DEFAULT_MAX_IMAGE_BYTES = 3 * 1024 * 1024
DEFAULT_MAX_SCREENSHOT_STORAGE_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_URL_BYTES = 1024 * 1024
DEFAULT_MAX_MEMORY_CHARS = 100_000
DEFAULT_MAX_CHUNKS_PER_MEMORY = 80
DEFAULT_MAX_PDF_PAGES = 25
DEFAULT_MAX_USER_MEMORIES = 500
DEFAULT_UPLOADS_PER_HOUR = 60
DEFAULT_AI_EXTERNAL_PROCESSING_ENABLED = True
DEFAULT_AI_IMAGE_PROCESSING_ENABLED = True
DEFAULT_LINK_READER_FALLBACK_ENABLED = True


class Settings(BaseSettings):
    openrouter_api_key: str = ""
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    openrouter_models: str = DEFAULT_OPENROUTER_MODELS
    gemini_api_key: str = ""
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_models: str = DEFAULT_GEMINI_MODELS
    ai_provider_order: str = DEFAULT_AI_PROVIDER_ORDER
    memory_similarity_threshold: float = DEFAULT_MEMORY_SIMILARITY_THRESHOLD
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES
    max_screenshot_storage_bytes: int = DEFAULT_MAX_SCREENSHOT_STORAGE_BYTES
    max_url_bytes: int = DEFAULT_MAX_URL_BYTES
    max_memory_chars: int = DEFAULT_MAX_MEMORY_CHARS
    max_chunks_per_memory: int = DEFAULT_MAX_CHUNKS_PER_MEMORY
    max_pdf_pages: int = DEFAULT_MAX_PDF_PAGES
    max_user_memories: int = DEFAULT_MAX_USER_MEMORIES
    uploads_per_hour: int = DEFAULT_UPLOADS_PER_HOUR
    ai_external_processing_enabled: bool = DEFAULT_AI_EXTERNAL_PROCESSING_ENABLED
    ai_image_processing_enabled: bool = DEFAULT_AI_IMAGE_PROCESSING_ENABLED
    link_reader_fallback_enabled: bool = DEFAULT_LINK_READER_FALLBACK_ENABLED
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

    @field_validator("memory_similarity_threshold", mode="before")
    @classmethod
    def default_memory_similarity_threshold(cls, value: float | str | None) -> float:
        if value is None or not str(value).strip():
            return DEFAULT_MEMORY_SIMILARITY_THRESHOLD
        return float(value)

    @field_validator(
        "max_request_bytes",
        "max_upload_bytes",
        "max_image_bytes",
        "max_screenshot_storage_bytes",
        "max_url_bytes",
        "max_memory_chars",
        "max_chunks_per_memory",
        "max_pdf_pages",
        "max_user_memories",
        "uploads_per_hour",
        mode="before",
    )
    @classmethod
    def positive_int(cls, value: int | str | None) -> int:
        if value is None or not str(value).strip():
            raise ValueError("Numeric settings cannot be empty.")
        parsed = int(value)
        if parsed < 1:
            raise ValueError("Numeric settings must be positive.")
        return parsed

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
