import base64
import json
from urllib.parse import urlparse

from supabase import create_client
from supabase.lib.client_options import ClientOptions

from app.config import settings


SUPABASE_CONFIG_ERROR = (
    "Invalid Supabase configuration. Set SUPABASE_URL and SUPABASE_SERVICE_KEY "
    "to real Supabase project values. SUPABASE_SERVICE_KEY must be the Supabase "
    "secret key that starts with sb_secret_ or the legacy service_role JWT API key "
    "that starts with eyJ. Do not use the JWT secret, anon key, publishable key, "
    "or database password."
)


def _decode_jwt_payload(token: str) -> dict | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None

    try:
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        decoded = base64.urlsafe_b64decode(payload.encode("utf-8"))
        return json.loads(decoded)
    except Exception:
        return None


def validate_supabase_config() -> None:
    supabase_url = settings.supabase_url.strip()
    service_key = settings.supabase_service_key.strip()

    parsed_url = urlparse(supabase_url)
    if (
        not supabase_url
        or supabase_url == "your_supabase_project_url"
        or parsed_url.scheme not in {"http", "https"}
        or not parsed_url.netloc
    ):
        raise RuntimeError(SUPABASE_CONFIG_ERROR)

    if (
        not service_key
        or service_key == "your_supabase_service_role_key"
        or service_key == "your_supabase_service_role_api_key"
        or service_key == "your_supabase_secret_or_service_role_key"
        or service_key == "your_supabase_secret_key_starts_with_sb_secret_or_legacy_service_role_jwt_starts_with_eyJ"
        or service_key.startswith("sb_publishable_")
        or service_key.startswith("sb_anon_")
    ):
        raise RuntimeError(SUPABASE_CONFIG_ERROR)

    if service_key.startswith("sb_secret_"):
        return

    jwt_payload = _decode_jwt_payload(service_key)
    if jwt_payload is None:
        raise RuntimeError(SUPABASE_CONFIG_ERROR)

    if jwt_payload.get("role") != "service_role":
        raise RuntimeError(SUPABASE_CONFIG_ERROR)


def create_supabase_client():
    validate_supabase_config()
    try:
        return create_client(
            settings.supabase_url,
            settings.supabase_service_key,
            options=ClientOptions(
                auto_refresh_token=False,
                persist_session=False,
            ),
        )
    except Exception as exc:
        raise RuntimeError(SUPABASE_CONFIG_ERROR) from exc


class LazySupabaseClient:
    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = create_supabase_client()
        return self._client

    def __getattr__(self, name: str):
        return getattr(self._get_client(), name)


supabase = LazySupabaseClient()
