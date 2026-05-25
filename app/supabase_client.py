import base64
import json
from urllib.parse import urlparse

from supabase import create_client

from app.config import settings


SUPABASE_CONFIG_ERROR = (
    "Invalid Supabase configuration. Set SUPABASE_URL and SUPABASE_SERVICE_KEY "
    "to real Supabase project values. SUPABASE_SERVICE_KEY must be the service_role "
    "API key, not the JWT secret or anon key."
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


validate_supabase_config()
try:
    supabase = create_client(settings.supabase_url, settings.supabase_service_key)
except Exception as exc:
    raise RuntimeError(SUPABASE_CONFIG_ERROR) from exc
