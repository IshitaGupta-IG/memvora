from supabase import create_client

from app.config import settings


supabase = create_client(settings.supabase_url, settings.supabase_service_key)

