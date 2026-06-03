from fastapi import Header, HTTPException

from app.supabase_client import SUPABASE_CONFIG_ERROR
from app.supabase_client import supabase


async def get_current_user(authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Please sign in to continue.")

    token = authorization.replace("Bearer ", "", 1).strip()

    try:
        user_response = supabase.auth.get_user(token)
        if not user_response.user:
            raise HTTPException(status_code=401, detail="Your session expired. Please sign in again.")
        email = user_response.user.email
        return {
            "id": user_response.user.id,
            "email": email,
            "is_guest": email is None,
        }
    except HTTPException:
        raise
    except RuntimeError as exc:
        if str(exc) == SUPABASE_CONFIG_ERROR:
            raise HTTPException(status_code=500, detail=SUPABASE_CONFIG_ERROR) from exc
        raise HTTPException(status_code=401, detail="Could not verify your session.") from exc
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Could not verify your session.") from exc
