import httpx
from fastapi import HTTPException

from app.config import settings


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
FREE_FALLBACK_MODELS = [
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-2-9b-it:free",
    "meta-llama/llama-3-8b-instruct:free",
]


def build_chat_prompts(question: str, context: str) -> tuple[str, str]:
    system_prompt = (
        "You are Memvora, a helpful AI memory assistant. "
        "Answer using only the provided memory context when possible. "
        "If the context does not contain the answer, say that you could not find it in the user's memories. "
        "Be concise, grounded, and friendly."
    )

    user_prompt = f"""
Memory context:
{context or "No relevant memories were found."}

User question:
{question}
""".strip()

    return system_prompt, user_prompt


def extract_gemini_text(data: dict) -> str:
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Gemini returned an unexpected response.") from exc

    text = "\n".join(part.get("text", "") for part in parts if part.get("text"))
    if not text.strip():
        raise RuntimeError("Gemini returned an empty response.")
    return text.strip()


async def ask_gemini(question: str, context: str, client: httpx.AsyncClient | None = None) -> str:
    if not settings.gemini_api_key:
        raise RuntimeError("Gemini API key is missing.")

    system_prompt, user_prompt = build_chat_prompts(question, context)
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 700,
        },
    }
    params = {"key": settings.gemini_api_key}

    owns_client = client is None
    active_client = client or httpx.AsyncClient(timeout=45)
    last_error: Exception | None = None
    try:
        for model in settings.gemini_model_list:
            try:
                response = await active_client.post(GEMINI_URL_TEMPLATE.format(model=model), params=params, json=payload)
                response.raise_for_status()
                return extract_gemini_text(response.json())
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code not in RETRYABLE_STATUS_CODES:
                    continue
            except Exception as exc:
                last_error = exc
                continue
        raise last_error or RuntimeError("Gemini returned no answer.")
    finally:
        if owns_client:
            await active_client.aclose()


async def ask_openrouter_only(question: str, context: str, client: httpx.AsyncClient) -> str:
    if not settings.openrouter_api_key:
        raise RuntimeError("OpenRouter API key is missing.")

    system_prompt, user_prompt = build_chat_prompts(question, context)
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": settings.frontend_url,
        "X-Title": "Memvora",
    }

    models_to_try = settings.openrouter_model_list
    models_to_try.extend(model for model in FREE_FALLBACK_MODELS if model not in models_to_try)
    models_to_try = models_to_try[:3]

    last_error: Exception | None = None
    for model in models_to_try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 700,
        }

        try:
            response = await client.post(OPENROUTER_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if exc.response.status_code in {401, 403}:
                break

    raise last_error or RuntimeError("OpenRouter returned no answer.")


async def ask_openrouter(question: str, context: str) -> str:
    if not settings.openrouter_api_key and not settings.gemini_api_key:
        raise HTTPException(status_code=500, detail="Configure OPENROUTER_API_KEY or GEMINI_API_KEY.")

    last_error: Exception | None = None
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            for provider in settings.ai_providers:
                if provider == "gemini" and settings.gemini_api_key:
                    try:
                        return await ask_gemini(question, context, client)
                    except Exception as exc:
                        last_error = exc
                        continue

                if provider == "openrouter" and settings.openrouter_api_key:
                    try:
                        return await ask_openrouter_only(question, context, client)
                    except httpx.HTTPStatusError as exc:
                        last_error = exc
                        if exc.response.status_code not in RETRYABLE_STATUS_CODES:
                            continue
                    except Exception as exc:
                        last_error = exc
                        continue

            raise last_error or RuntimeError("No AI provider returned an answer.")
    except httpx.HTTPStatusError as exc:
        provider = "Gemini" if "generativelanguage.googleapis.com" in str(exc.request.url) else "OpenRouter"
        message = f"{provider} could not complete the request. Check your API key, model, or rate limits."
        raise HTTPException(status_code=502, detail=message) from exc
    except Exception as exc:
        if settings.openrouter_api_key and settings.gemini_api_key:
            message = "Gemini and OpenRouter both failed after trying their fallback models. Please try again shortly."
        elif settings.gemini_api_key:
            message = "Gemini failed after trying its fallback models. Please try again shortly."
        else:
            message = "OpenRouter failed after trying its fallback models. Please try again shortly."
        raise HTTPException(status_code=502, detail=message) from exc


async def summarize_memories(memories: list[dict], days: int) -> str:
    if not memories:
        return f"No memories were found from the last {days} days."

    context = "\n\n".join(
        f"{index}. {memory.get('title', 'Untitled')}\n{memory.get('original_content', '')[:1200]}"
        for index, memory in enumerate(memories, start=1)
    )

    prompt = (
        f"Summarize the user's memories from the last {days} days. "
        "Organize the answer into: key themes, important details, open questions, and suggested next actions."
    )

    try:
        return await ask_openrouter(prompt, context)
    except HTTPException:
        return build_local_summary(memories, days)


def build_local_summary(memories: list[dict], days: int) -> str:
    lines = [
        f"AI summary is temporarily unavailable, so here is a basic summary of memories from the last {days} days.",
        "",
        "Key themes:",
    ]

    for memory in memories[:8]:
        title = memory.get("title") or "Untitled"
        content = " ".join((memory.get("original_content") or "").split())
        snippet = content[:220].rstrip()
        if len(content) > 220:
            snippet += "..."
        lines.append(f"- {title}: {snippet or 'No readable preview available.'}")

    if len(memories) > 8:
        lines.append(f"- Plus {len(memories) - 8} more saved memories.")

    lines.extend(
        [
            "",
            "Suggested next actions:",
            "- Check that GEMINI_API_KEY is configured on the backend.",
            "- Redeploy the backend if the Gemini-first fallback commit is not live yet.",
        ]
    )
    return "\n".join(lines)
