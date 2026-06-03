import httpx
from fastapi import HTTPException

from app.config import settings


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
FREE_FALLBACK_MODELS = [
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-2-9b-it:free",
    "meta-llama/llama-3-8b-instruct:free",
]


async def ask_openrouter(question: str, context: str) -> str:
    if not settings.openrouter_api_key:
        raise HTTPException(status_code=500, detail="OpenRouter API key is missing.")

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

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": settings.frontend_url,
        "X-Title": "Memvora",
    }

    models_to_try = [settings.openrouter_model]
    models_to_try.extend(model for model in FREE_FALLBACK_MODELS if model not in models_to_try)

    last_error: Exception | None = None
    try:
        async with httpx.AsyncClient(timeout=45) as client:
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
    except httpx.HTTPStatusError as exc:
        message = "OpenRouter could not complete the request. Check your API key, model, or rate limits."
        raise HTTPException(status_code=502, detail=message) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="AI response failed. Please try again.") from exc


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

    return await ask_openrouter(prompt, context)
