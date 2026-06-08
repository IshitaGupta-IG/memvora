from datetime import UTC, datetime, timedelta

from fastapi import HTTPException
from postgrest.exceptions import APIError

from app.services.embeddings import create_embedding, create_embeddings
from app.services.text_processing import chunk_text
from app.supabase_client import supabase
from app.config import settings

SUPABASE_SCHEMA_ERROR = (
    "Memvora database tables are missing. In Supabase, open SQL Editor and run "
    "the full setup_supabase.sql file from the Memvora project root. Make sure "
    "you are running it in the same Supabase project used by SUPABASE_URL."
)
SEARCH_CANDIDATE_MULTIPLIER = 4
SEARCH_MIN_CANDIDATES = 12
COUNTRY_QUERY_ALIASES = {
    "thai": "thailand",
    "indian": "india",
    "american": "america",
    "british": "britain",
    "french": "france",
    "german": "germany",
    "japanese": "japan",
    "chinese": "china",
    "korean": "korea",
    "italian": "italy",
}


def handle_supabase_error(exc: APIError) -> None:
    message = str(exc)
    if "public.memories" in message or "public.memory_chunks" in message or "match_memory_chunks" in message:
        raise HTTPException(status_code=503, detail=SUPABASE_SCHEMA_ERROR) from exc
    if "image_data_url" in message:
        raise HTTPException(
            status_code=503,
            detail="Memvora database needs the latest screenshot migration. Run setup_supabase.sql in Supabase SQL Editor.",
        ) from exc
    raise HTTPException(status_code=502, detail="Supabase request failed. Check backend logs and Supabase configuration.") from exc


def create_memory(user_id: str, title: str, content: str, source_type: str, image_data_url: str | None = None) -> dict:
    chunks = chunk_text(content)
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text was found.")

    enforce_memory_quota(user_id)
    embeddings = create_embeddings(chunks)
    chunk_rows = []
    memory = None
    memory_payload = {
        "user_id": user_id,
        "title": title,
        "source_type": source_type,
        "original_content": content,
    }
    if image_data_url:
        memory_payload["image_data_url"] = image_data_url

    try:
        try:
            memory_response = supabase.table("memories").insert(memory_payload).execute()
        except APIError as exc:
            if image_data_url and "image_data_url" in str(exc):
                memory_payload.pop("image_data_url", None)
                memory_response = supabase.table("memories").insert(memory_payload).execute()
            else:
                raise

        memory = memory_response.data[0]
        chunk_rows = [
            {
                "memory_id": memory["id"],
                "user_id": user_id,
                "content": chunk,
                "chunk_index": index,
                "embedding": embedding,
            }
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        supabase.table("memory_chunks").insert(chunk_rows).execute()
    except APIError as exc:
        if memory:
            try:
                supabase.table("memories").delete().eq("id", memory["id"]).eq("user_id", user_id).execute()
            except APIError:
                pass
        handle_supabase_error(exc)

    return {
        "memory": memory,
        "chunks_created": len(chunk_rows),
    }


def list_memories(user_id: str, days: int | None = None, limit: int = 20) -> list[dict]:
    query = (
        supabase.table("memories")
        .select("id,title,source_type,original_content,created_at,image_data_url")
        .eq("user_id", user_id)
    )

    if days:
        since = datetime.now(UTC) - timedelta(days=days)
        query = query.gte("created_at", since.isoformat())

    try:
        response = query.order("created_at", desc=True).limit(limit).execute()
    except APIError as exc:
        if "image_data_url" not in str(exc):
            handle_supabase_error(exc)
        query = (
            supabase.table("memories")
            .select("id,title,source_type,original_content,created_at")
            .eq("user_id", user_id)
        )
        if days:
            query = query.gte("created_at", since.isoformat())
        try:
            response = query.order("created_at", desc=True).limit(limit).execute()
        except APIError as retry_exc:
            handle_supabase_error(retry_exc)
    return response.data


def enforce_memory_quota(user_id: str) -> None:
    try:
        response = supabase.table("memories").select("id", count="exact").eq("user_id", user_id).limit(1).execute()
    except APIError as exc:
        handle_supabase_error(exc)

    if (response.count or 0) >= settings.max_user_memories:
        raise HTTPException(status_code=429, detail=f"Memory limit reached. Current limit is {settings.max_user_memories} memories.")


def update_memory(user_id: str, memory_id: str, title: str, content: str) -> dict:
    chunks = chunk_text(content)
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text was found.")

    embeddings = create_embeddings(chunks)
    chunk_rows = [
        {
            "memory_id": memory_id,
            "user_id": user_id,
            "content": chunk,
            "chunk_index": index,
            "embedding": embedding,
        }
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    try:
        existing = supabase.table("memories").select("id,title,original_content").eq("id", memory_id).eq("user_id", user_id).limit(1).execute()
        if not existing.data:
            raise HTTPException(status_code=404, detail="Memory not found.")

        old_memory = existing.data[0]

        supabase.table("memory_chunks").delete().eq("memory_id", memory_id).eq("user_id", user_id).execute()
        try:
            supabase.table("memory_chunks").insert(chunk_rows).execute()
            memory_response = (
                supabase.table("memories")
                .update({"title": title, "original_content": content})
                .eq("id", memory_id)
                .eq("user_id", user_id)
                .execute()
            )
        except APIError:
            try:
                supabase.table("memory_chunks").delete().eq("memory_id", memory_id).eq("user_id", user_id).execute()
            except APIError:
                pass
            restore_memory_chunks(user_id, memory_id, old_memory.get("original_content") or "")
            raise
    except APIError as exc:
        handle_supabase_error(exc)

    return {
        "memory": memory_response.data[0] if memory_response.data else {"id": memory_id, "title": title},
        "chunks_created": len(chunk_rows),
    }


def delete_memory(user_id: str, memory_id: str) -> None:
    try:
        existing = supabase.table("memories").select("id").eq("id", memory_id).eq("user_id", user_id).limit(1).execute()
        if not existing.data:
            raise HTTPException(status_code=404, detail="Memory not found.")

        supabase.table("memories").delete().eq("id", memory_id).eq("user_id", user_id).execute()
    except APIError as exc:
        handle_supabase_error(exc)


def restore_memory_chunks(user_id: str, memory_id: str, content: str) -> None:
    chunks = chunk_text(content)
    if not chunks:
        return

    embeddings = create_embeddings(chunks)
    chunk_rows = [
        {
            "memory_id": memory_id,
            "user_id": user_id,
            "content": chunk,
            "chunk_index": index,
            "embedding": embedding,
        }
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    supabase.table("memory_chunks").insert(chunk_rows).execute()


def search_memories(user_id: str, query: str, limit: int = 5) -> list[dict]:
    clean_query = query.strip()
    if not clean_query:
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    candidate_limit = max(limit * SEARCH_CANDIDATE_MULTIPLIER, SEARCH_MIN_CANDIDATES)
    merged_results: dict[str, dict] = {}
    for search_query in expand_search_query(clean_query):
        embedding = create_embedding(search_query)
        response = supabase.rpc(
            "match_memory_chunks",
            {
                "query_embedding": embedding,
                "match_user_id": user_id,
                "match_count": candidate_limit,
            },
        )
        try:
            response = response.execute()
        except APIError as exc:
            handle_supabase_error(exc)

        for result in response.data:
            similarity = float(result.get("similarity") or 0)
            if similarity < settings.memory_similarity_threshold:
                continue

            result_key = str(result.get("chunk_id") or result.get("memory_id"))
            existing = merged_results.get(result_key)
            if not existing or similarity > float(existing.get("similarity") or 0):
                merged_results[result_key] = result

    return sorted(
        merged_results.values(),
        key=lambda result: float(result.get("similarity") or 0),
        reverse=True,
    )[:limit]


def expand_search_query(query: str) -> list[str]:
    variants = [query]
    words = query.split()
    lowered_words = [word.strip(".,;:!?()[]{}\"'").lower() for word in words]

    for index, word in enumerate(lowered_words):
        alias = COUNTRY_QUERY_ALIASES.get(word)
        if not alias:
            continue
        next_words = words.copy()
        next_words[index] = alias
        expanded_query = " ".join(next_words)
        if expanded_query not in variants:
            variants.append(expanded_query)

    return variants[:3]


def build_context(chunks: list[dict]) -> str:
    if not chunks:
        return ""

    context_parts = []
    for index, chunk in enumerate(chunks, start=1):
        title = chunk.get("title", "Untitled")
        content = chunk.get("content", "")
        context_parts.append(f"[Memory {index}: {title}]\n{content}")

    return "\n\n".join(context_parts)
