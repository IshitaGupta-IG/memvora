from datetime import UTC, datetime, timedelta

from fastapi import HTTPException
from postgrest.exceptions import APIError

from app.services.embeddings import create_embedding, create_embeddings
from app.services.text_processing import chunk_text
from app.supabase_client import supabase

SUPABASE_SCHEMA_ERROR = (
    "Memvora database tables are missing. In Supabase, open SQL Editor and run "
    "the full setup_supabase.sql file from the Memvora project root. Make sure "
    "you are running it in the same Supabase project used by SUPABASE_URL."
)


def handle_supabase_error(exc: APIError) -> None:
    message = str(exc)
    if "public.memories" in message or "public.memory_chunks" in message or "match_memory_chunks" in message:
        raise HTTPException(status_code=503, detail=SUPABASE_SCHEMA_ERROR) from exc
    raise HTTPException(status_code=502, detail="Supabase request failed. Check backend logs and Supabase configuration.") from exc


def create_memory(user_id: str, title: str, content: str, source_type: str) -> dict:
    chunks = chunk_text(content)
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text was found.")

    try:
        memory_response = (
            supabase.table("memories")
            .insert(
                {
                    "user_id": user_id,
                    "title": title,
                    "source_type": source_type,
                    "original_content": content,
                }
            )
            .execute()
        )

        memory = memory_response.data[0]
        embeddings = create_embeddings(chunks)

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
        handle_supabase_error(exc)

    return {
        "memory": memory,
        "chunks_created": len(chunk_rows),
    }


def list_memories(user_id: str, days: int | None = None, limit: int = 20) -> list[dict]:
    query = (
        supabase.table("memories")
        .select("id,title,source_type,original_content,created_at")
        .eq("user_id", user_id)
    )

    if days:
        since = datetime.now(UTC) - timedelta(days=days)
        query = query.gte("created_at", since.isoformat())

    try:
        response = query.order("created_at", desc=True).limit(limit).execute()
    except APIError as exc:
        handle_supabase_error(exc)
    return response.data


def search_memories(user_id: str, query: str, limit: int = 5) -> list[dict]:
    if not query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    embedding = create_embedding(query)
    response = supabase.rpc(
        "match_memory_chunks",
        {
            "query_embedding": embedding,
            "match_user_id": user_id,
            "match_count": limit,
        },
    )
    try:
        response = response.execute()
    except APIError as exc:
        handle_supabase_error(exc)
    return response.data


def build_context(chunks: list[dict]) -> str:
    if not chunks:
        return ""

    context_parts = []
    for index, chunk in enumerate(chunks, start=1):
        title = chunk.get("title", "Untitled")
        content = chunk.get("content", "")
        context_parts.append(f"[Memory {index}: {title}]\n{content}")

    return "\n\n".join(context_parts)
