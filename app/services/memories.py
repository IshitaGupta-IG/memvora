from fastapi import HTTPException

from app.services.embeddings import create_embedding, create_embeddings
from app.services.text_processing import chunk_text
from app.supabase_client import supabase


def create_memory(user_id: str, title: str, content: str, source_type: str) -> dict:
    chunks = chunk_text(content)
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text was found.")

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

    return {
        "memory": memory,
        "chunks_created": len(chunk_rows),
    }


def list_memories(user_id: str) -> list[dict]:
    response = (
        supabase.table("memories")
        .select("id,title,source_type,original_content,created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(20)
        .execute()
    )
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
    ).execute()
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

