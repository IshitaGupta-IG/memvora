from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.auth import get_current_user
from app.config import settings
from app.models import ChatRequest, ChatResponse, MemoryUpdateRequest, SearchResponse, SummaryRequest, SummaryResponse, UploadResponse
from app.services.memories import build_context, create_memory, delete_memory, list_memories, search_memories, update_memory
from app.services.openrouter import ask_openrouter, summarize_memories
from app.services.text_processing import extract_text_from_file, extract_text_from_url, preview_text

app = FastAPI(title="Memvora API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_origin_regex=r"https://.*\.up\.railway\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "app": "Memvora"}


@app.get("/me")
async def me(user: dict = Depends(get_current_user)) -> dict:
    return {"user": user}


@app.get("/memories")
async def get_memories(days: int | None = None, user: dict = Depends(get_current_user)) -> dict:
    if days is not None and (days < 1 or days > 365):
        raise HTTPException(status_code=400, detail="Days must be between 1 and 365.")
    memories = list_memories(user["id"], days=days)
    return {"memories": memories}


@app.post("/upload", response_model=UploadResponse)
async def upload_memory(
    title: str = Form(default=""),
    note: str = Form(default=""),
    link_url: str = Form(default=""),
    file: UploadFile | None = File(default=None),
    user: dict = Depends(get_current_user),
) -> UploadResponse:
    try:
        source_type = "note"
        content = note.strip()

        if file and file.filename:
            file_text, source_type = await extract_text_from_file(file)
            content = file_text
        elif link_url.strip():
            link_text, link_title, domain = await extract_text_from_url(link_url)
            source_type = "link"
            content = link_text
            if note.strip():
                content = f"{content}\n\nUser note:\n{note.strip()}"
            if not title.strip():
                title = link_title or f"Saved link from {domain}"

        if not content:
            raise HTTPException(status_code=400, detail="Add a note, link, or upload a readable file.")

        final_title = title.strip() or (file.filename if file else preview_text(content, 60)) or "Untitled memory"
        result = create_memory(user["id"], final_title, content, source_type)

        return UploadResponse(
            memory_id=result["memory"]["id"],
            title=result["memory"]["title"],
            chunks_created=result["chunks_created"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/memories/{memory_id}", response_model=UploadResponse)
async def edit_memory(memory_id: str, request: MemoryUpdateRequest, user: dict = Depends(get_current_user)) -> UploadResponse:
    result = update_memory(user["id"], memory_id, request.title.strip(), request.original_content.strip())
    return UploadResponse(
        memory_id=result["memory"]["id"],
        title=result["memory"]["title"],
        chunks_created=result["chunks_created"],
    )


@app.delete("/memories/{memory_id}")
async def remove_memory(memory_id: str, user: dict = Depends(get_current_user)) -> dict:
    delete_memory(user["id"], memory_id)
    return {"status": "deleted"}


@app.get("/search", response_model=SearchResponse)
async def search(query: str, user: dict = Depends(get_current_user)) -> SearchResponse:
    results = search_memories(user["id"], query)
    return SearchResponse(results=results)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user: dict = Depends(get_current_user)) -> ChatResponse:
    chunks = search_memories(user["id"], request.message, limit=6)
    context = build_context(chunks)
    answer = await ask_openrouter(request.message, context)
    return ChatResponse(answer=answer, sources=chunks)


@app.post("/summary", response_model=SummaryResponse)
async def summary(request: SummaryRequest, user: dict = Depends(get_current_user)) -> SummaryResponse:
    memories = list_memories(user["id"], days=request.days, limit=30)
    result = await summarize_memories(memories, request.days)
    return SummaryResponse(summary=result, memories_count=len(memories))
