from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.auth import get_current_user
from app.config import settings
from app.models import ChatRequest, ChatResponse, MemoryUpdateRequest, SearchResponse, SummaryRequest, SummaryResponse, UploadResponse
from app.services.memories import build_context, create_memory, delete_memory, list_memories, search_memories, update_memory
from app.services.openrouter import ask_openrouter, summarize_memories
from app.services.rate_limit import check_rate_limit
from app.services.text_processing import extract_text_from_file, extract_text_from_url, preview_text

app = FastAPI(title="Memvora API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def enforce_request_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > settings.max_request_bytes:
                return JSONResponse(status_code=413, content={"detail": "Request body is too large."})
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})
    return await call_next(request)


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
        check_rate_limit(user["id"], "upload", settings.uploads_per_hour, 3600)
        source_type = "note"
        content = note.strip()

        if len(note) > settings.max_memory_chars:
            raise HTTPException(status_code=400, detail=f"Notes are limited to {settings.max_memory_chars} characters.")
        if len(title) > 200:
            raise HTTPException(status_code=400, detail="Titles are limited to 200 characters.")

        if file and file.filename:
            extraction = await extract_text_from_file(file)
            source_type = extraction.source_type
            content = extraction.text
        elif link_url.strip():
            if len(link_url) > 2000:
                raise HTTPException(status_code=400, detail="Links are limited to 2000 characters.")
            link_text, link_title, domain = await extract_text_from_url(link_url)
            source_type = "link"
            content = link_text
            if note.strip():
                content = f"{content}\n\nUser note:\n{note.strip()}"
            if not title.strip():
                title = link_title or f"Saved link from {domain}"

        if not content:
            raise HTTPException(status_code=400, detail="Add a note, link, or upload a readable file.")
        if len(content) > settings.max_memory_chars:
            raise HTTPException(status_code=400, detail=f"Memories are limited to {settings.max_memory_chars} characters after extraction.")

        final_title = title.strip() or (file.filename if file else preview_text(content, 60)) or "Untitled memory"
        result = create_memory(
            user["id"],
            final_title,
            content,
            source_type,
            image_data_url=extraction.image_data_url if file and file.filename else None,
        )

        return UploadResponse(
            memory_id=result["memory"]["id"],
            title=result["memory"]["title"],
            chunks_created=result["chunks_created"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/memories/{memory_id}", response_model=UploadResponse)
async def edit_memory(memory_id: str, request: MemoryUpdateRequest, user: dict = Depends(get_current_user)) -> UploadResponse:
    if len(request.original_content) > settings.max_memory_chars:
        raise HTTPException(status_code=400, detail=f"Memories are limited to {settings.max_memory_chars} characters.")
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
    check_rate_limit(user["id"], "search", 120, 3600)
    results = search_memories(user["id"], query)
    return SearchResponse(results=results)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user: dict = Depends(get_current_user)) -> ChatResponse:
    check_rate_limit(user["id"], "chat", 60, 3600)
    chunks = search_memories(user["id"], request.message, limit=6)
    if not chunks:
        return ChatResponse(
            answer="I could not find anything relevant in your memories for that question yet. Try saving more context or asking in a different way.",
            sources=[],
        )
    context = build_context(chunks)
    answer = await ask_openrouter(request.message, context)
    return ChatResponse(answer=answer, sources=chunks)


@app.post("/summary", response_model=SummaryResponse)
async def summary(request: SummaryRequest, user: dict = Depends(get_current_user)) -> SummaryResponse:
    check_rate_limit(user["id"], "summary", 30, 3600)
    memories = list_memories(user["id"], days=request.days, limit=30)
    result = await summarize_memories(memories, request.days)
    return SummaryResponse(summary=result, memories_count=len(memories))
