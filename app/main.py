from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.auth import get_current_user
from app.config import settings
from app.models import ChatRequest, ChatResponse, SearchResponse, UploadResponse
from app.services.memories import build_context, create_memory, list_memories, search_memories
from app.services.openrouter import ask_openrouter
from app.services.text_processing import extract_text_from_file, preview_text

app = FastAPI(title="Memvora API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:5173"],
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
async def get_memories(user: dict = Depends(get_current_user)) -> dict:
    memories = list_memories(user["id"])
    return {"memories": memories}


@app.post("/upload", response_model=UploadResponse)
async def upload_memory(
    title: str = Form(default=""),
    note: str = Form(default=""),
    file: UploadFile | None = File(default=None),
    user: dict = Depends(get_current_user),
) -> UploadResponse:
    try:
        source_type = "note"
        content = note.strip()

        if file and file.filename:
            file_text, source_type = await extract_text_from_file(file)
            content = file_text

        if not content:
            raise HTTPException(status_code=400, detail="Add a note or upload a readable file.")

        final_title = title.strip() or (file.filename if file else preview_text(content, 60)) or "Untitled memory"
        result = create_memory(user["id"], final_title, content, source_type)

        return UploadResponse(
            memory_id=result["memory"]["id"],
            title=result["memory"]["title"],
            chunks_created=result["chunks_created"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
