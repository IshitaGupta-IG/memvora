from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    memory_id: str
    title: str
    chunks_created: int


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


class MemoryResponse(BaseModel):
    id: str
    title: str
    source_type: str
    original_content: str
    created_at: str


class MemoryUpdateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    original_content: str = Field(..., min_length=1)


class SearchResponse(BaseModel):
    results: list[dict]


class SummaryRequest(BaseModel):
    days: int = Field(default=30, ge=1, le=365)


class SummaryResponse(BaseModel):
    summary: str
    memories_count: int
