# Memvora Backend

Memvora is an AI-powered personal memory vault. The backend turns notes, files, links, and screenshots into searchable semantic memories, then uses grounded AI to answer questions from a user's own saved context.

This service is built as a production-minded FastAPI API with Supabase Auth, Supabase Postgres, pgvector search, local embeddings, Gemini/OpenRouter LLM fallbacks, URL-ingestion hardening, rate limits, and careful data boundaries.

## Product Capabilities

- Save memories from notes, pasted text, PDFs, TXT/Markdown files, public links, and screenshots.
- Extract text and visual context from screenshots with Gemini OCR.
- Preserve screenshot previews for later viewing when the Supabase `image_data_url` column is present.
- Convert saved content into semantic chunks for vector search.
- Ask grounded AI questions over the user's relevant memories through AI Chat.
- Summarize recent memories into themes, details, open questions, and next actions.
- Use forgiving retrieval with lower-threshold semantic matches, typo-tolerant keyword fallback, and recent-memory fallback when no strong match appears.
- Serve lightweight memory lists and lazy-load full memory content/screenshots for faster dashboard responses.
- Fall back across multiple Gemini and OpenRouter models to reduce free-tier rate-limit failures.
- Support guest sessions through Supabase anonymous auth.
- Edit and delete memories while keeping search chunks in sync.
- Protect users with bounded uploads, bounded extraction, URL fetch restrictions, and provider privacy controls.

## Tech Stack

| Layer | Technology | Why It Matters |
| --- | --- | --- |
| API framework | FastAPI | High-performance Python API with async support and automatic request validation. |
| Runtime | Python | Strong AI/ML ecosystem and clean backend ergonomics. |
| Auth | Supabase Auth | Email/password and anonymous guest auth with JWT access tokens. |
| Database | Supabase Postgres | Managed relational database for users, memories, and chunks. |
| Vector search | pgvector | Stores embeddings inside Postgres and enables semantic similarity search. |
| Embeddings | sentence-transformers | Creates local embeddings without sending every memory to an LLM provider. |
| AI providers | Google Gemini + OpenRouter | Multi-provider model fallback for chat, summaries, and screenshot OCR. |
| Link parsing | httpx + BeautifulSoup + reader fallback | Fetches public pages directly and can fall back for LinkedIn/Facebook-style blockers. |
| PDF parsing | pypdf | Extracts text from uploaded PDFs with page limits. |
| Security controls | SSRF checks, CORS allowlist, size limits, rate limits | Reduces abuse risk for public deployments. |

## Architecture

1. The frontend sends an authenticated upload request with a Supabase JWT.
2. The backend verifies the token and determines the source type: note, file, link, or screenshot.
3. Content is extracted with strict limits:
   - Upload byte limits
   - Image byte limits
   - PDF page limits
   - URL response byte limits
   - Memory length and chunk-count limits
4. Text is chunked and embedded locally.
5. The memory and its chunks are stored in Supabase.
6. Chat/search generate embeddings for the user query and selected query variants.
7. Supabase pgvector returns nearest chunks.
8. The backend merges semantic results with fuzzy keyword and recent-memory fallbacks.
9. Relevant or closest-available memory context is sent to Gemini/OpenRouter for grounded answers.

## Security And Privacy

Memvora treats user memory as sensitive data.

- Backend-only secrets: `SUPABASE_SERVICE_KEY`, `GEMINI_API_KEY`, and `OPENROUTER_API_KEY` must never be exposed to the frontend.
- SSRF protection: link ingestion blocks localhost, loopback, private, link-local, non-global IPs, embedded credentials, non-standard ports, and unsafe redirects.
- Bounded extraction: uploads, URLs, PDFs, images, chunks, and memory size are capped.
- AI privacy switches:
  - `AI_EXTERNAL_PROCESSING_ENABLED=false` disables chat and summary provider calls.
  - `AI_IMAGE_PROCESSING_ENABLED=false` saves screenshots without sending images to Gemini OCR.
  - `LINK_READER_FALLBACK_ENABLED=false` disables the public reader fallback for sites that block direct server-side reading.
- Prompt-injection boundary: memory content is treated as untrusted evidence, not system instructions.
- CORS is exact-origin based in production. Avoid wildcard origins when credentials or auth are involved.
- Service-role access remains server-only. Backend queries still filter by authenticated `user_id`.

## API Routes

| Route | Purpose |
| --- | --- |
| `GET /health` | Health check. |
| `GET /me` | Return the current authenticated or guest user. |
| `GET /memories` | List lightweight recent memory metadata for fast dashboard loads. |
| `GET /memories?days=7` | List lightweight memory metadata from a time window. |
| `GET /memories/{memory_id}` | Fetch full content and screenshot preview for one memory. |
| `POST /upload` | Save notes, files, screenshots, and public links. |
| `PUT /memories/{memory_id}` | Edit a memory and rebuild its chunks. |
| `DELETE /memories/{memory_id}` | Delete a memory and its chunks. |
| `GET /search?query=...` | API-level semantic memory search. The UI now uses AI Chat as the primary retrieval surface. |
| `POST /chat` | Grounded chat over relevant or closest-available memories. |
| `POST /summary` | AI summary of recent memories. |

All routes except `/health` require:

```http
Authorization: Bearer <supabase_access_token>
```

## Supabase Schema

Run the project SQL setup in Supabase SQL Editor. The key tables are:

- `profiles`: user profile records.
- `memories`: saved memory metadata, original content, and optional screenshot preview.
- `memory_chunks`: searchable text chunks with pgvector embeddings.

For screenshot previews, the `memories` table needs:

```sql
alter table public.memories
add column if not exists image_data_url text;
```

This is optional for normal text search, but required if users should reopen saved screenshots visually.

## Environment Variables

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openrouter/free
OPENROUTER_MODELS=openrouter/free,mistralai/mistral-7b-instruct:free,meta-llama/llama-3.2-3b-instruct:free

GEMINI_API_KEY=your_google_ai_studio_api_key
GEMINI_MODEL=gemini-3.1-flash-lite
GEMINI_MODELS=gemini-3.1-flash-lite,gemini-2.5-flash-lite,gemini-2.5-flash
AI_PROVIDER_ORDER=gemini,openrouter

MEMORY_SIMILARITY_THRESHOLD=0.20
MAX_REQUEST_BYTES=8388608
MAX_UPLOAD_BYTES=5242880
MAX_IMAGE_BYTES=3145728
MAX_SCREENSHOT_STORAGE_BYTES=2097152
MAX_URL_BYTES=1048576
MAX_MEMORY_CHARS=100000
MAX_CHUNKS_PER_MEMORY=80
MAX_PDF_PAGES=25
MAX_USER_MEMORIES=500
UPLOADS_PER_HOUR=60

AI_EXTERNAL_PROCESSING_ENABLED=true
AI_IMAGE_PROCESSING_ENABLED=true
LINK_READER_FALLBACK_ENABLED=true

SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_server_only_supabase_secret_or_service_role_key

FRONTEND_URL=http://localhost:5173
CORS_ORIGINS=http://localhost:5173
```

Notes:

- `SUPABASE_SERVICE_KEY` must be a server-side Supabase secret key or legacy service-role JWT. Do not use the anon key, publishable key, database password, or JWT secret.
- `GEMINI_MODELS` and `OPENROUTER_MODELS` are comma-separated fallback lists. Memvora tries up to three models per provider.
- `AI_PROVIDER_ORDER=gemini,openrouter` means Gemini is tried first, then OpenRouter.
- `MEMORY_SIMILARITY_THRESHOLD=0.20` is recommended for forgiving retrieval. The backend also clamps overly strict values for chat/search so spelling mistakes and weaker semantic matches still have a chance.
- `MAX_SCREENSHOT_STORAGE_BYTES` controls which screenshots get saved as previews. Larger screenshots still save OCR text, but not image previews.
- `CORS_ORIGINS` must exactly match deployed frontend origins, including `https://`.

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

The API will run at:

```text
http://localhost:8000
```

## Railway Deployment

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Deployment checklist:

- Add every required backend env var in Railway.
- Set `FRONTEND_URL` and `CORS_ORIGINS` to the deployed frontend URL.
- Keep all AI and Supabase service keys only in the backend service.
- Run the Supabase SQL setup before first production use.
- Enable Supabase anonymous sign-ins if guest mode should work.
- Configure Supabase email settings and redirect URLs for production auth.

## Operational Notes

- If Supabase email rate limits block signup, the frontend can offer guest session fallback.
- If Gemini OCR is rate-limited, screenshot memories still save with fallback text.
- The embedding model is warmed on startup. Startup may take a little longer, but the first search/chat/upload request should avoid the cold model-load penalty.
- Memory list responses intentionally omit full text and screenshot data. Use `GET /memories/{memory_id}` for full memory detail.
- Larger API responses are gzip-compressed when the client supports it.
- Chat uses closest-available memory context instead of immediately failing when semantic matches are weak.
- The current rate limiter is process-local. For heavy production traffic, move rate limiting to Redis, a gateway, or Supabase-backed counters.

## Repository

Backend repository: `IshitaGupta-IG/memvora`
