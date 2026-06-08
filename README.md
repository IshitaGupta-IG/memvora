# Memvora Backend

FastAPI backend for Memvora, an AI semantic memory vault powered by Supabase pgvector, sentence-transformers, OpenRouter, and optional Gemini fallback.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

## Environment Variables

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=mistralai/mistral-7b-instruct:free
GEMINI_API_KEY=your_google_ai_studio_api_key
GEMINI_MODEL=gemini-2.5-flash
AI_PROVIDER_ORDER=gemini,openrouter
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_secret_key_starts_with_sb_secret_or_legacy_service_role_jwt_starts_with_eyJ
FRONTEND_URL=http://localhost:5173
CORS_ORIGINS=http://localhost:5173
```

`SUPABASE_SERVICE_KEY` must be Supabase's server-side secret key, usually starting with `sb_secret_`, or the legacy `service_role` JWT API key, usually starting with `eyJ`. Do not use the database password, JWT secret, anon key, or publishable key.

`OPENROUTER_MODEL` controls the first chat and summary model. If it is missing or blank, Memvora uses `mistralai/mistral-7b-instruct:free`.

`GEMINI_API_KEY` is optional but recommended. When OpenRouter free models are rate-limited or temporarily unavailable, Memvora automatically falls back to Gemini. If `OPENROUTER_API_KEY` is blank and `GEMINI_API_KEY` is set, Memvora uses Gemini directly.

`AI_PROVIDER_ORDER` controls provider priority. The default is `gemini,openrouter`, which uses Gemini first and falls back to OpenRouter.

## API Routes

- `GET /health` - backend health check
- `GET /me` - confirm the current signed-in or guest user
- `GET /memories` - list current user's recent memories
- `GET /memories?days=7` - list memories from a recent time window
- `POST /upload` - upload PDF/TXT/Markdown/images or pasted text
- `POST /upload` - also accepts `link_url` to save readable web links
- Screenshot/image uploads use Gemini to extract readable text and a short visual summary, so `GEMINI_API_KEY` is required for image memories.
- `GET /search?query=...` - semantic search over memories
- `POST /chat` - ask a question using retrieved memory context
- `POST /summary` - summarize recent memories and themes

All routes except `/health` require a Supabase access token:

```http
Authorization: Bearer <supabase_access_token>
```

## Deployment on Railway

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Add all environment variables in Railway before deploying.

Required Railway backend variables:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=mistralai/mistral-7b-instruct:free
GEMINI_API_KEY=your_google_ai_studio_api_key
GEMINI_MODEL=gemini-2.5-flash
AI_PROVIDER_ORDER=gemini,openrouter
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_secret_or_service_role_key
FRONTEND_URL=your_deployed_memvora_ui_url
CORS_ORIGINS=your_deployed_memvora_ui_url
```

Use full URLs for CORS values, including `https://`. For example: `https://memvora-ui-production.up.railway.app`.

## Guest Sign In

Guest users are Supabase anonymous users. Enable anonymous sign-ins in Supabase:

Supabase Dashboard -> Authentication -> Sign In / Providers -> Anonymous sign-ins.
