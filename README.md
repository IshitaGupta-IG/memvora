# Memvora Backend

FastAPI backend for Memvora, an AI semantic memory vault powered by Supabase pgvector, sentence-transformers, and OpenRouter.

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
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_role_key
FRONTEND_URL=http://localhost:5173
```

## API Routes

- `GET /health` - backend health check
- `GET /me` - confirm the current signed-in or guest user
- `GET /memories` - list current user's recent memories
- `POST /upload` - upload PDF/TXT/Markdown or pasted text
- `GET /search?query=...` - semantic search over memories
- `POST /chat` - ask a question using retrieved memory context

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

## Guest Sign In

Guest users are Supabase anonymous users. Enable anonymous sign-ins in Supabase:

Supabase Dashboard -> Authentication -> Sign In / Providers -> Anonymous sign-ins.
