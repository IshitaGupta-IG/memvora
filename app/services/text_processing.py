from io import BytesIO

from fastapi import UploadFile
from pypdf import PdfReader


SUPPORTED_FILE_TYPES = {
    "application/pdf": "pdf",
    "text/plain": "text",
    "text/markdown": "markdown",
}


async def extract_text_from_file(file: UploadFile) -> tuple[str, str]:
    filename = file.filename or "Untitled upload"
    content = await file.read()
    content_type = file.content_type or ""

    if filename.lower().endswith(".pdf") or content_type == "application/pdf":
        reader = PdfReader(BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip(), "pdf"

    if filename.lower().endswith((".txt", ".md", ".markdown")) or content_type.startswith("text/"):
        return content.decode("utf-8", errors="ignore").strip(), "text"

    raise ValueError("Please upload a PDF, TXT, or Markdown file.")


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    clean_text = " ".join(text.split())
    if not clean_text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(clean_text):
        end = start + chunk_size
        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def preview_text(text: str, limit: int = 240) -> str:
    clean_text = " ".join(text.split())
    if len(clean_text) <= limit:
        return clean_text
    return clean_text[:limit].rstrip() + "..."

