import base64
from io import BytesIO
from urllib.parse import urlparse

import httpx
from fastapi import UploadFile

from app.config import settings


SUPPORTED_FILE_TYPES = {
    "application/pdf": "pdf",
    "text/plain": "text",
    "text/markdown": "markdown",
}
SUPPORTED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
GEMINI_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


async def extract_text_from_file(file: UploadFile) -> tuple[str, str]:
    filename = file.filename or "Untitled upload"
    content = await file.read()
    content_type = file.content_type or ""

    if filename.lower().endswith(".pdf") or content_type == "application/pdf":
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.strip(), "pdf"

    if filename.lower().endswith((".txt", ".md", ".markdown")) or content_type.startswith("text/"):
        return content.decode("utf-8", errors="ignore").strip(), "text"

    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) or content_type in SUPPORTED_IMAGE_TYPES:
        return await extract_text_from_image(content, content_type or guess_image_content_type(filename), filename), "screenshot"

    raise ValueError("Please upload a PDF, TXT, Markdown, PNG, JPG, or WebP file.")


def guess_image_content_type(filename: str) -> str:
    lower_filename = filename.lower()
    if lower_filename.endswith(".png"):
        return "image/png"
    if lower_filename.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lower_filename.endswith(".webp"):
        return "image/webp"
    return "image/png"


def extract_gemini_text(data: dict) -> str:
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Gemini returned an unexpected image response.") from exc

    text = "\n".join(part.get("text", "") for part in parts if part.get("text"))
    if not text.strip():
        raise RuntimeError("Gemini returned no readable text for this screenshot.")
    return text.strip()


def build_image_fallback_text(filename: str, reason: str) -> str:
    return (
        f"Screenshot upload: {filename}\n\n"
        f"{reason} Memvora saved this screenshot memory without OCR text. "
        "Add a short note or edit this memory later to make it easier to search and chat with."
    )


async def extract_text_from_image(content: bytes, content_type: str, filename: str = "screenshot") -> str:
    if not settings.gemini_api_key:
        return build_image_fallback_text(filename, "Gemini OCR is not configured.")

    if content_type not in SUPPORTED_IMAGE_TYPES:
        content_type = "image/png"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": content_type,
                            "data": base64.b64encode(content).decode("ascii"),
                        },
                    },
                    {
                        "text": (
                            "Extract all readable text from this screenshot. "
                            "Then add a short visual summary and any important context. "
                            "Return plain text only."
                        )
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 1200,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            last_error: Exception | None = None
            for model in settings.gemini_model_list:
                try:
                    response = await client.post(GEMINI_URL_TEMPLATE.format(model=model), params={"key": settings.gemini_api_key}, json=payload)
                    response.raise_for_status()
                    extracted_text = extract_gemini_text(response.json())
                    break
                except Exception as exc:
                    last_error = exc
            else:
                raise last_error or RuntimeError("Gemini returned no screenshot OCR response.")
    except httpx.HTTPStatusError as exc:
        return build_image_fallback_text(filename, "Gemini could not read this screenshot because the key, model, or rate limit failed.")
    except Exception as exc:
        return build_image_fallback_text(filename, "Memvora could not read text from this screenshot.")

    return f"Screenshot text and visual summary:\n\n{extracted_text}"


async def extract_text_from_url(url: str) -> tuple[str, str, str]:
    parsed_url = urlparse(url.strip())
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("Please enter a valid http or https link.")

    normalized_url = parsed_url.geturl()
    headers = {
        "User-Agent": "MemvoraBot/1.0 (+https://memvora.app)",
        "Accept": "text/html,text/plain,application/xhtml+xml",
    }

    def fallback_link_memory(reason: str) -> tuple[str, str, str]:
        domain = parsed_url.netloc.replace("www.", "")
        fallback = (
            f"Source URL: {normalized_url}\n"
            f"Source domain: {domain}\n\n"
            f"{reason} "
            "Memvora saved the link itself. Paste a short excerpt, comment, or your thoughts with the link "
            "to make this memory richer for search and chat."
        )
        return fallback, f"Saved link from {domain}", domain

    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers=headers) as client:
            response = await client.get(normalized_url)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in {401, 403, 999}:
            return fallback_link_memory(
                "This link appears to require login or blocks automated reading."
            )
        return fallback_link_memory("Memvora could not read the page content from this link.")
    except Exception:
        return fallback_link_memory("Memvora could not connect to this link.")

    content_type = response.headers.get("content-type", "")
    domain = urlparse(str(response.url)).netloc.replace("www.", "") or parsed_url.netloc.replace("www.", "")

    if "text/plain" in content_type:
        title = f"Saved link from {domain}"
        text = response.text
    else:
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(response.text, "html.parser")
            for element in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
                element.decompose()

            title = (soup.title.string or "").strip() if soup.title else ""
            title = title or f"Saved link from {domain}"

            main_content = soup.find("article") or soup.find("main") or soup.body or soup
            text = " ".join(main_content.get_text(" ").split())
        except Exception:
            return fallback_link_memory("Memvora saved the link but could not parse the page content.")

    if not text:
        text = "No readable page text was found. Add a pasted excerpt or note with this link."

    memory_text = f"Source URL: {response.url}\nSource domain: {domain}\n\n{text}"
    return memory_text.strip(), title, domain


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
