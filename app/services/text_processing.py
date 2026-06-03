from io import BytesIO
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
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


async def extract_text_from_url(url: str) -> tuple[str, str, str]:
    parsed_url = urlparse(url.strip())
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("Please enter a valid http or https link.")

    normalized_url = parsed_url.geturl()
    headers = {
        "User-Agent": "MemvoraBot/1.0 (+https://memvora.app)",
        "Accept": "text/html,text/plain,application/xhtml+xml",
    }

    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers=headers) as client:
            response = await client.get(normalized_url)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in {401, 403, 999}:
            domain = parsed_url.netloc.replace("www.", "")
            fallback = (
                f"Source URL: {normalized_url}\n"
                f"Source domain: {domain}\n\n"
                "This link appears to require login or blocks automated reading. "
                "Add a pasted excerpt or note with the link so Memvora can remember the details."
            )
            return fallback, f"Saved link from {domain}", domain
        raise ValueError("Memvora could not read that link. Try pasting the article text with the URL.") from exc
    except Exception as exc:
        raise ValueError("Memvora could not read that link. Try pasting the article text with the URL.") from exc

    content_type = response.headers.get("content-type", "")
    domain = urlparse(str(response.url)).netloc.replace("www.", "") or parsed_url.netloc.replace("www.", "")

    if "text/plain" in content_type:
        title = f"Saved link from {domain}"
        text = response.text
    else:
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
            element.decompose()

        title = (soup.title.string or "").strip() if soup.title else ""
        title = title or f"Saved link from {domain}"

        main_content = soup.find("article") or soup.find("main") or soup.body or soup
        text = " ".join(main_content.get_text(" ").split())

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
