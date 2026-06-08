import asyncio
import base64
from dataclasses import dataclass
from html.parser import HTMLParser
import ipaddress
from io import BytesIO
import socket
from urllib.parse import urljoin, urlparse

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
IMAGE_OCR_TIMEOUT_SECONDS = 10
ALLOWED_URL_PORTS = {80, 443}
MAX_URL_REDIRECTS = 3
IGNORED_HTML_TAGS = {"script", "style", "nav", "footer", "header", "noscript", "svg"}


@dataclass(frozen=True)
class FileExtraction:
    text: str
    source_type: str
    image_data_url: str | None = None


class ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._ignored_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_name = tag.lower()
        if tag_name in IGNORED_HTML_TAGS:
            self._ignored_depth += 1
        if tag_name == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        tag_name = tag.lower()
        if tag_name in IGNORED_HTML_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
        if tag_name == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        if not self._ignored_depth and not self._in_title:
            self.text_parts.append(text)

    @property
    def title(self) -> str:
        return " ".join(self.title_parts).strip()

    @property
    def readable_text(self) -> str:
        return " ".join(self.text_parts).strip()


async def extract_text_from_file(file: UploadFile) -> FileExtraction:
    filename = file.filename or "Untitled upload"
    content = await read_limited_upload(file, settings.max_upload_bytes)
    content_type = file.content_type or ""

    if filename.lower().endswith(".pdf") or content_type == "application/pdf":
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        if len(reader.pages) > settings.max_pdf_pages:
            raise ValueError(f"PDF uploads are limited to {settings.max_pdf_pages} pages.")
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return FileExtraction(text.strip(), "pdf")

    if filename.lower().endswith((".txt", ".md", ".markdown")) or content_type.startswith("text/"):
        return FileExtraction(content.decode("utf-8", errors="ignore").strip(), "text")

    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) or content_type in SUPPORTED_IMAGE_TYPES:
        if len(content) > settings.max_image_bytes:
            raise ValueError(f"Image uploads are limited to {settings.max_image_bytes // (1024 * 1024)} MB.")
        normalized_content_type = content_type if content_type in SUPPORTED_IMAGE_TYPES else guess_image_content_type(filename)
        image_data_url = build_image_data_url(content, normalized_content_type)
        return FileExtraction(
            await extract_text_from_image(content, normalized_content_type, filename),
            "screenshot",
            image_data_url,
        )

    raise ValueError("Please upload a PDF, TXT, Markdown, PNG, JPG, or WebP file.")


async def read_limited_upload(file: UploadFile, max_bytes: int) -> bytes:
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(f"Uploads are limited to {max_bytes // (1024 * 1024)} MB.")
    return content


def guess_image_content_type(filename: str) -> str:
    lower_filename = filename.lower()
    if lower_filename.endswith(".png"):
        return "image/png"
    if lower_filename.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lower_filename.endswith(".webp"):
        return "image/webp"
    return "image/png"


def build_image_data_url(content: bytes, content_type: str) -> str | None:
    if len(content) > settings.max_screenshot_storage_bytes:
        return None
    if content_type not in SUPPORTED_IMAGE_TYPES:
        return None
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


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
    if not settings.ai_image_processing_enabled:
        return build_image_fallback_text(filename, "AI image processing is disabled for this deployment.")

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
        timeout = httpx.Timeout(IMAGE_OCR_TIMEOUT_SECONDS, connect=5)
        async with httpx.AsyncClient(timeout=timeout) as client:
            last_error: Exception | None = None
            for model in settings.gemini_model_list:
                try:
                    response = await client.post(GEMINI_URL_TEMPLATE.format(model=model), params={"key": settings.gemini_api_key}, json=payload)
                    response.raise_for_status()
                    extracted_text = extract_gemini_text(response.json())
                    break
                except httpx.HTTPStatusError as exc:
                    last_error = exc
                    if exc.response.status_code in {401, 403, 429}:
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
    normalized_url = await validate_public_url(url.strip())
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 Memvora/1.0"
        ),
        "Accept": "text/html,text/plain,application/xhtml+xml",
    }
    parsed_url = urlparse(normalized_url)

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
        response = await fetch_public_url(normalized_url, headers)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in {401, 403, 999}:
            return fallback_link_memory(
                "This link appears to require login or blocks automated reading."
            )
        return fallback_link_memory("Memvora could not read the page content from this link.")
    except ValueError as exc:
        raise exc
    except Exception:
        return fallback_link_memory("Memvora could not connect to this link.")

    content_type = response.headers.get("content-type", "")
    domain = urlparse(str(response.url)).netloc.replace("www.", "") or parsed_url.netloc.replace("www.", "")

    if "text/plain" in content_type:
        title = f"Saved link from {domain}"
        text = response.text
    else:
        try:
            title, text = parse_html_content(response.text, domain)
        except Exception:
            return fallback_link_memory("Memvora saved the link but could not parse the page content.")

    if not text:
        text = "No readable page text was found. Add a pasted excerpt or note with this link."

    if response.extensions.get("memvora_truncated"):
        text = f"{text}\n\n[Memvora read the first {settings.max_url_bytes // 1024} KB of this page to keep link ingestion fast and safe.]"

    memory_text = f"Source URL: {response.url}\nSource domain: {domain}\n\n{text}"
    return memory_text.strip(), title, domain


def parse_html_content(html: str, domain: str) -> tuple[str, str]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        parser = ReadableHTMLParser()
        parser.feed(html)
        title = parser.title or f"Saved link from {domain}"
        return title, parser.readable_text

    soup = BeautifulSoup(html, "html.parser")
    for element in soup(list(IGNORED_HTML_TAGS)):
        element.decompose()

    title = (soup.title.string or "").strip() if soup.title else ""
    title = title or f"Saved link from {domain}"

    main_content = soup.find("article") or soup.find("main") or soup.body or soup
    text = " ".join(main_content.get_text(" ").split())
    return title, text


async def fetch_public_url(url: str, headers: dict[str, str]) -> httpx.Response:
    current_url = url
    async with httpx.AsyncClient(timeout=20, follow_redirects=False, headers=headers, trust_env=False) as client:
        for _ in range(MAX_URL_REDIRECTS + 1):
            await validate_public_url(current_url)
            async with client.stream("GET", current_url) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        response.raise_for_status()
                    current_url = urljoin(current_url, location)
                    continue

                response.raise_for_status()
                body = bytearray()
                truncated = False
                async for chunk in response.aiter_bytes():
                    remaining = settings.max_url_bytes - len(body)
                    if remaining <= 0:
                        truncated = True
                        break
                    body.extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        truncated = True
                        break

                decoded_headers = dict(response.headers)
                decoded_headers.pop("content-encoding", None)
                decoded_headers.pop("content-length", None)
                decoded_headers.pop("transfer-encoding", None)

                return httpx.Response(
                    status_code=response.status_code,
                    headers=decoded_headers,
                    content=bytes(body),
                    request=response.request,
                    extensions={**response.extensions, "memvora_truncated": truncated},
                )

    raise ValueError("This link redirects too many times.")


async def validate_public_url(url: str) -> str:
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        raise ValueError("Please enter a valid http or https link.")
    if parsed_url.username or parsed_url.password:
        raise ValueError("Links with embedded usernames or passwords are not supported.")

    port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
    if port not in ALLOWED_URL_PORTS:
        raise ValueError("Only standard HTTP and HTTPS ports are supported.")

    hostname = parsed_url.hostname.lower()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".localhost"):
        raise ValueError("Localhost links are not supported.")

    await ensure_public_hostname(hostname)
    return parsed_url.geturl()


async def ensure_public_hostname(hostname: str) -> None:
    try:
        ipaddress.ip_address(hostname)
        addresses = [hostname]
    except ValueError:
        try:
            resolved = await asyncio.to_thread(socket.getaddrinfo, hostname, None, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ValueError("Memvora could not resolve this link.") from exc
        addresses = sorted({item[4][0] for item in resolved})

    if not addresses:
        raise ValueError("Memvora could not resolve this link.")

    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("Private, local, or internal network links are not supported.")


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
            if len(chunks) > settings.max_chunks_per_memory:
                raise ValueError(f"Memories are limited to {settings.max_chunks_per_memory} searchable chunks.")
        start += chunk_size - overlap

    return chunks


def preview_text(text: str, limit: int = 240) -> str:
    clean_text = " ".join(text.split())
    if len(clean_text) <= limit:
        return clean_text
    return clean_text[:limit].rstrip() + "..."
