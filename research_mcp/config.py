"""Runtime settings, read from environment variables (all prefixed ``RESEARCH_``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"RESEARCH_{name}", default)


@dataclass
class Settings:
    data_dir: Path
    # Embedding backend: "auto" | "hash" | "sentence-transformers" | "ollama".
    embedder: str = "auto"
    embedding_model: str = ""
    ollama_url: str = "http://localhost:11434"
    # Chunking (in words).
    chunk_size: int = 220
    chunk_overlap: int = 40
    # Network / download limits.
    http_timeout: float = 30.0
    max_pdf_bytes: int = 50 * 1024 * 1024
    semantic_scholar_api_key: str = ""
    allow_private_urls: bool = False
    # Local directories that `ingest_pdf` may read from.
    allowed_dirs: list[Path] = field(default_factory=list)

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def inbox_dir(self) -> Path:
        return self.data_dir / "inbox"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "library.sqlite3"

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(_env("DATA_DIR", os.path.join(os.getcwd(), "data"))).expanduser().resolve()
        raw_dirs = _env("ALLOWED_DIRS", "")
        allowed = [Path(p).expanduser().resolve() for p in raw_dirs.split(os.pathsep) if p.strip()]
        settings = cls(
            data_dir=data_dir,
            embedder=_env("EMBEDDER", "auto"),
            embedding_model=_env("EMBEDDING_MODEL", ""),
            ollama_url=_env("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
            chunk_size=int(_env("CHUNK_SIZE", "220")),
            chunk_overlap=int(_env("CHUNK_OVERLAP", "40")),
            http_timeout=float(_env("HTTP_TIMEOUT", "30")),
            max_pdf_bytes=int(_env("MAX_PDF_MB", "50")) * 1024 * 1024,
            semantic_scholar_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY", ""),
            allowed_dirs=allowed,
            allow_private_urls=_env("ALLOW_PRIVATE_URLS", "0").lower() in ("1", "true", "yes"),
        )
        if not settings.allowed_dirs:
            settings.allowed_dirs = [settings.inbox_dir]
        return settings

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.pdf_dir, self.inbox_dir):
            d.mkdir(parents=True, exist_ok=True)
