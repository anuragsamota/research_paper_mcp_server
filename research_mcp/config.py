"""Runtime settings, read from environment variables (all prefixed ``RESEARCH_``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: str | os.PathLike = ".env") -> None:
    """Load KEY=VALUE lines from a .env file without overriding variables that are already set."""
    file = Path(os.environ.get("RESEARCH_ENV_FILE", path))
    if not file.is_file():
        return
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip().removeprefix("export ").strip(), value.strip().strip("\"'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def _env(name: str, default: str) -> str:
    return os.environ.get(f"RESEARCH_{name}", default)


@dataclass
class Settings:
    data_dir: Path
    # Embedding backend: "auto" | "hash" | "sentence-transformers" | "ollama".
    embedder: str = "auto"
    embedding_model: str = ""
    ollama_url: str = "http://localhost:11434"
    ollama_api_key: str = ""  # only needed for Ollama Cloud / authenticated proxies
    # LLM used for generation (ask_papers, write_literature_review, research-agent).
    llm_url: str = ""  # defaults to ollama_url
    llm_model: str = "llama3.1:8b"
    llm_timeout: float = 300.0
    llm_temperature: float = 0.2
    llm_num_ctx: int = 8192
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
        load_dotenv()
        data_dir = Path(_env("DATA_DIR", os.path.join(os.getcwd(), "data"))).expanduser().resolve()
        raw_dirs = _env("ALLOWED_DIRS", "")
        allowed = [Path(p).expanduser().resolve() for p in raw_dirs.split(os.pathsep) if p.strip()]
        settings = cls(
            data_dir=data_dir,
            embedder=_env("EMBEDDER", "auto"),
            embedding_model=_env("EMBEDDING_MODEL", ""),
            ollama_url=_env("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
            ollama_api_key=os.environ.get("OLLAMA_API_KEY", ""),
            llm_url=_env("LLM_URL", "").rstrip("/"),
            llm_model=_env("LLM_MODEL", "llama3.1:8b"),
            llm_timeout=float(_env("LLM_TIMEOUT", "300")),
            llm_temperature=float(_env("LLM_TEMPERATURE", "0.2")),
            llm_num_ctx=int(_env("LLM_NUM_CTX", "8192")),
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
