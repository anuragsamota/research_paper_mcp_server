"""SQLite persistence for paper metadata, text chunks and their embeddings."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authors TEXT NOT NULL DEFAULT '[]',
    year INTEGER,
    abstract TEXT NOT NULL DEFAULT '',
    venue TEXT NOT NULL DEFAULT '',
    doi TEXT NOT NULL DEFAULT '',
    arxiv_id TEXT NOT NULL DEFAULT '',
    s2_id TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    pdf_path TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    sections TEXT NOT NULL DEFAULT '{}',
    reference_list TEXT NOT NULL DEFAULT '[]',
    n_pages INTEGER NOT NULL DEFAULT 0,
    added_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    section TEXT NOT NULL,
    page INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_paper ON chunks(paper_id);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

_JSON_FIELDS = ("authors", "tags", "sections", "reference_list")


@dataclass
class Paper:
    id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    s2_id: str = ""
    url: str = ""
    pdf_path: str = ""
    source: str = ""
    tags: list[str] = field(default_factory=list)
    # canonical section name -> full section text
    sections: dict[str, str] = field(default_factory=dict)
    reference_list: list[str] = field(default_factory=list)
    n_pages: int = 0
    added_at: str = ""

    def summary(self) -> dict[str, Any]:
        """Compact metadata (no full text) for listings."""
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "url": self.url,
            "tags": self.tags,
            "has_fulltext": bool(self.sections),
            "n_pages": self.n_pages,
        }

    @property
    def cite_key(self) -> str:
        first = self.authors[0].split()[-1] if self.authors else "Anon"
        first = "".join(ch for ch in first if ch.isalnum()) or "Anon"
        return f"{first}{self.year or 'nd'}"


@dataclass
class StoredChunk:
    id: int
    paper_id: str
    idx: int
    section: str
    page: int
    text: str


class Store:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self.version = 0  # bumped on every write so caches can invalidate

    # ------------------------------------------------------------------ meta

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    # ------------------------------------------------------------------ papers

    def upsert_paper(self, paper: Paper) -> None:
        if not paper.added_at:
            paper.added_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        row = asdict(paper)
        for key in _JSON_FIELDS:
            row[key] = json.dumps(row[key])
        cols = ", ".join(row)
        marks = ", ".join(f":{k}" for k in row)
        with self._lock, self._conn:
            self._conn.execute(f"INSERT OR REPLACE INTO papers ({cols}) VALUES ({marks})", row)
        self.version += 1

    def _row_to_paper(self, row: sqlite3.Row) -> Paper:
        data = dict(row)
        for key in _JSON_FIELDS:
            data[key] = json.loads(data[key])
        return Paper(**data)

    def get_paper(self, paper_id: str) -> Paper | None:
        row = self._conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        return self._row_to_paper(row) if row else None

    def find_paper(self, *, doi: str = "", arxiv_id: str = "") -> Paper | None:
        if doi:
            row = self._conn.execute("SELECT * FROM papers WHERE lower(doi) = lower(?)", (doi,)).fetchone()
            if row:
                return self._row_to_paper(row)
        if arxiv_id:
            row = self._conn.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
            if row:
                return self._row_to_paper(row)
        return None

    def list_papers(self) -> list[Paper]:
        rows = self._conn.execute("SELECT * FROM papers ORDER BY added_at DESC, id").fetchall()
        return [self._row_to_paper(r) for r in rows]

    def delete_paper(self, paper_id: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
        self.version += 1
        return cur.rowcount > 0

    # ------------------------------------------------------------------ chunks

    def replace_chunks(
        self, paper_id: str, chunks: list[tuple[int, str, int, str]], embeddings: np.ndarray
    ) -> None:
        """chunks: (idx, section, page, text) tuples aligned with ``embeddings`` rows."""
        embeddings = np.asarray(embeddings, dtype=np.float32)
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM chunks WHERE paper_id = ?", (paper_id,))
            self._conn.executemany(
                "INSERT INTO chunks (paper_id, idx, section, page, text, embedding) VALUES (?, ?, ?, ?, ?, ?)",
                [(paper_id, i, s, p, t, embeddings[n].tobytes()) for n, (i, s, p, t) in enumerate(chunks)],
            )
        self.version += 1

    def iter_chunks(self, with_embeddings: bool = True) -> Iterator[tuple[StoredChunk, np.ndarray | None]]:
        cols = "id, paper_id, idx, section, page, text" + (", embedding" if with_embeddings else "")
        for row in self._conn.execute(f"SELECT {cols} FROM chunks ORDER BY paper_id, idx"):
            chunk = StoredChunk(row["id"], row["paper_id"], row["idx"], row["section"], row["page"], row["text"])
            vector = np.frombuffer(row["embedding"], dtype=np.float32) if with_embeddings else None
            yield chunk, vector

    def chunks_for(self, paper_id: str) -> list[StoredChunk]:
        rows = self._conn.execute(
            "SELECT id, paper_id, idx, section, page, text FROM chunks WHERE paper_id = ? ORDER BY idx", (paper_id,)
        ).fetchall()
        return [StoredChunk(r["id"], r["paper_id"], r["idx"], r["section"], r["page"], r["text"]) for r in rows]

    def count_chunks(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
