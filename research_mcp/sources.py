"""Clients for external scholarly sources: arXiv and Semantic Scholar."""

from __future__ import annotations

import ipaddress
import re
import socket
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from .config import Settings

ARXIV_API = "https://export.arxiv.org/api/query"
S2_GRAPH = "https://api.semanticscholar.org/graph/v1"
S2_RECS = "https://api.semanticscholar.org/recommendations/v1"
S2_FIELDS = "title,authors,year,abstract,venue,externalIds,citationCount,openAccessPdf,url"
S2_EDGE_FIELDS = "title,authors,year,venue,externalIds,citationCount,url,isInfluential,contexts"

_ARXIV_ID_RE = re.compile(r"(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)", re.I)
_DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s\"<>]+)", re.I)
_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class SourceError(RuntimeError):
    pass


@dataclass
class PaperRecord:
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    s2_id: str = ""
    url: str = ""
    pdf_url: str = ""
    citation_count: int | None = None
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, abstract_chars: int = 600) -> dict[str, Any]:
        data = asdict(self)
        if abstract_chars and len(self.abstract) > abstract_chars:
            data["abstract"] = self.abstract[: abstract_chars - 1] + "…"
        return {k: v for k, v in data.items() if v not in ("", None, [], {})}

    @property
    def s2_lookup_id(self) -> str:
        if self.s2_id:
            return self.s2_id
        if self.arxiv_id:
            return f"arXiv:{strip_version(self.arxiv_id)}"
        if self.doi:
            return f"DOI:{self.doi}"
        return ""


def strip_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


def parse_identifier(ref: str) -> tuple[str, str]:
    """Classify a paper reference as ("arxiv", id), ("doi", doi), ("s2", id) or ("url", url)."""
    ref = ref.strip()
    lower = ref.lower()
    if "arxiv.org/" in lower:
        m = re.search(r"arxiv\.org/(?:abs|pdf|html)/([^\s?#]+?)(?:\.pdf)?(?:[?#].*)?$", ref, re.I)
        if m:
            return "arxiv", m.group(1)
    if lower.startswith("doi:") or "doi.org/" in lower or re.match(r"^10\.\d{4,9}/", ref):
        m = _DOI_RE.search(ref)
        if m:
            return "doi", m.group(1).rstrip(".")
    if re.match(r"^https?://", ref, re.I):
        return "url", ref
    if re.fullmatch(r"[0-9a-f]{40}", lower):
        return "s2", ref
    m = _ARXIV_ID_RE.fullmatch(ref)
    if m:
        return "arxiv", m.group(1)
    raise SourceError(f"Unrecognized paper identifier: {ref!r}. Use an arXiv id, DOI, Semantic Scholar id or URL.")


class ScholarlySources:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings = settings
        headers = {"User-Agent": "research-paper-mcp/0.1 (+https://github.com/anuragsamota/research_paper_mcp_server)"}
        self.client = client or httpx.Client(timeout=settings.http_timeout, follow_redirects=True, headers=headers)
        self._s2_headers = {"x-api-key": settings.semantic_scholar_api_key} if settings.semantic_scholar_api_key else {}

    # ------------------------------------------------------------------ arXiv

    def search_arxiv(self, query: str, limit: int = 10, sort: str = "relevance") -> list[PaperRecord]:
        sort_by = {"relevance": "relevance", "recent": "submittedDate"}.get(sort, "relevance")
        q = query if re.search(r"\b(ti|au|abs|cat|all):", query) else f"all:{query}"
        params = {"search_query": q, "start": 0, "max_results": max(1, min(limit, 50)), "sortBy": sort_by}
        return self._arxiv(params)

    def get_arxiv(self, arxiv_id: str) -> PaperRecord:
        records = self._arxiv({"id_list": arxiv_id, "max_results": 1})
        if not records or not records[0].title:
            raise SourceError(f"arXiv paper {arxiv_id} not found")
        return records[0]

    def _arxiv(self, params: dict[str, Any]) -> list[PaperRecord]:
        try:
            response = self.client.get(ARXIV_API, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SourceError(f"arXiv request failed: {exc}") from exc
        root = ET.fromstring(response.text)
        records = []
        for entry in root.findall("a:entry", _ATOM):
            raw_id = (entry.findtext("a:id", "", _ATOM) or "").rsplit("/abs/", 1)[-1]
            title = " ".join((entry.findtext("a:title", "", _ATOM) or "").split())
            if not title or title.lower() == "error":
                continue
            published = entry.findtext("a:published", "", _ATOM) or ""
            pdf_url = ""
            for link in entry.findall("a:link", _ATOM):
                if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                    pdf_url = link.get("href", "")
            records.append(
                PaperRecord(
                    title=title,
                    authors=[a.findtext("a:name", "", _ATOM).strip() for a in entry.findall("a:author", _ATOM)],
                    year=int(published[:4]) if published[:4].isdigit() else None,
                    abstract=" ".join((entry.findtext("a:summary", "", _ATOM) or "").split()),
                    venue=entry.findtext("arxiv:journal_ref", "", _ATOM) or "arXiv",
                    doi=entry.findtext("arxiv:doi", "", _ATOM) or "",
                    arxiv_id=raw_id,
                    url=f"https://arxiv.org/abs/{raw_id}",
                    pdf_url=pdf_url or f"https://arxiv.org/pdf/{raw_id}",
                    source="arxiv",
                    extra={"categories": [c.get("term") for c in entry.findall("a:category", _ATOM)]},
                )
            )
        return records

    # ------------------------------------------------------------------ Semantic Scholar

    def _s2_get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(3):
            try:
                response = self.client.get(url, params=params, headers=self._s2_headers)
            except httpx.HTTPError as exc:
                raise SourceError(f"Semantic Scholar request failed: {exc}") from exc
            if response.status_code == 429 and attempt < 2:
                time.sleep(1.5 * (attempt + 1))  # unauthenticated rate limit is shared; back off briefly
                continue
            if response.status_code == 404:
                raise SourceError("Paper not found on Semantic Scholar")
            if response.status_code >= 400:
                raise SourceError(f"Semantic Scholar returned HTTP {response.status_code}: {response.text[:200]}")
            return response.json()
        raise SourceError("Semantic Scholar rate limit hit; set SEMANTIC_SCHOLAR_API_KEY or retry later")

    @staticmethod
    def _s2_record(data: dict[str, Any]) -> PaperRecord:
        ext = data.get("externalIds") or {}
        pdf = (data.get("openAccessPdf") or {}).get("url") or ""
        arxiv_id = ext.get("ArXiv") or ""
        return PaperRecord(
            title=data.get("title") or "",
            authors=[a.get("name", "") for a in data.get("authors") or []],
            year=data.get("year"),
            abstract=data.get("abstract") or "",
            venue=data.get("venue") or "",
            doi=ext.get("DOI") or "",
            arxiv_id=arxiv_id,
            s2_id=data.get("paperId") or "",
            url=data.get("url") or "",
            pdf_url=pdf or (f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else ""),
            citation_count=data.get("citationCount"),
            source="semantic_scholar",
        )

    def search_semantic_scholar(self, query: str, limit: int = 10, year: str = "") -> list[PaperRecord]:
        params: dict[str, Any] = {"query": query, "limit": max(1, min(limit, 100)), "fields": S2_FIELDS}
        if year:
            params["year"] = year
        data = self._s2_get(f"{S2_GRAPH}/paper/search", params)
        return [self._s2_record(p) for p in data.get("data") or [] if p.get("title")]

    def get_semantic_scholar(self, paper_id: str) -> PaperRecord:
        return self._s2_record(self._s2_get(f"{S2_GRAPH}/paper/{paper_id}", {"fields": S2_FIELDS}))

    def citation_edges(self, paper_id: str, direction: str = "references", limit: int = 25) -> list[PaperRecord]:
        """direction="references": papers this one cites; "citations": papers that cite this one."""
        if direction not in ("references", "citations"):
            raise ValueError("direction must be 'references' or 'citations'")
        data = self._s2_get(
            f"{S2_GRAPH}/paper/{paper_id}/{direction}", {"fields": S2_EDGE_FIELDS, "limit": max(1, min(limit, 100))}
        )
        key = "citedPaper" if direction == "references" else "citingPaper"
        out = []
        for edge in data.get("data") or []:
            paper = edge.get(key) or {}
            if not paper.get("title"):
                continue
            record = self._s2_record(paper)
            record.extra = {
                "influential": bool(edge.get("isInfluential")),
                "contexts": [" ".join(c.split()) for c in (edge.get("contexts") or [])[:2]],
            }
            out.append(record)
        return out

    def recommendations(self, paper_id: str, limit: int = 10) -> list[PaperRecord]:
        data = self._s2_get(
            f"{S2_RECS}/papers/forpaper/{paper_id}", {"fields": S2_FIELDS, "limit": max(1, min(limit, 100))}
        )
        return [self._s2_record(p) for p in data.get("recommendedPapers") or [] if p.get("title")]

    # ------------------------------------------------------------------ resolve + download

    def resolve(self, ref: str) -> PaperRecord:
        kind, value = parse_identifier(ref)
        if kind == "arxiv":
            return self.get_arxiv(value)
        if kind == "doi":
            return self.get_semantic_scholar(f"DOI:{value}")
        if kind == "s2":
            return self.get_semantic_scholar(value)
        return PaperRecord(title="", url=value, pdf_url=value, source="url")

    def download_pdf(self, url: str) -> bytes:
        limit = self.settings.max_pdf_bytes
        chunks: list[bytes] = []
        try:
            # Follow redirects by hand so every hop is checked against private networks.
            for _ in range(6):
                self._check_public(url)
                with self.client.stream("GET", url, follow_redirects=False) as response:
                    if response.is_redirect:
                        url = str(response.url.join(response.headers["location"]))
                        continue
                    response.raise_for_status()
                    total = 0
                    for part in response.iter_bytes():
                        total += len(part)
                        if total > limit:
                            raise SourceError(f"PDF exceeds the {limit // (1024 * 1024)} MB limit")
                        chunks.append(part)
                    break
            else:
                raise SourceError("Too many redirects")
        except httpx.HTTPError as exc:
            raise SourceError(f"Download failed: {exc}") from exc
        data = b"".join(chunks)
        if not data.startswith(b"%PDF"):
            raise SourceError(f"{url} did not return a PDF (it may be a paywall or landing page)")
        return data

    def _check_public(self, url: str) -> None:
        """Refuse URLs that point at loopback/private hosts (SSRF guard) unless explicitly allowed."""
        parsed = httpx.URL(url)
        if parsed.scheme not in ("http", "https"):
            raise SourceError("Only http(s) URLs can be downloaded")
        if self.settings.allow_private_urls:
            return
        try:
            infos = socket.getaddrinfo(parsed.host, parsed.port or 443, proto=socket.IPPROTO_TCP)
        except (socket.gaierror, UnicodeError):
            return  # unresolvable here; the request itself will fail or go via a proxy
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                raise SourceError(f"Refusing to download from a private address ({parsed.host})")
