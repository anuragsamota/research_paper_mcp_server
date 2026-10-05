"""The research library: ingestion pipeline plus every high-level operation the MCP tools expose."""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from . import analysis
from .config import Settings
from .embeddings import Embedder, create_embedder
from .pdf import chunk_sections, guess_title, parse_pdf
from .retrieval import VectorIndex
from .sources import PaperRecord, ScholarlySources, SourceError, parse_identifier, strip_version
from .store import Paper, Store
from .text import truncate

log = logging.getLogger(__name__)


class LibraryError(ValueError):
    """A user-facing error (bad id, missing paper, ...)."""


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9.\-]+", "_", value).strip("_")


def make_paper_id(record: PaperRecord | None, pdf_bytes: bytes | None) -> str:
    if record and record.arxiv_id:
        return "arxiv-" + _slug(strip_version(record.arxiv_id))
    if record and record.doi:
        return "doi-" + _slug(record.doi.lower())
    if record and record.s2_id:
        return "s2-" + record.s2_id[:16]
    if pdf_bytes:
        return "pdf-" + hashlib.sha1(pdf_bytes).hexdigest()[:12]
    seed = (record.title if record else "") or "untitled"
    return "paper-" + hashlib.sha1(seed.lower().encode()).hexdigest()[:12]


class ResearchLibrary:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder | None = None,
        sources: ScholarlySources | None = None,
    ):
        settings.ensure_dirs()
        self.settings = settings
        self.store = Store(settings.db_path)
        self.embedder = embedder or create_embedder(settings)
        self.sources = sources or ScholarlySources(settings)
        self.index = VectorIndex(self.store, self.embedder)
        previous = self.store.get_meta("embedder")
        if previous and previous != self.embedder.name:
            log.warning("Library was embedded with %s, now using %s: re-indexing", previous, self.embedder.name)
            self.reindex()
        self.store.set_meta("embedder", self.embedder.name)

    # ================================================================== helpers

    def get(self, paper_id: str) -> Paper:
        paper = self.store.get_paper(paper_id.strip())
        if paper is None:
            known = ", ".join(p.id for p in self.store.list_papers()[:15]) or "the library is empty"
            raise LibraryError(f"No paper with id {paper_id!r}. Known ids: {known}")
        return paper

    def _corpus_df(self) -> tuple[Counter[str], int]:
        papers = self.store.list_papers()
        return analysis.document_frequencies(self._analysis_text(p) for p in papers)

    @staticmethod
    def _analysis_text(paper: Paper) -> str:
        if paper.sections:
            return "\n".join(t for n, t in paper.sections.items() if n not in ("references", "front", "acknowledgments"))
        return f"{paper.title}. {paper.abstract}"

    def _cite(self, paper: Paper, page: int | None = None) -> str:
        loc = f", p. {page}" if page else ""
        return f"[{paper.cite_key}{loc}]"

    # ================================================================== ingestion

    def ingest_pdf_bytes(
        self, data: bytes, record: PaperRecord | None = None, *, source: str = "upload", tags: list[str] | None = None
    ) -> Paper:
        parsed = parse_pdf(data)
        paper_id = make_paper_id(record, data)
        record = record or PaperRecord(title="")
        pdf_path = self.settings.pdf_dir / f"{paper_id}.pdf"
        pdf_path.write_bytes(data)
        sections: dict[str, str] = {}
        for section in parsed.sections:
            sections[section.name] = (sections.get(section.name, "") + "\n" + section.text).strip()
        references = analysis.parse_references(sections.get("references", ""))
        abstract = record.abstract or sections.get("abstract", "")
        if not abstract:
            # No explicit abstract heading: the first ~120 words after the front matter.
            abstract = truncate(sections.get("front", "") or parsed.full_text, 900)
        paper = Paper(
            id=paper_id,
            title=record.title or guess_title(parsed),
            authors=record.authors or _split_authors(parsed.metadata.get("author", "")),
            year=record.year,
            abstract=abstract,
            venue=record.venue,
            doi=record.doi,
            arxiv_id=record.arxiv_id,
            s2_id=record.s2_id,
            url=record.url,
            pdf_path=str(pdf_path),
            source=record.source or source,
            tags=tags or [],
            sections=sections,
            reference_list=references,
            n_pages=len(parsed.pages),
        )
        if not paper.year:
            m = re.search(r"\b(19[89]\d|20[0-4]\d)\b", parsed.pages[0].text)
            paper.year = int(m.group(1)) if m else None
        chunks = chunk_sections(parsed, self.settings.chunk_size, self.settings.chunk_overlap)
        self._save_with_chunks(paper, [(c.index, c.section, c.page, c.text) for c in chunks])
        return paper

    def _save_with_chunks(self, paper: Paper, chunks: list[tuple[int, str, int, str]]) -> None:
        if not chunks:
            chunks = [(0, "abstract", 1, f"{paper.title}. {paper.abstract}".strip())]
        # Prefix each chunk with the title so embeddings carry document context.
        vectors = self.embedder.embed([f"{paper.title}. {text}" for _, _, _, text in chunks])
        self.store.upsert_paper(paper)
        self.store.replace_chunks(paper.id, chunks, vectors)

    def add_metadata_only(self, record: PaperRecord, tags: list[str] | None = None) -> Paper:
        paper = Paper(
            id=make_paper_id(record, None),
            title=record.title,
            authors=record.authors,
            year=record.year,
            abstract=record.abstract,
            venue=record.venue,
            doi=record.doi,
            arxiv_id=record.arxiv_id,
            s2_id=record.s2_id,
            url=record.url,
            source=record.source,
            tags=tags or [],
        )
        self._save_with_chunks(paper, [])
        return paper

    def add_paper(self, ref: str, *, download_pdf: bool = True, tags: list[str] | None = None) -> tuple[Paper, str]:
        """Add a paper by arXiv id / DOI / Semantic Scholar id / PDF URL. Returns (paper, status message)."""
        kind, value = parse_identifier(ref)
        existing = self.store.find_paper(
            doi=value if kind == "doi" else "", arxiv_id=value if kind == "arxiv" else ""
        ) or (self.store.get_paper("arxiv-" + _slug(strip_version(value))) if kind == "arxiv" else None)
        if existing and (existing.sections or not download_pdf):
            return existing, "already in library"

        record = self.sources.resolve(ref)
        if kind == "url":
            data = self.sources.download_pdf(value)
            return self.ingest_pdf_bytes(data, None, source="url", tags=tags), "ingested PDF from URL"
        if download_pdf and record.pdf_url:
            try:
                data = self.sources.download_pdf(record.pdf_url)
                return self.ingest_pdf_bytes(data, record, tags=tags), "ingested full text"
            except (SourceError, ValueError) as exc:
                paper = self.add_metadata_only(record, tags)
                return paper, f"added metadata + abstract only (PDF unavailable: {exc})"
        paper = self.add_metadata_only(record, tags)
        note = "no open-access PDF found" if download_pdf else "PDF download skipped"
        return paper, f"added metadata + abstract only ({note})"

    def ingest_local_pdf(self, path: str, title: str = "", tags: list[str] | None = None) -> Paper:
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = self.settings.inbox_dir / resolved
        resolved = resolved.resolve()
        if not any(resolved.is_relative_to(d) for d in self.settings.allowed_dirs):
            allowed = ", ".join(str(d) for d in self.settings.allowed_dirs)
            raise LibraryError(f"{resolved} is outside the allowed directories ({allowed}); set RESEARCH_ALLOWED_DIRS")
        if not resolved.is_file():
            raise LibraryError(f"File not found: {resolved}")
        if resolved.stat().st_size > self.settings.max_pdf_bytes:
            raise LibraryError("PDF is larger than RESEARCH_MAX_PDF_MB")
        record = PaperRecord(title=title, source="local") if title else None
        return self.ingest_pdf_bytes(resolved.read_bytes(), record, source="local", tags=tags)

    def remove(self, paper_id: str) -> bool:
        paper = self.get(paper_id)
        if paper.pdf_path:
            Path(paper.pdf_path).unlink(missing_ok=True)
        return self.store.delete_paper(paper.id)

    def reindex(self) -> int:
        """Re-embed every stored chunk with the current embedder."""
        n = 0
        for paper in self.store.list_papers():
            chunks = [(c.idx, c.section, c.page, c.text) for c in self.store.chunks_for(paper.id)]
            self._save_with_chunks(paper, chunks)
            n += len(chunks)
        self.store.set_meta("embedder", self.embedder.name)
        return n

    # ================================================================== search

    def search_external(self, query: str, source: str = "all", limit: int = 10, year: str = "") -> list[dict]:
        results: list[PaperRecord] = []
        errors = []
        if source in ("all", "arxiv"):
            try:
                results.extend(self.sources.search_arxiv(query, limit))
            except SourceError as exc:
                errors.append(str(exc))
        if source in ("all", "semantic_scholar"):
            try:
                results.extend(self.sources.search_semantic_scholar(query, limit, year))
            except SourceError as exc:
                errors.append(str(exc))
        if not results and errors:
            raise LibraryError("; ".join(errors))
        # De-duplicate by arXiv id / DOI / normalized title, merging citation counts.
        merged: dict[str, PaperRecord] = {}
        for r in results:
            key = strip_version(r.arxiv_id) or r.doi.lower() or re.sub(r"\W+", "", r.title.lower())
            if key in merged:
                kept = merged[key]
                kept.citation_count = kept.citation_count or r.citation_count
                kept.s2_id = kept.s2_id or r.s2_id
                kept.doi = kept.doi or r.doi
            else:
                merged[key] = r
        records = list(merged.values())
        # Re-rank by semantic similarity of title+abstract to the query (mixed sources have no shared score).
        if records:
            q = self.embedder.embed([query])[0]
            docs = self.embedder.embed([f"{r.title}. {r.abstract}" for r in records])
            order = np.argsort(-(docs @ q), kind="stable")
            records = [records[i] for i in order]
        out = []
        for r in records[:limit]:
            item = r.to_dict()
            item["add_with"] = r.arxiv_id or (f"doi:{r.doi}" if r.doi else r.s2_id)
            in_lib = self.store.find_paper(doi=r.doi, arxiv_id=r.arxiv_id) if (r.doi or r.arxiv_id) else None
            item["in_library"] = in_lib.id if in_lib else None
            out.append(item)
        return out

    def semantic_search(
        self, query: str, top_k: int = 8, paper_ids: list[str] | None = None, sections: list[str] | None = None
    ) -> list[dict]:
        hits = self.index.search(query, top_k=top_k, paper_ids=paper_ids, sections=sections)
        papers = {p.id: p for p in self.store.list_papers()}
        out = []
        for h in hits:
            paper = papers.get(h.chunk.paper_id)
            if not paper:
                continue
            out.append(
                {
                    "paper_id": paper.id,
                    "title": paper.title,
                    "year": paper.year,
                    "section": h.chunk.section,
                    "page": h.chunk.page,
                    "citation": self._cite(paper, h.chunk.page),
                    "relevance": round(h.score * 100, 2),
                    "similarity": round(h.dense, 3),
                    "text": h.chunk.text,
                }
            )
        return out

    # ================================================================== analysis

    def summarize(self, paper_id: str) -> dict:
        paper = self.get(paper_id)
        df, n = self._corpus_df()
        summary = analysis.summarize_paper(paper, self.embedder, df, n)
        summary["cite_key"] = paper.cite_key
        return summary

    def compare(self, paper_ids: list[str], aspects: list[str] | None = None) -> dict:
        if len(paper_ids) < 2:
            raise LibraryError("Give at least two paper ids to compare.")
        papers = [self.get(pid) for pid in paper_ids]
        aspects = aspects or ["problem", "method", "data", "results", "limitations"]
        df, n = self._corpus_df()
        keyword_sets = {p.id: analysis.keywords(self._analysis_text(p), df, n, 20) for p in papers}
        all_terms = Counter(t for terms in keyword_sets.values() for t in set(terms))
        shared = [t for t, c in all_terms.most_common() if c >= 2][:15]

        table: dict[str, dict[str, Any]] = {}
        for aspect in aspects:
            query = analysis.ASPECT_QUERIES.get(aspect, aspect)
            row = {}
            for p in papers:
                hits = self.index.search(query, top_k=2, paper_ids=[p.id], sections=analysis.ASPECT_SECTIONS.get(aspect))
                if not hits:
                    hits = self.index.search(query, top_k=2, paper_ids=[p.id])
                row[p.id] = [
                    {"page": h.chunk.page, "section": h.chunk.section, "evidence": truncate(h.chunk.text, 700)}
                    for h in hits
                ]
            table[aspect] = row

        vectors = self.index.paper_vectors([p.id for p in papers])
        return {
            "papers": [
                {
                    **p.summary(),
                    "cite_key": p.cite_key,
                    "distinctive_keywords": [t for t in keyword_sets[p.id] if all_terms[t] == 1][:10],
                }
                for p in papers
            ],
            "shared_keywords": shared,
            "similarity": analysis.similarity_matrix(vectors),
            "aspects": table,
            "guidance": "For each aspect, contrast the evidence passages across papers; cite as [cite_key, p. N].",
        }

    def extract_references(self, paper_id: str) -> dict:
        paper = self.get(paper_id)
        library = self.store.list_papers()
        return {
            "paper_id": paper.id,
            "count": len(paper.reference_list),
            "references": [
                {"n": i + 1, "text": ref, "title_guess": analysis.reference_title_guess(ref)}
                for i, ref in enumerate(paper.reference_list)
            ],
            "in_library": analysis.match_references_to_library(paper, library),
        }

    def citations(self, paper_id: str, direction: str = "references", limit: int = 20) -> dict:
        """Citation discovery via Semantic Scholar; falls back to the PDF's own reference list."""
        paper = self.get(paper_id)
        lookup = paper.s2_id or (f"arXiv:{paper.arxiv_id}" if paper.arxiv_id else (f"DOI:{paper.doi}" if paper.doi else ""))
        if not lookup:
            try:
                found = self.sources.search_semantic_scholar(paper.title, 1)
                if found and analysis.title_similarity(found[0].title, paper.title) > 0.85:
                    lookup = found[0].s2_id
            except SourceError:
                pass
        result: dict[str, Any] = {"paper_id": paper.id, "title": paper.title, "direction": direction}
        if lookup:
            try:
                edges = self.sources.citation_edges(lookup, direction, limit)
                items = []
                for r in edges:
                    item = r.to_dict(300)
                    local = self.store.find_paper(doi=r.doi, arxiv_id=strip_version(r.arxiv_id)) if (r.doi or r.arxiv_id) else None
                    item["in_library"] = local.id if local else None
                    item["add_with"] = r.arxiv_id or (f"doi:{r.doi}" if r.doi else r.s2_id)
                    items.append(item)
                items.sort(key=lambda x: (not x.get("extra", {}).get("influential"), -(x.get("citation_count") or 0)))
                result.update(source="semantic_scholar", count=len(items), papers=items)
                return result
            except SourceError as exc:
                result["warning"] = str(exc)
        if direction == "references" and paper.reference_list:
            result.update(source="pdf_reference_list", **self.extract_references(paper.id))
        else:
            result.setdefault("warning", "Paper not found on Semantic Scholar")
            result["papers"] = []
        return result

    def related(self, paper_id: str, limit: int = 8, include_external: bool = True) -> dict:
        paper = self.get(paper_id)
        vectors = self.index.paper_vectors()
        target = vectors.get(paper.id)
        local = []
        if target is not None:
            scored = sorted(((float(v @ target), pid) for pid, v in vectors.items() if pid != paper.id), reverse=True)
            for score, pid in scored[:limit]:
                other = self.store.get_paper(pid)
                if other:
                    local.append({"paper_id": pid, "title": other.title, "year": other.year, "similarity": round(score, 3)})
        result: dict[str, Any] = {"paper_id": paper.id, "title": paper.title, "in_library": local}
        if include_external:
            lookup = paper.s2_id or (f"arXiv:{paper.arxiv_id}" if paper.arxiv_id else (f"DOI:{paper.doi}" if paper.doi else ""))
            try:
                recs = self.sources.recommendations(lookup, limit) if lookup else self.sources.search_semantic_scholar(paper.title, limit + 1)
                result["external"] = [
                    {**r.to_dict(300), "add_with": r.arxiv_id or (f"doi:{r.doi}" if r.doi else r.s2_id)}
                    for r in recs
                    if analysis.title_similarity(r.title, paper.title) < 0.9
                ][:limit]
            except SourceError as exc:
                result["external_warning"] = str(exc)
        return result

    # ================================================================== literature review

    def literature_review(
        self,
        topic: str,
        paper_ids: list[str] | None = None,
        *,
        fetch_new: int = 0,
        n_themes: int = 0,
        min_year: int | None = None,
    ) -> dict:
        fetched: list[str] = []
        if fetch_new > 0:
            for item in self.search_external(topic, "all", fetch_new * 2):
                if len(fetched) >= fetch_new:
                    break
                if item.get("in_library") or not item.get("add_with"):
                    continue
                try:
                    paper, _ = self.add_paper(item["add_with"], tags=[f"review:{topic[:40]}"])
                    fetched.append(paper.id)
                except (SourceError, LibraryError, ValueError) as exc:
                    log.info("skipping %s: %s", item.get("title"), exc)

        if paper_ids:
            papers = [self.get(pid) for pid in paper_ids]
        else:
            # Pick the library papers most relevant to the topic.
            ranking = self.index.search(topic, top_k=60, per_paper_limit=1)
            ids = [h.chunk.paper_id for h in ranking if h.dense > 0.05 or h.lexical > 0]
            papers = [p for pid in ids if (p := self.store.get_paper(pid))]
            papers = papers[:25]
        if min_year:
            papers = [p for p in papers if not p.year or p.year >= min_year]
        if not papers:
            raise LibraryError("No relevant papers in the library. Add papers first or pass fetch_new > 0.")

        df, n = self._corpus_df()
        keys = analysis.unique_cite_keys(papers)
        vectors = self.index.paper_vectors([p.id for p in papers])
        ordered = [p for p in papers if p.id in vectors]
        k = n_themes or analysis.default_theme_count(len(ordered))
        labels = analysis.kmeans(np.vstack([vectors[p.id] for p in ordered]), k) if ordered else np.array([])

        themes = []
        for c in sorted(set(labels.tolist())):
            members = [p for p, lab in zip(ordered, labels) if lab == c]
            label_terms = analysis.keywords(" ".join(self._analysis_text(p) for p in members), df, n, 5)
            entries = []
            for p in sorted(members, key=lambda x: x.year or 0):
                contribution = analysis.key_sentences(p.abstract or self._analysis_text(p)[:4000], self.embedder, 1)
                entries.append(
                    {
                        "paper_id": p.id,
                        "cite_key": keys[p.id],
                        "title": p.title,
                        "year": p.year,
                        "contribution": contribution[0] if contribution else truncate(p.abstract, 300),
                    }
                )
            themes.append({"theme": ", ".join(label_terms[:3]) or f"Theme {c + 1}", "keywords": label_terms, "papers": entries})

        evidence = [
            {**hit, "citation": f"[{keys.get(hit['paper_id'], hit['paper_id'])}, p. {hit['page']}]"}
            for hit in self.semantic_search(topic, top_k=12, paper_ids=[p.id for p in papers])
        ]
        gaps = []
        for p in papers:
            text = " ".join(p.sections.get(s, "") for s in ("limitations", "discussion", "conclusion")) if p.sections else ""
            for sentence in analysis.limitation_sentences(text, 2):
                gaps.append({"cite_key": keys[p.id], "text": sentence})
        timeline = sorted(({"year": p.year, "cite_key": keys[p.id], "title": p.title} for p in papers if p.year), key=lambda x: x["year"])

        draft = _review_markdown(topic, themes, gaps, timeline)
        bib = "\n\n".join(analysis.bibtex_entry(p, keys[p.id]) for p in papers)
        return {
            "topic": topic,
            "n_papers": len(papers),
            "newly_fetched": fetched,
            "themes": themes,
            "timeline": timeline,
            "research_gaps": gaps[:15],
            "evidence": evidence,
            "draft_markdown": draft,
            "bibtex": bib,
            "guidance": (
                "Use the themes as section structure, the evidence passages for specific claims, and research_gaps "
                "for the open-problems section. Cite with the given cite keys; don't invent sources."
            ),
        }

    def bibtex(self, paper_ids: list[str] | None = None) -> str:
        papers = [self.get(pid) for pid in paper_ids] if paper_ids else self.store.list_papers()
        keys = analysis.unique_cite_keys(papers)
        return "\n\n".join(analysis.bibtex_entry(p, keys[p.id]) for p in papers)

    def stats(self) -> dict:
        papers = self.store.list_papers()
        return {
            "papers": len(papers),
            "with_fulltext": sum(1 for p in papers if p.sections),
            "chunks": self.store.count_chunks(),
            "embedder": self.embedder.name,
            "embedding_dim": self.embedder.dim,
            "data_dir": str(self.settings.data_dir),
            "inbox_dir": str(self.settings.inbox_dir),
        }


def _split_authors(raw: str) -> list[str]:
    if not raw:
        return []
    parts = re.split(r";|,| and ", raw)
    return [p.strip() for p in parts if len(p.strip()) > 2][:20]


def _review_markdown(topic: str, themes: list[dict], gaps: list[dict], timeline: list[dict]) -> str:
    lines = [f"# Literature review: {topic}", ""]
    if timeline:
        lines += [
            f"*{len({t['cite_key'] for t in timeline})} dated papers, {timeline[0]['year']}–{timeline[-1]['year']}.*",
            "",
        ]
    lines += ["## Overview", "", "_(Synthesize the themes below into 1–2 paragraphs.)_", ""]
    for i, theme in enumerate(themes, 1):
        lines += [f"## {i}. {theme['theme'].title()}", ""]
        for entry in theme["papers"]:
            year = f" ({entry['year']})" if entry["year"] else ""
            lines.append(f"- **{entry['title']}**{year} [{entry['cite_key']}]: {entry['contribution']}")
        lines.append("")
    if gaps:
        lines += ["## Open problems and research gaps", ""]
        lines += [f"- {g['text']} [{g['cite_key']}]" for g in gaps[:10]]
        lines.append("")
    lines += ["## References", "", "_(See the BibTeX output.)_"]
    return "\n".join(lines)
