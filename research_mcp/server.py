"""MCP server: tools, resources and prompts for an LLM research assistant."""

from __future__ import annotations

import json
import threading
from typing import Literal

from mcp.server.fastmcp import FastMCP

from . import generation
from .config import Settings
from .llm import OllamaLLM
from .service import ResearchLibrary

INSTRUCTIONS = """\
You are connected to a research paper library with semantic search.

Typical workflow:
1. `search_papers` to discover papers on arXiv / Semantic Scholar.
2. `add_paper` (arXiv id, DOI or PDF URL) or `ingest_pdf` (local file) to index full text.
3. `semantic_search` to retrieve passages that answer a question. Always ground claims in the
   returned passages and cite them as [cite_key, p. N].
4. `summarize_paper`, `compare_papers`, `find_citations`, `find_related_papers` for analysis.
5. `literature_review` to get a themed review scaffold, then write the review from it.
6. Generation via the server's Ollama model: `ask_papers` (grounded Q&A), `synthesize_comparison`,
   `write_literature_review`. Check `llm_status` if they fail.

Paper ids look like `arxiv-2106.09685`, `doi-10.1145_...` or `pdf-<hash>`; see `list_papers`.
"""

mcp = FastMCP("research-assistant", instructions=INSTRUCTIONS, stateless_http=True, json_response=True)

_library: ResearchLibrary | None = None
_llm: OllamaLLM | None = None
_lock = threading.Lock()


def get_library() -> ResearchLibrary:
    global _library
    with _lock:
        if _library is None:
            _library = ResearchLibrary(Settings.from_env())
        return _library


def set_library(library: ResearchLibrary | None) -> None:
    """Swap the library (used by tests)."""
    global _library
    _library = library


def get_llm() -> OllamaLLM:
    global _llm
    with _lock:
        if _llm is None:
            _llm = OllamaLLM.from_settings(Settings.from_env())
        return _llm


def set_llm(llm: OllamaLLM | None) -> None:
    """Swap the LLM client (used by tests)."""
    global _llm
    _llm = llm


# =========================================================================== discovery & ingestion


@mcp.tool()
def search_papers(
    query: str,
    source: Literal["all", "arxiv", "semantic_scholar"] = "all",
    limit: int = 10,
    year: str = "",
) -> list[dict]:
    """Search arXiv and/or Semantic Scholar for papers.

    Args:
        query: Keywords or a natural-language description. arXiv field syntax (ti:, au:, cat:) also works.
        source: Which index to search.
        limit: Maximum results (1-50).
        year: Semantic Scholar year filter, e.g. "2020-" or "2018-2022".

    Returns metadata (title, authors, year, abstract, citation count) plus `add_with`, the identifier
    to pass to `add_paper`, and `in_library` if the paper is already indexed.
    """
    return get_library().search_external(query, source, max(1, min(limit, 50)), year)


@mcp.tool()
def add_paper(identifier: str, download_pdf: bool = True, tags: list[str] | None = None) -> dict:
    """Add a paper to the library and index it for semantic search.

    Args:
        identifier: arXiv id ("2106.09685"), arXiv URL, DOI ("10.1145/..."), Semantic Scholar id, or a direct PDF URL.
        download_pdf: Fetch and index the open-access full text when available (otherwise abstract only).
        tags: Optional labels for organizing papers.
    """
    paper, status = get_library().add_paper(identifier, download_pdf=download_pdf, tags=tags)
    return {"status": status, **paper.summary(), "cite_key": paper.cite_key, "sections": list(paper.sections)}


@mcp.tool()
def ingest_pdf(path: str, title: str = "", tags: list[str] | None = None) -> dict:
    """Index a local PDF file. Relative paths are resolved against the inbox directory
    (see `library_stats`); absolute paths must be inside RESEARCH_ALLOWED_DIRS.
    """
    paper = get_library().ingest_local_pdf(path, title, tags)
    return {"status": "ingested", **paper.summary(), "cite_key": paper.cite_key, "sections": list(paper.sections)}


@mcp.tool()
def list_papers(tag: str = "") -> list[dict]:
    """List papers in the library (optionally only those with a tag)."""
    papers = get_library().store.list_papers()
    return [{**p.summary(), "cite_key": p.cite_key} for p in papers if not tag or tag in p.tags]


@mcp.tool()
def get_paper(paper_id: str, include_sections: list[str] | None = None) -> dict:
    """Get a paper's metadata, abstract and (optionally) the full text of chosen sections.

    Args:
        paper_id: Library id.
        include_sections: Section names to return in full, e.g. ["method", "results"].
            Available names are listed in `sections`.
    """
    paper = get_library().get(paper_id)
    data = {**paper.summary(), "cite_key": paper.cite_key, "abstract": paper.abstract, "sections": list(paper.sections)}
    if include_sections:
        data["section_text"] = {s: paper.sections.get(s, "") for s in include_sections}
    return data


@mcp.tool()
def remove_paper(paper_id: str) -> str:
    """Delete a paper and its index entries from the library."""
    get_library().remove(paper_id)
    return f"Removed {paper_id}"


# =========================================================================== retrieval


@mcp.tool()
def semantic_search(
    query: str,
    top_k: int = 8,
    paper_ids: list[str] | None = None,
    sections: list[str] | None = None,
) -> list[dict]:
    """Retrieve the passages most relevant to a question from indexed papers (hybrid dense + BM25 search).

    Args:
        query: A question or description of the information needed.
        top_k: Number of passages (1-30).
        paper_ids: Restrict to these papers.
        sections: Restrict to sections, e.g. ["method", "results", "conclusion"].

    Each passage carries a `citation` like [Hu2021, p. 4] to use when quoting it.
    """
    return get_library().semantic_search(query, max(1, min(top_k, 30)), paper_ids, sections)


# =========================================================================== analysis


@mcp.tool()
def summarize_paper(paper_id: str) -> dict:
    """Structured summary: key points, keywords, per-section highlights and stated limitations
    (extractive, so every sentence is quoted from the paper)."""
    return get_library().summarize(paper_id)


@mcp.tool()
def compare_papers(paper_ids: list[str], aspects: list[str] | None = None) -> dict:
    """Compare two or more papers side by side.

    Args:
        paper_ids: Papers to compare.
        aspects: Any of problem, method, data, results, limitations, contributions, or free-text aspects
            (e.g. "computational cost"). Defaults to problem/method/data/results/limitations.

    Returns, per aspect, the most relevant evidence passage from each paper, plus shared and
    distinctive keywords and a pairwise semantic similarity matrix.
    """
    return get_library().compare(paper_ids, aspects)


@mcp.tool()
def find_citations(
    paper_id: str, direction: Literal["references", "citations"] = "references", limit: int = 20
) -> dict:
    """Citation discovery.

    direction="references": works this paper cites. direction="citations": later works that cite it.
    Uses Semantic Scholar (influential citations first, with citation contexts) and falls back
    to the reference list parsed from the PDF. Results include `add_with` to pull a paper in.
    """
    return get_library().citations(paper_id, direction, max(1, min(limit, 100)))


@mcp.tool()
def extract_references(paper_id: str) -> dict:
    """Return the reference list parsed from a paper's PDF and which references are already in the library."""
    return get_library().extract_references(paper_id)


@mcp.tool()
def find_related_papers(paper_id: str, limit: int = 8, include_external: bool = True) -> dict:
    """Papers similar to this one: from the library (embedding similarity) and Semantic Scholar recommendations."""
    return get_library().related(paper_id, max(1, min(limit, 30)), include_external)


@mcp.tool()
def literature_review(
    topic: str,
    paper_ids: list[str] | None = None,
    fetch_new: int = 0,
    n_themes: int = 0,
    min_year: int = 0,
) -> dict:
    """Build an automated literature review scaffold on a topic.

    Selects relevant library papers (or `paper_ids`), clusters them into themes, extracts each paper's
    key contribution, a timeline, research gaps from limitation statements, supporting evidence passages,
    a Markdown draft and BibTeX.

    Args:
        topic: Review topic or research question.
        paper_ids: Use exactly these papers instead of automatic selection.
        fetch_new: First search arXiv/Semantic Scholar and add up to this many new papers (0-10).
        n_themes: Number of themes (0 = automatic).
        min_year: Ignore papers published before this year (0 = no filter).
    """
    return get_library().literature_review(
        topic, paper_ids, fetch_new=max(0, min(fetch_new, 10)), n_themes=max(0, n_themes), min_year=min_year or None
    )


@mcp.tool()
def export_bibtex(paper_ids: list[str] | None = None) -> str:
    """BibTeX for the given papers (or the whole library)."""
    return get_library().bibtex(paper_ids)


@mcp.tool()
def library_stats() -> dict:
    """Library size, the active embedding model and where files are stored."""
    return get_library().stats()


@mcp.tool()
def reindex_library() -> str:
    """Re-embed all chunks with the current embedding model (after changing RESEARCH_EMBEDDER)."""
    n = get_library().reindex()
    return f"Re-embedded {n} chunks with {get_library().embedder.name}"


# =========================================================================== LLM generation (Ollama)


@mcp.tool()
def ask_papers(question: str, paper_ids: list[str] | None = None, top_k: int = 8) -> dict:
    """Answer a research question with the server's Ollama LLM, grounded in retrieved passages.

    Retrieves the most relevant passages (optionally from `paper_ids` only), has the model answer
    using only those passages with [cite_key, p. N] citations, and returns the answer plus its sources.
    A `warning` is added if the model cites anything that was not among the sources.
    """
    return generation.ask(get_library(), get_llm(), question, paper_ids, max(1, min(top_k, 20)))


@mcp.tool()
def synthesize_comparison(paper_ids: list[str], aspects: list[str] | None = None) -> dict:
    """Have the Ollama LLM write a comparison table and trade-off analysis from `compare_papers` evidence."""
    return generation.synthesize_comparison(get_library(), get_llm(), paper_ids, aspects)


@mcp.tool()
def write_literature_review(
    topic: str,
    paper_ids: list[str] | None = None,
    fetch_new: int = 0,
    n_themes: int = 0,
    min_year: int = 0,
) -> dict:
    """Write a complete literature review (Markdown + BibTeX) with the Ollama LLM.

    Builds the `literature_review` scaffold, then writes the introduction, one section per theme from
    retrieved evidence, open problems and conclusion. Slow on local models (one LLM call per section).
    """
    return generation.write_literature_review(
        get_library(), get_llm(), topic, paper_ids,
        fetch_new=max(0, min(fetch_new, 10)), n_themes=max(0, n_themes), min_year=min_year or None,
    )


@mcp.tool()
def llm_status() -> dict:
    """Check the Ollama connection: is the server reachable and is the configured model installed?"""
    return get_llm().health()


# =========================================================================== resources


@mcp.resource("papers://library", mime_type="application/json")
def library_resource() -> str:
    """All papers in the library (metadata only)."""
    return json.dumps([p.summary() for p in get_library().store.list_papers()], indent=2)


@mcp.resource("papers://{paper_id}", mime_type="application/json")
def paper_resource(paper_id: str) -> str:
    """Metadata and abstract of one paper."""
    paper = get_library().get(paper_id)
    return json.dumps({**paper.summary(), "abstract": paper.abstract, "sections": list(paper.sections)}, indent=2)


@mcp.resource("papers://{paper_id}/fulltext", mime_type="text/markdown")
def fulltext_resource(paper_id: str) -> str:
    """Full extracted text of a paper, organized by section."""
    paper = get_library().get(paper_id)
    parts = [f"# {paper.title}", ""]
    for name, text in paper.sections.items():
        if name != "references":
            parts += [f"## {name.replace('_', ' ').title()}", "", text, ""]
    if not paper.sections:
        parts += ["## Abstract", "", paper.abstract]
    return "\n".join(parts)


@mcp.resource("papers://{paper_id}/bibtex", mime_type="application/x-bibtex")
def bibtex_resource(paper_id: str) -> str:
    """BibTeX entry for one paper."""
    return get_library().bibtex([paper_id])


# =========================================================================== prompts


@mcp.prompt()
def review_literature(topic: str) -> str:
    """Write a literature review on a topic using the library."""
    return f"""Write a literature review on: "{topic}".

1. Call `literature_review` with topic="{topic}" (use fetch_new=5 if the library has few relevant papers).
2. For each theme, call `semantic_search` with focused questions to gather specific evidence.
3. Write the review with sections: Introduction, one section per theme (compare approaches, don't just list
   papers), Open Problems, Conclusion.
4. Cite every claim as [cite_key] or [cite_key, p. N] using only papers returned by the tools.
5. End with the BibTeX from the tool output."""


@mcp.prompt()
def critique_paper(paper_id: str) -> str:
    """Critically appraise a single paper."""
    return f"""Critically appraise paper `{paper_id}`.

Call `summarize_paper`, then `semantic_search` restricted to this paper for its method, evaluation and
limitations, and `find_related_papers` for context. Report: contribution and novelty, soundness of method,
strength of evidence, limitations (stated and unstated), and how it relates to prior work.
Quote passages with [cite_key, p. N]."""


@mcp.prompt()
def compare_approaches(paper_ids: str, focus: str = "") -> str:
    """Compare papers (comma-separated ids) and produce a comparison table."""
    ids = [p.strip() for p in paper_ids.split(",") if p.strip()]
    focus_line = f' Pay special attention to "{focus}".' if focus else ""
    return f"""Compare these papers: {", ".join(ids)}.{focus_line}

Call `compare_papers` with paper_ids={ids}{f' and include "{focus}" in aspects' if focus else ''}. Produce a
Markdown table (rows = aspects, columns = papers) followed by a short analysis of trade-offs and which
approach suits which setting. Base every cell on the evidence passages and cite pages."""


@mcp.prompt()
def answer_question(question: str) -> str:
    """Answer a research question grounded in the indexed papers."""
    return f"""Answer the question: "{question}"

Use `semantic_search` (try 2-3 phrasings) to retrieve evidence. If coverage is thin, use `search_papers` and
`add_paper` to bring in relevant papers, then search again. Answer only from retrieved passages, cite each
claim as [cite_key, p. N], and say clearly what the literature does not settle."""
