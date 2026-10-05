"""LLM-backed generation on top of retrieval: grounded Q&A, comparison synthesis and full literature reviews.

Prompts are kept small and section-by-section so they fit the context windows of local 7-8B models.
"""

from __future__ import annotations

import re
from typing import Any

from .llm import OllamaLLM
from .service import LibraryError, ResearchLibrary
from .text import truncate

GROUNDED_SYSTEM = """\
You are a careful research assistant. Answer ONLY from the numbered source passages provided.
Cite every factual claim with the source's citation tag exactly as given, e.g. [Hu2021, p. 4].
If the sources do not contain the answer, say so plainly. Do not invent papers, numbers or citations."""

WRITER_SYSTEM = """\
You are an expert academic writer producing a literature review. Write fluent, analytical prose that
compares and connects works rather than listing them. Use ONLY the information provided and cite with
the given tags, e.g. [Li2021] or [Li2021, p. 3]. Never invent citations. Output Markdown without a title."""

_CITE_RE = re.compile(r"\[([A-Z][A-Za-z]+\d{4}[a-z]?|[A-Z][A-Za-z]+nd)(?:,\s*p\.\s*\d+)?\]")


def _sources_block(passages: list[dict], max_chars: int) -> str:
    lines, used = [], 0
    for i, p in enumerate(passages, 1):
        entry = f"({i}) {p['citation']} {p['title']} [{p['section']}]\n{p['text']}\n"
        if used + len(entry) > max_chars and lines:
            break
        lines.append(entry)
        used += len(entry)
    return "\n".join(lines)


def _check_citations(text: str, allowed_keys: set[str]) -> list[str]:
    """Citation keys in ``text`` that do not belong to any provided source (likely hallucinated)."""
    return sorted({m.group(1) for m in _CITE_RE.finditer(text)} - allowed_keys)


def _key(citation: str) -> str:
    return citation.strip("[]").split(",")[0].strip()


def ask(
    library: ResearchLibrary,
    llm: OllamaLLM,
    question: str,
    paper_ids: list[str] | None = None,
    top_k: int = 8,
    max_context_chars: int = 12000,
) -> dict[str, Any]:
    passages = library.semantic_search(question, top_k=top_k, paper_ids=paper_ids)
    if not passages:
        raise LibraryError("No indexed passages to answer from. Add papers first (add_paper / ingest_pdf).")
    prompt = (
        f"Sources:\n\n{_sources_block(passages, max_context_chars)}\n\n"
        f"Question: {question}\n\n"
        "Write a concise, well-structured answer with citations. Finish with one sentence on what the sources leave open."
    )
    answer = llm.complete(prompt, GROUNDED_SYSTEM)
    keys = {_key(p["citation"]) for p in passages}
    result: dict[str, Any] = {
        "answer": answer,
        "model": llm.name,
        "sources": [
            {"citation": p["citation"], "paper_id": p["paper_id"], "title": p["title"], "section": p["section"],
             "page": p["page"], "excerpt": truncate(p["text"], 300)}
            for p in passages
        ],
    }
    unknown = _check_citations(answer, keys)
    if unknown:
        result["warning"] = f"The model cited keys not among the sources: {', '.join(unknown)}. Verify before use."
    return result


def synthesize_comparison(
    library: ResearchLibrary, llm: OllamaLLM, paper_ids: list[str], aspects: list[str] | None = None
) -> dict[str, Any]:
    data = library.compare(paper_ids, aspects)
    keys = {p["id"]: p["cite_key"] for p in data["papers"]}
    blocks = []
    for aspect, row in data["aspects"].items():
        lines = [f"### {aspect}"]
        for pid, evidence in row.items():
            text = " … ".join(truncate(e["evidence"], 450) for e in evidence[:1]) or "(no relevant passage found)"
            page = f", p. {evidence[0]['page']}" if evidence else ""
            lines.append(f"- [{keys[pid]}{page}]: {text}")
        blocks.append("\n".join(lines))
    papers = "\n".join(f"- [{p['cite_key']}] {p['title']} ({p.get('year') or 'n.d.'})" for p in data["papers"])
    prompt = (
        f"Papers:\n{papers}\n\nEvidence by aspect:\n\n" + "\n\n".join(blocks) + "\n\n"
        "1. Produce a Markdown comparison table: one row per aspect, one column per paper, short cells with citations.\n"
        "2. Then write 2-3 paragraphs on the key differences, trade-offs and when each approach is preferable."
    )
    text = llm.complete(prompt, GROUNDED_SYSTEM)
    out = {"comparison": text, "model": llm.name, "papers": data["papers"], "similarity": data["similarity"]}
    unknown = _check_citations(text, set(keys.values()))
    if unknown:
        out["warning"] = f"Unrecognized citation keys: {', '.join(unknown)}"
    return out


def write_literature_review(
    library: ResearchLibrary,
    llm: OllamaLLM,
    topic: str,
    paper_ids: list[str] | None = None,
    *,
    fetch_new: int = 0,
    n_themes: int = 0,
    min_year: int | None = None,
) -> dict[str, Any]:
    scaffold = library.literature_review(topic, paper_ids, fetch_new=fetch_new, n_themes=n_themes, min_year=min_year)
    cite_of = {e["paper_id"]: e["cite_key"] for t in scaffold["themes"] for e in t["papers"]}
    sections: list[str] = []

    for i, theme in enumerate(scaffold["themes"], 1):
        ids = [e["paper_id"] for e in theme["papers"]]
        evidence = library.semantic_search(f"{topic} {' '.join(theme['keywords'][:4])}", top_k=8, paper_ids=ids)
        for e in evidence:
            e["citation"] = f"[{cite_of.get(e['paper_id'], e['paper_id'])}, p. {e['page']}]"
        papers = "\n".join(
            f"- [{e['cite_key']}] {e['title']} ({e['year'] or 'n.d.'}): {e['contribution']}" for e in theme["papers"]
        )
        prompt = (
            f"Review topic: {topic}\nTheme: {theme['theme']} (keywords: {', '.join(theme['keywords'])})\n\n"
            f"Papers in this theme:\n{papers}\n\nSupporting passages:\n\n{_sources_block(evidence, 7000)}\n\n"
            "Write this theme's section (2-4 paragraphs). Start with a line '## <a descriptive section title>'. "
            "Compare the approaches, note agreements and disagreements, and cite every claim."
        )
        section = llm.complete(prompt, WRITER_SYSTEM)
        if not section.lstrip().startswith("#"):
            section = f"## {theme['theme'].title()}\n\n{section}"
        sections.append(section)

    overview = "\n".join(f"- {t['theme']}: " + ", ".join(f"[{e['cite_key']}]" for e in t["papers"]) for t in scaffold["themes"])
    gaps = "\n".join(f"- [{g['cite_key']}] {g['text']}" for g in scaffold["research_gaps"][:10]) or "- (none stated)"
    timeline = ", ".join(f"{t['year']} [{t['cite_key']}]" for t in scaffold["timeline"])
    intro = llm.complete(
        f"Review topic: {topic}\nThemes and papers:\n{overview}\nTimeline: {timeline}\n\n"
        "Write the Introduction (1-2 paragraphs): motivate the topic, outline how the field developed, and preview the themes. "
        "Start with '## Introduction'.",
        WRITER_SYSTEM,
    )
    closing = llm.complete(
        f"Review topic: {topic}\nThemes:\n{overview}\n\nLimitations stated by the papers:\n{gaps}\n\n"
        "Write two sections: '## Open Problems and Future Directions' (synthesize the gaps, cite them) and "
        "'## Conclusion' (one paragraph).",
        WRITER_SYSTEM,
    )
    body = "\n\n".join([f"# Literature Review: {topic}", intro, *sections, closing])
    keys = set(cite_of.values())
    result = {
        "review_markdown": body + "\n\n## References\n\n```bibtex\n" + scaffold["bibtex"] + "\n```\n",
        "model": llm.name,
        "n_papers": scaffold["n_papers"],
        "newly_fetched": scaffold["newly_fetched"],
        "themes": [{"theme": t["theme"], "papers": [e["cite_key"] for e in t["papers"]]} for t in scaffold["themes"]],
    }
    unknown = _check_citations(body, keys)
    if unknown:
        result["warning"] = f"The model used citation keys not in the library: {', '.join(unknown)}. Check those claims."
    return result

