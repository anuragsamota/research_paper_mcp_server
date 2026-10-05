"""PDF processing: text extraction, section detection and chunking."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

from pypdf import PdfReader

from .text import normalize

# Canonical section names and the heading words that map to them.
_SECTION_ALIASES = {
    "abstract": "abstract",
    "introduction": "introduction",
    "background": "background",
    "related work": "related_work",
    "related works": "related_work",
    "literature review": "related_work",
    "prior work": "related_work",
    "preliminaries": "background",
    "method": "method",
    "methods": "method",
    "methodology": "method",
    "approach": "method",
    "proposed method": "method",
    "model": "method",
    "our approach": "method",
    "materials and methods": "method",
    "experiments": "experiments",
    "experiment": "experiments",
    "experimental setup": "experiments",
    "experimental results": "results",
    "evaluation": "experiments",
    "results": "results",
    "results and discussion": "results",
    "discussion": "discussion",
    "analysis": "discussion",
    "limitations": "limitations",
    "future work": "limitations",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "conclusion and future work": "conclusion",
    "conclusions and future work": "conclusion",
    "summary": "conclusion",
    "references": "references",
    "bibliography": "references",
    "acknowledgments": "acknowledgments",
    "acknowledgements": "acknowledgments",
    "appendix": "appendix",
}

# "3 Method", "3. Method", "III. METHOD", "2.1 Data" or a bare "Abstract".
_HEADING_RE = re.compile(
    r"^(?:(?P<num>(?:\d{1,2}(?:\.\d{1,2})*\.?)|(?:[IVX]{1,5}\.))\s+)?(?P<title>[A-Za-z][A-Za-z &\-]{2,60})$"
)


@dataclass
class Page:
    number: int  # 1-based
    text: str


@dataclass
class Section:
    name: str  # canonical name, e.g. "method"; "body" if unknown
    heading: str  # heading as printed
    page: int
    text: str = ""


@dataclass
class Chunk:
    index: int
    section: str
    page: int
    text: str


@dataclass
class ParsedPdf:
    pages: list[Page]
    sections: list[Section]
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    def section_text(self, name: str) -> str:
        return "\n".join(s.text for s in self.sections if s.name == name).strip()


class PdfError(ValueError):
    pass


def extract_pages(data: bytes) -> tuple[list[Page], dict[str, str]]:
    if not data.startswith(b"%PDF"):
        raise PdfError("Not a PDF file (missing %PDF header).")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            reader.decrypt("")
        pages = [Page(i + 1, normalize(page.extract_text() or "")) for i, page in enumerate(reader.pages)]
        meta = {}
        if reader.metadata:
            for key in ("title", "author", "subject"):
                value = getattr(reader.metadata, key, None)
                if value:
                    meta[key] = str(value).strip()
    except PdfError:
        raise
    except Exception as exc:  # pypdf raises many exception types for malformed files
        raise PdfError(f"Could not read PDF: {exc}") from exc
    if not any(p.text for p in pages):
        raise PdfError("The PDF has no extractable text (it may be a scanned image; run OCR first).")
    return pages, meta


def _canonical_heading(line: str) -> str | None:
    line = line.strip()
    if len(line) > 70:
        return None
    match = _HEADING_RE.match(line)
    if not match:
        return None
    title = re.sub(r"\s+", " ", match.group("title")).strip().lower()
    canonical = _SECTION_ALIASES.get(title)
    if canonical:
        return canonical
    # Numbered headings with an unknown title ("4 Scaling Laws") start a generic section,
    # but only if the title looks like a heading (Title Case or ALL CAPS, few words).
    if match.group("num"):
        raw = match.group("title").strip()
        words = raw.split()
        if 1 <= len(words) <= 6 and (raw.isupper() or all(w[0].isupper() for w in words if len(w) > 3)):
            return "body"
    return None


def detect_sections(pages: list[Page]) -> list[Section]:
    sections: list[Section] = [Section("front", "", 1)]
    buffers: list[list[str]] = [[]]
    for page in pages:
        for line in page.text.split("\n"):
            canonical = _canonical_heading(line)
            # A section that is only "Abstract text..." on one line: split heading from content.
            inline = re.match(r"^(abstract)[\s.:\-—]+(.{20,})$", line.strip(), re.IGNORECASE)
            if inline and sections[-1].name == "front":
                sections.append(Section("abstract", "Abstract", page.number))
                buffers.append([inline.group(2)])
                continue
            if canonical:
                # Sub-sections of a known section stay in that section.
                is_subsection = re.match(r"^\d{1,2}\.\d", line.strip()) is not None
                if canonical == "body" and is_subsection and sections[-1].name not in ("front", "references"):
                    buffers[-1].append(line)
                    continue
                sections.append(Section(canonical, line.strip(), page.number))
                buffers.append([])
            else:
                buffers[-1].append(line)
    for section, lines in zip(sections, buffers):
        section.text = "\n".join(lines).strip()
    return [s for s in sections if s.text]


def parse_pdf(data: bytes) -> ParsedPdf:
    pages, meta = extract_pages(data)
    return ParsedPdf(pages=pages, sections=detect_sections(pages), metadata=meta)


def chunk_sections(parsed: ParsedPdf, size: int = 220, overlap: int = 40) -> list[Chunk]:
    """Split each section into overlapping word windows, tracking the page each chunk starts on."""
    size = max(size, 20)
    overlap = max(0, min(overlap, size // 2))
    chunks: list[Chunk] = []
    page_texts = {p.number: p.text for p in parsed.pages}
    for section in parsed.sections:
        if section.name in ("references", "acknowledgments"):
            continue
        words = section.text.split()
        if not words:
            continue
        step = size - overlap
        start = 0
        while start < len(words):
            window = words[start : start + size]
            text = " ".join(window)
            chunks.append(Chunk(len(chunks), section.name, _locate_page(text, section.page, page_texts), text))
            if start + size >= len(words):
                break
            start += step
    return chunks


def _locate_page(text: str, default: int, page_texts: dict[int, str]) -> int:
    probe = " ".join(text.split()[:8])
    for number in sorted(page_texts):
        if number < default:
            continue
        if probe and probe in " ".join(page_texts[number].split()):
            return number
    return default


def guess_title(parsed: ParsedPdf) -> str:
    title = parsed.metadata.get("title", "")
    if title and len(title) > 8 and not title.lower().endswith((".pdf", ".tex", ".dvi")) and "untitled" not in title.lower():
        return title
    for line in parsed.pages[0].text.split("\n")[:12]:
        line = line.strip()
        if 15 <= len(line) <= 200 and not re.search(r"arxiv|preprint|@|http|university|conference|journal", line, re.I):
            return line
    return "Untitled paper"
