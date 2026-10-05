"""Paper analysis: keywords, extractive summaries, reference parsing, comparison, theme clustering, BibTeX."""

from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Iterable

import numpy as np

from .embeddings import Embedder
from .store import Paper
from .text import split_sentences, tokenize

# Phrases that tend to mark a paper's key claims.
_CUE_PATTERNS = [
    (re.compile(r"\b(we|this paper|this work|in this paper,? we) (propose|present|introduce|develop|describe)", re.I), 2.0),
    (re.compile(r"\b(our|the proposed) (method|model|approach|framework|system)\b", re.I), 1.0),
    (re.compile(r"\b(outperform|state[- ]of[- ]the[- ]art|improv\w+|achiev\w+|surpass\w*)\b", re.I), 1.2),
    (re.compile(r"\b(we (show|find|demonstrate|observe)|results (show|indicate|suggest))\b", re.I), 1.5),
    (re.compile(r"\b(contribution|novel|first)\b", re.I), 0.8),
]
_LIMITATION_RE = re.compile(
    r"\b(limitation|limited|does not|do not|cannot|fail\w*|future work|remain\w* (open|unclear|challenging)|"
    r"we leave|beyond the scope|drawback|shortcoming)\b",
    re.I,
)

# Query templates used to locate each aspect inside a paper.
ASPECT_QUERIES = {
    "problem": "problem addressed motivation research question challenge",
    "method": "proposed method approach model architecture algorithm framework",
    "data": "dataset benchmark corpus data collection experimental setup",
    "results": "results performance accuracy outperforms improvement evaluation metrics",
    "limitations": "limitations future work drawbacks fails does not address",
    "contributions": "main contributions we propose we introduce novel",
}
ASPECT_SECTIONS = {
    "problem": ["abstract", "introduction"],
    "method": ["method", "abstract", "introduction", "body"],
    "data": ["experiments", "method", "results", "body"],
    "results": ["results", "experiments", "abstract", "conclusion", "discussion"],
    "limitations": ["limitations", "discussion", "conclusion"],
    "contributions": ["introduction", "abstract", "conclusion"],
}


# ---------------------------------------------------------------------------- keywords


def document_frequencies(docs: Iterable[str]) -> tuple[Counter[str], int]:
    df: Counter[str] = Counter()
    n = 0
    for doc in docs:
        tokens = tokenize(doc)
        df.update(set(tokens) | {f"{a} {b}" for a, b in zip(tokens, tokens[1:])})
        n += 1
    return df, n


def keywords(text: str, df: Counter[str] | None = None, n_docs: int = 1, top_n: int = 12) -> list[str]:
    """TF-IDF key terms (unigrams and bigrams) of ``text`` relative to a corpus."""
    tokens = tokenize(text)
    tf: Counter[str] = Counter(tokens)
    bigrams = Counter(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))
    tf.update({bg: c for bg, c in bigrams.items() if c >= 2})
    df = df or Counter()
    scored = {}
    for term, count in tf.items():
        if term.replace("-", "").isdigit() or len(term) < 3:
            continue
        idf = math.log((1 + n_docs) / (1 + df.get(term, 0))) + 1
        scored[term] = (1 + math.log(count)) * idf * (1.3 if " " in term else 1.0)
    ranked = sorted(scored, key=scored.get, reverse=True)
    out: list[str] = []
    for term in ranked:
        # Skip unigrams already covered by a chosen bigram, and vice versa.
        if any(term in chosen.split() or chosen in term.split() for chosen in out):
            continue
        out.append(term)
        if len(out) >= top_n:
            break
    return out


# ---------------------------------------------------------------------------- summaries


def key_sentences(
    text: str, embedder: Embedder, top_n: int = 5, *, min_words: int = 6, max_words: int = 60
) -> list[str]:
    """Extractive summary: centrality to the document centroid + cue phrases + position."""
    sentences = [s for s in split_sentences(text) if min_words <= len(s.split()) <= max_words]
    if not sentences:
        return []
    if len(sentences) <= top_n:
        return sentences
    vectors = embedder.embed(sentences)
    centroid = vectors.mean(axis=0)
    centroid /= np.linalg.norm(centroid) or 1.0
    centrality = vectors @ centroid
    scores = []
    for i, sentence in enumerate(sentences):
        cue = sum(weight for pattern, weight in _CUE_PATTERNS if pattern.search(sentence))
        position = 0.3 * (1 - i / len(sentences))
        numbers = 0.2 if re.search(r"\d+(\.\d+)?\s*%|\b\d+\.\d+\b", sentence) else 0.0
        scores.append(float(centrality[i]) * 3 + cue * 0.5 + position + numbers)
    chosen: list[int] = []
    for i in np.argsort(scores)[::-1]:
        # Maximal marginal relevance: skip near-duplicates of already chosen sentences.
        if any(float(vectors[i] @ vectors[j]) > 0.85 for j in chosen):
            continue
        chosen.append(int(i))
        if len(chosen) >= top_n:
            break
    return [sentences[i] for i in sorted(chosen)]


def limitation_sentences(text: str, top_n: int = 4) -> list[str]:
    hits = [s for s in split_sentences(text) if _LIMITATION_RE.search(s) and 6 <= len(s.split()) <= 70]
    return hits[:top_n]


def summarize_paper(paper: Paper, embedder: Embedder, df: Counter[str] | None = None, n_docs: int = 1) -> dict:
    sections = paper.sections
    body = "\n".join(t for name, t in sections.items() if name not in ("references", "front"))
    abstract = paper.abstract or sections.get("abstract", "")
    summary: dict = {
        "id": paper.id,
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "abstract": abstract,
        "keywords": keywords(body or abstract, df, n_docs),
        "sections_found": [s for s in sections if s != "front"],
    }
    if not body:
        summary["key_points"] = key_sentences(abstract, embedder, 3)
        summary["note"] = "Only metadata is available (no PDF full text); summary is based on the abstract."
        return summary
    summary["key_points"] = key_sentences(
        " ".join(sections.get(s, "") for s in ("abstract", "introduction", "conclusion")) or body, embedder, 5
    )
    per_section = {}
    for name in ("method", "experiments", "results", "discussion", "conclusion"):
        if sections.get(name):
            per_section[name] = key_sentences(sections[name], embedder, 3)
    summary["section_highlights"] = per_section
    summary["limitations"] = limitation_sentences(
        " ".join(sections.get(s, "") for s in ("limitations", "discussion", "conclusion")) or body
    )
    summary["n_references"] = len(paper.reference_list)
    return summary


# ---------------------------------------------------------------------------- references


def parse_references(text: str, limit: int = 400) -> list[str]:
    """Split a References section into individual entries."""
    text = text.strip()
    if not text:
        return []
    # Numbered styles: "[12] Foo..." or "12. Foo..."
    numbered = re.split(r"(?:^|\n)\s*(?:\[\d{1,3}\]|\d{1,3}\.)\s+", "\n" + text)
    entries = [e for e in numbered if e.strip()]
    if len(entries) < 3:
        # Author-year styles: a new entry starts on a line beginning "Surname, X." / "Surname X," etc.
        lines = text.split("\n")
        entries, current = [], []
        starter = re.compile(r"^[A-Z][A-Za-z'\-]+(?:,| [A-Z]\.|,? and | et al)")
        for line in lines:
            if starter.match(line.strip()) and current and re.search(r"(19|20)\d{2}", " ".join(current)):
                entries.append(" ".join(current))
                current = []
            current.append(line.strip())
        if current:
            entries.append(" ".join(current))
    cleaned = [re.sub(r"\s+", " ", e).strip() for e in entries]
    return [e for e in cleaned if len(e) > 25][:limit]


def reference_title_guess(entry: str) -> str:
    """Heuristically pull the title out of a reference string."""
    quoted = re.search(r"[“\"](.{10,250}?)[”\"]", entry)
    if quoted:
        return quoted.group(1).strip(" .,")
    # "Authors. Year. Title. Venue" or "Authors (Year). Title. Venue" or "Authors. Title. Venue, Year"
    after_year = re.search(r"(?:\(|\b)(?:19|20)\d{2}[a-z]?\)?[.,]\s+(.{10,250}?)[.?!](?:\s|$)", entry)
    if after_year:
        return after_year.group(1).strip()
    parts = [p.strip() for p in re.split(r"\.\s+", entry) if p.strip()]
    if len(parts) >= 2:
        return max(parts[1:3], key=len)
    return entry[:150]


def title_similarity(a: str, b: str) -> float:
    norm = lambda s: " ".join(tokenize(s, keep_stopwords=True))  # noqa: E731
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def match_references_to_library(paper: Paper, library: list[Paper], threshold: float = 0.82) -> list[dict]:
    """Find which entries in ``paper``'s reference list are papers already in the library."""
    matches = []
    others = [p for p in library if p.id != paper.id]
    for entry in paper.reference_list:
        guess = reference_title_guess(entry)
        best, best_score = None, 0.0
        for other in others:
            score = title_similarity(guess, other.title)
            if score > best_score:
                best, best_score = other, score
            elif other.title.lower() in entry.lower():
                best, best_score = other, 1.0
        if best and best_score >= threshold:
            matches.append({"reference": entry[:300], "paper_id": best.id, "title": best.title, "score": round(best_score, 2)})
    return matches


# ---------------------------------------------------------------------------- comparison & clustering


def similarity_matrix(vectors: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    ids = list(vectors)
    out: dict[str, dict[str, float]] = {}
    for a in ids:
        out[a] = {b: round(float(vectors[a] @ vectors[b]), 3) for b in ids if b != a}
    return out


def kmeans(matrix: np.ndarray, k: int, iterations: int = 50, seed: int = 0) -> np.ndarray:
    """Spherical k-means with k-means++ init; returns a label per row."""
    n = matrix.shape[0]
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    centers = [matrix[rng.integers(n)]]
    for _ in range(1, k):
        dist = 1 - np.max(matrix @ np.array(centers).T, axis=1)
        dist = np.clip(dist, 0, None)
        probs = dist / dist.sum() if dist.sum() > 0 else np.full(n, 1 / n)
        centers.append(matrix[rng.choice(n, p=probs)])
    centers_arr = np.array(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iterations):
        new_labels = np.argmax(matrix @ centers_arr.T, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for c in range(k):
            members = matrix[labels == c]
            if len(members):
                v = members.mean(axis=0)
                centers_arr[c] = v / (np.linalg.norm(v) or 1.0)
    return labels


def default_theme_count(n_papers: int) -> int:
    if n_papers < 4:
        return 1
    return max(2, min(6, round(math.sqrt(n_papers / 1.5))))


# ---------------------------------------------------------------------------- BibTeX


def bibtex_entry(paper: Paper, key: str | None = None) -> str:
    key = key or paper.cite_key
    fields = {
        "title": "{" + paper.title + "}",
        "author": " and ".join(paper.authors) or "Unknown",
        "year": str(paper.year) if paper.year else "",
        "journal" if paper.venue and paper.venue != "arXiv" else "howpublished": paper.venue
        or (f"arXiv:{paper.arxiv_id}" if paper.arxiv_id else ""),
        "doi": paper.doi,
        "eprint": paper.arxiv_id,
        "archivePrefix": "arXiv" if paper.arxiv_id else "",
        "url": paper.url,
    }
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
    kind = "article" if paper.venue and paper.venue != "arXiv" else "misc"
    return f"@{kind}{{{key},\n{body}\n}}"


def unique_cite_keys(papers: list[Paper]) -> dict[str, str]:
    """paper id -> cite key, disambiguated with a/b/c suffixes."""
    counts = Counter(p.cite_key for p in papers)
    seen: Counter[str] = Counter()
    out = {}
    for p in papers:
        key = p.cite_key
        if counts[key] > 1:
            key = f"{key}{'abcdefghijklmnopqrstuvwxyz'[seen[key] % 26]}"
            seen[p.cite_key] += 1
        out[p.id] = key
    return out
