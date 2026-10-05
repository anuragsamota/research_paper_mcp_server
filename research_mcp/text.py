"""Small, dependency-free text utilities: tokenizing, sentence splitting, cleaning."""

from __future__ import annotations

import re
import unicodedata

STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before being below
    between both but by can could did do does doing down during each either et al etc few for from further
    had has have having he her here hers herself him himself his how however i if in into is it its itself
    just may me might more most must my myself no nor not now of off on once only or other our ours
    ourselves out over own same she should so some such than that the their theirs them themselves then
    there these they this those through thus to too under until up upon us very via was we were what when
    where which while who whom why will with within without would yet you your yours yourself yourselves
    one two three use used using based show shown shows paper work new propose proposed approach method
    methods results result figure fig table section e g ie eg i.e e.g
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z][a-z0-9\-]*[a-z0-9]|[a-z]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")
_ABBREV_RE = re.compile(r"\b(e\.g|i\.e|et al|Fig|Eq|Sec|vs|cf|resp|approx|No)\.\s", re.IGNORECASE)


def normalize(text: str) -> str:
    """Normalize unicode, fix hyphenation across line breaks, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")  # soft hyphen
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate "embed-\nding"
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def stem(token: str) -> str:
    """Very light suffix stripping so "trained"/"training"/"trains" match "train"."""
    if len(token) <= 4 or "-" in token:
        return token
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    for suffix, min_len in (("ing", 6), ("ed", 5), ("es", 5), ("s", 4)):
        if token.endswith(suffix) and len(token) >= min_len and not token.endswith(("ss", "us", "is")):
            base = token[: -len(suffix)]
            if suffix in ("ing", "ed") and len(base) > 2 and base[-1] == base[-2] and base[-1] not in "lsz":
                base = base[:-1]  # "running" -> "run"
            if suffix == "es" and not base.endswith(("ch", "sh", "x", "ss", "z")):
                base = token[:-1]  # "molecules" -> "molecule"
            return base
    return token


def tokenize(text: str, *, keep_stopwords: bool = False, stemmed: bool = False) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if not keep_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return [stem(t) for t in tokens] if stemmed else tokens


def split_sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text).strip()
    if not flat:
        return []
    # Protect common abbreviations from being treated as sentence ends.
    protected = _ABBREV_RE.sub(lambda m: m.group(0).replace(". ", "<DOT> "), flat)
    parts = _SENT_SPLIT_RE.split(protected)
    return [p.replace("<DOT> ", ". ").strip() for p in parts if len(p.strip()) > 1]


def truncate(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"
