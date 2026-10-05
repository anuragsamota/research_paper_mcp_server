import pytest

from research_mcp.analysis import kmeans, parse_references, reference_title_guess
from research_mcp.embeddings import HashingEmbedder
from research_mcp.pdf import PdfError, chunk_sections, parse_pdf
from research_mcp.sources import parse_identifier
from research_mcp.text import split_sentences

from conftest import PAPERS


def test_parse_pdf_detects_sections_and_pages(pdfs):
    parsed = parse_pdf(pdfs["lora"])
    names = [s.name for s in parsed.sections]
    for expected in ("abstract", "introduction", "method", "experiments", "results", "conclusion", "references"):
        assert expected in names
    assert parsed.section_text("experiments").startswith("We evaluate on the GLUE")
    chunks = chunk_sections(parsed, size=30, overlap=5)
    assert all(c.section != "references" for c in chunks)
    assert {c.page for c in chunks if c.section == "results"} == {2}
    assert any(len(c.text.split()) == 30 for c in chunks)  # long sections split into windows


def test_rejects_non_pdf():
    with pytest.raises(PdfError):
        parse_pdf(b"<html>not a pdf</html>")


def test_reference_parsing():
    refs = parse_references(PAPERS["gnn"]["sections"][-1][1])
    assert len(refs) == 3
    assert reference_title_guess(refs[0]) == "Neural Message Passing for Quantum Chemistry"


def test_hashing_embedder_similarity():
    e = HashingEmbedder()
    v = e.embed(["low rank adaptation of language models", "adaptation of language models with low rank", "molecular graphs"])
    assert v.shape == (3, e.dim)
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])


def test_kmeans_separates_clusters():
    e = HashingEmbedder()
    texts = ["fine tuning language models parameters"] * 3 + ["molecules graph neural chemistry"] * 3
    labels = kmeans(e.embed(texts), 2)
    assert len(set(labels[:3])) == 1 and len(set(labels[3:])) == 1 and labels[0] != labels[3]


@pytest.mark.parametrize(
    "ref,expected",
    [
        ("2106.09685", ("arxiv", "2106.09685")),
        ("arXiv:2106.09685v2", ("arxiv", "2106.09685v2")),
        ("https://arxiv.org/abs/2106.09685", ("arxiv", "2106.09685")),
        ("https://arxiv.org/pdf/2106.09685v1.pdf", ("arxiv", "2106.09685v1")),
        ("10.1145/3292500.3330701", ("doi", "10.1145/3292500.3330701")),
        ("https://doi.org/10.1038/nature14539", ("doi", "10.1038/nature14539")),
        ("https://example.org/paper.pdf", ("url", "https://example.org/paper.pdf")),
        ("cs/0112017", ("arxiv", "cs/0112017")),
    ],
)
def test_parse_identifier(ref, expected):
    assert parse_identifier(ref) == expected


def test_split_sentences_keeps_abbreviations():
    s = split_sentences("We follow Smith et al. in this setup. Results improve, e.g. by 3%. Done here.")
    assert len(s) == 3
