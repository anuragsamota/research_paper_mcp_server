import pytest

from research_mcp.service import LibraryError


def test_ingest_and_semantic_search(loaded):
    stats = loaded.stats()
    assert stats["papers"] == 3 and stats["with_fulltext"] == 3
    hits = loaded.semantic_search("which dataset of molecules was used for training?", top_k=3)
    assert hits[0]["title"].startswith("Message Passing")
    assert hits[0]["citation"].startswith("[")
    method_hits = loaded.semantic_search("trainable rank decomposition matrices", sections=["method"])
    assert all(h["section"] == "method" for h in method_hits)
    assert method_hits[0]["title"].startswith("Low-Rank")


def test_search_filters_by_paper(loaded):
    gnn = next(p for p in loaded.store.list_papers() if "Graph" in p.title)
    hits = loaded.semantic_search("fine-tuning language models", paper_ids=[gnn.id])
    assert hits and {h["paper_id"] for h in hits} == {gnn.id}


def test_summarize(loaded):
    lora = next(p for p in loaded.store.list_papers() if p.title.startswith("Low-Rank"))
    s = loaded.summarize(lora.id)
    assert s["key_points"] and s["keywords"]
    assert any("rank" in k for k in s["keywords"])
    assert any("limitation" in x.lower() for x in s["limitations"])
    assert s["n_references"] == 3
    assert s["year"] == 2021


def test_compare(loaded):
    ids = [p.id for p in loaded.store.list_papers()]
    result = loaded.compare(ids, ["method", "data"])
    assert set(result["aspects"]) == {"method", "data"}
    for pid in ids:
        assert result["aspects"]["data"][pid]
    sims = result["similarity"]
    lora = next(p.id for p in loaded.store.list_papers() if p.title.startswith("Low-Rank"))
    prefix = next(p.id for p in loaded.store.list_papers() if p.title.startswith("Prefix"))
    gnn = next(p.id for p in loaded.store.list_papers() if "Graph" in p.title)
    assert sims[lora][prefix] > sims[lora][gnn]
    with pytest.raises(LibraryError):
        loaded.compare([lora])


def test_references_link_to_library(loaded):
    lora = next(p for p in loaded.store.list_papers() if p.title.startswith("Low-Rank"))
    refs = loaded.extract_references(lora.id)
    assert refs["count"] == 3
    assert [m["title"] for m in refs["in_library"]] == ["Prefix-Tuning: Optimizing Continuous Prompts for Generation"]


def test_literature_review(loaded):
    review = loaded.literature_review("parameter-efficient fine-tuning of language models", n_themes=2)
    assert review["n_papers"] >= 2
    assert review["draft_markdown"].startswith("# Literature review")
    assert "@" in review["bibtex"]
    theme_of = {e["title"][:6]: i for i, t in enumerate(review["themes"]) for e in t["papers"]}
    if "Messag" in theme_of:
        assert theme_of["Low-Ra"] == theme_of["Prefix"] != theme_of["Messag"]
    assert review["research_gaps"]


def test_add_paper_from_arxiv_downloads_pdf(library, calls):
    paper, status = library.add_paper("2106.09685")
    assert status == "ingested full text"
    assert paper.id == "arxiv-2106.09685"
    assert paper.title.startswith("LoRA")  # arXiv metadata wins over PDF metadata
    assert paper.authors == ["Edward Hu", "Yelong Shen"]
    assert "method" in paper.sections
    again, status = library.add_paper("https://arxiv.org/abs/2106.09685v2")
    assert status == "already in library" and again.id == paper.id


def test_add_paper_from_url_follows_redirects(library):
    paper, status = library.add_paper("https://example.org/redirect.pdf")
    assert paper.title.startswith("Prefix-Tuning") and "url" in status.lower()


def test_add_paper_rejects_non_pdf(library):
    with pytest.raises(Exception, match="did not return a PDF"):
        library.add_paper("https://example.org/paywall")


def test_metadata_only_paper_is_searchable(library):
    paper, status = library.add_paper("2106.09685", download_pdf=False)
    assert "abstract only" in status
    assert library.semantic_search("low-rank adaptation")[0]["paper_id"] == paper.id


def test_external_search_merges_sources(library):
    results = library.search_external("low rank adaptation", limit=5)
    loras = [r for r in results if r.get("arxiv_id", "").startswith("2106.09685")]
    assert len(loras) == 1 and loras[0]["citation_count"] == 9000
    assert loras[0]["add_with"]


def test_citations_and_related(library):
    paper, _ = library.add_paper("2106.09685")
    refs = library.citations(paper.id, "references")
    assert refs["source"] == "semantic_scholar"
    assert refs["papers"][0]["title"].startswith("Prefix")  # influential first
    cites = library.citations(paper.id, "citations")
    assert cites["papers"][0]["title"].startswith("QLoRA")
    related = library.related(paper.id)
    assert related["external"][0]["title"].startswith("DoRA")


def test_local_ingest_respects_allowed_dirs(library, pdfs, tmp_path):
    (library.settings.inbox_dir / "gnn.pdf").write_bytes(pdfs["gnn"])
    paper = library.ingest_local_pdf("gnn.pdf")
    assert "Graph" in paper.title
    outside = tmp_path / "secret.pdf"
    outside.write_bytes(pdfs["gnn"])
    with pytest.raises(LibraryError, match="outside the allowed"):
        library.ingest_local_pdf(str(outside))
    with pytest.raises(LibraryError):
        library.ingest_local_pdf("../../secret.pdf")


def test_private_urls_are_blocked(library):
    with pytest.raises(Exception, match="private address"):
        library.sources.download_pdf("http://127.0.0.1:8080/x.pdf")


def test_remove_and_bibtex(loaded):
    ids = [p.id for p in loaded.store.list_papers()]
    bib = loaded.bibtex(ids[:2])
    assert bib.count("@") == 2
    loaded.remove(ids[0])
    assert loaded.stats()["papers"] == 2
    assert all(h["paper_id"] != ids[0] for h in loaded.semantic_search("model", top_k=20))
