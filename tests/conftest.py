from __future__ import annotations

import io
import json
from pathlib import Path

import httpx
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from research_mcp.config import Settings
from research_mcp.embeddings import HashingEmbedder
from research_mcp.service import ResearchLibrary
from research_mcp.sources import ScholarlySources

PAPERS = {
    "lora": {
        "title": "Low-Rank Adaptation for Efficient Fine-Tuning of Large Language Models",
        "author": "Alice Smith, Bob Jones",
        "sections": [
            ("Abstract", "We propose low-rank adaptation, a parameter-efficient fine-tuning method for large language "
             "models. Low-rank adaptation freezes the pretrained weights and injects trainable rank decomposition "
             "matrices into each transformer layer. This reduces the number of trainable parameters by 10,000 times "
             "while matching full fine-tuning quality on GLUE benchmarks."),
            ("1 Introduction", "Fine-tuning large language models is expensive because every parameter must be "
             "updated and stored for each downstream task. We show that weight updates during adaptation have a low "
             "intrinsic rank. Our main contribution is a simple low-rank parameterization of weight updates."),
            ("2 Method", "Low-rank adaptation represents the weight update as the product of two small matrices A and B "
             "with rank r much smaller than the hidden dimension. Only A and B are trained with the Adam optimizer, "
             "the pretrained transformer weights stay frozen. The low-rank matrices can be merged into the weights "
             "at inference so there is no additional latency."),
            ("3 Experiments", "We evaluate on the GLUE benchmark and on the E2E natural language generation dataset "
             "using RoBERTa and GPT-2 models. Training uses a learning rate of 0.0002 and rank r of 8."),
            ("4 Results", "Low-rank adaptation achieves 89.7 percent average accuracy on GLUE and outperforms adapter "
             "layers and prefix tuning with far fewer trainable parameters. GPU memory use drops by two thirds."),
            ("5 Conclusion", "Low-rank adaptation makes fine-tuning cheap. A limitation is that choosing the rank for "
             "each layer requires manual tuning, which we leave to future work."),
            ("References", "[1] Li, X. and Liang, P. 2021. Prefix-Tuning: Optimizing Continuous Prompts for Generation. ACL.\n"
             "[2] Houlsby, N. et al. 2019. Parameter-Efficient Transfer Learning for NLP. ICML.\n"
             "[3] Vaswani, A. et al. 2017. Attention Is All You Need. NeurIPS."),
        ],
    },
    "prefix": {
        "title": "Prefix-Tuning: Optimizing Continuous Prompts for Generation",
        "author": "Xiang Li; Percy Liang",
        "sections": [
            ("Abstract", "We introduce prefix tuning, a lightweight alternative to fine-tuning for natural language "
             "generation tasks. Prefix tuning keeps language model parameters frozen and optimizes a small continuous "
             "task-specific vector called the prefix. Prefix tuning obtains comparable performance to full fine-tuning "
             "while learning only 0.1 percent of the parameters."),
            ("1 Introduction", "Fine-tuning requires storing a full copy of the language model for every task. Inspired "
             "by prompting, we propose to prepend trainable continuous prefix vectors to every transformer layer."),
            ("2 Method", "The prefix is a sequence of continuous vectors prepended to the keys and values of every "
             "attention layer. We reparameterize the prefix with a small multilayer perceptron for stable optimization."),
            ("3 Experiments", "We evaluate table-to-text generation on the E2E and WebNLG datasets with GPT-2 and "
             "summarization on XSUM with BART."),
            ("4 Results", "Prefix tuning outperforms fine-tuning in low-data settings and extrapolates better to unseen "
             "topics. On XSUM prefix tuning slightly underperforms full fine-tuning."),
            ("5 Conclusion", "Prefix tuning is an efficient alternative to fine-tuning. It does not yet close the gap on "
             "summarization, which remains challenging."),
            ("References", "[1] Radford, A. et al. 2019. Language Models are Unsupervised Multitask Learners.\n"
             "[2] Houlsby, N. et al. 2019. Parameter-Efficient Transfer Learning for NLP. ICML.\n"
             "[3] Lewis, M. et al. 2020. BART: Denoising Sequence-to-Sequence Pre-training. ACL."),
        ],
    },
    "gnn": {
        "title": "Message Passing Graph Neural Networks for Molecular Property Prediction",
        "author": "Carol White",
        "sections": [
            ("Abstract", "We present a message passing graph neural network that predicts quantum chemical properties "
             "of molecules. Atoms are nodes and chemical bonds are edges. Our model reaches chemical accuracy on 11 of "
             "13 targets of the QM9 dataset."),
            ("1 Introduction", "Predicting molecular properties with density functional theory is slow. Graph neural "
             "networks learn directly from molecular graphs and are invariant to atom ordering."),
            ("2 Method", "Each atom holds a hidden state updated by aggregating messages from neighbouring atoms over "
             "several message passing steps. A readout function pools atom states into a molecule embedding."),
            ("3 Experiments", "We train on the QM9 dataset of 134 thousand small organic molecules and report mean "
             "absolute error per target."),
            ("4 Results", "The model achieves state of the art mean absolute error and outperforms handcrafted molecular "
             "fingerprints by a large margin."),
            ("5 Conclusion", "Message passing networks are accurate for molecules. A limitation is that they cannot "
             "capture long-range interactions in large proteins."),
            ("References", "[1] Gilmer, J. et al. 2017. Neural Message Passing for Quantum Chemistry. ICML.\n"
             "[2] Kipf, T. and Welling, M. 2017. Semi-Supervised Classification with Graph Convolutional Networks. ICLR.\n"
             "[3] Ramakrishnan, R. et al. 2014. Quantum chemistry structures and properties of 134 kilo molecules."),
        ],
    },
}


def make_pdf(spec: dict) -> bytes:
    """Render a small two-page paper with real section headings."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.setTitle(spec["title"])
    c.setAuthor(spec["author"])
    y = 740

    def line(text: str, size: int = 10):
        nonlocal y
        if y < 60:
            c.showPage()
            y = 740
        c.setFont("Helvetica", size)
        c.drawString(50, y, text)
        y -= size + 4

    line(spec["title"], 13)
    line(spec["author"])
    line("2021")
    y -= 10
    for i, (heading, body) in enumerate(spec["sections"]):
        if i == 3:
            c.showPage()
            y = 740
        line(heading, 12)
        if heading == "References":
            for ref in body.split("\n"):
                _wrap(ref, line)
        else:
            _wrap(body, line)
        y -= 6
    c.save()
    return buf.getvalue()


def _wrap(text: str, line, width: int = 95):
    words, current = text.split(), ""
    for w in words:
        if len(current) + len(w) + 1 > width:
            line(current)
            current = w
        else:
            current = f"{current} {w}".strip()
    if current:
        line(current)


@pytest.fixture(scope="session")
def pdfs() -> dict[str, bytes]:
    return {key: make_pdf(spec) for key, spec in PAPERS.items()}


ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2106.09685v2</id>
    <published>2021-06-17T17:37:18Z</published>
    <title>LoRA: Low-Rank Adaptation of
      Large Language Models</title>
    <summary>We propose low-rank adaptation for fine-tuning large language models efficiently.</summary>
    <author><name>Edward Hu</name></author>
    <author><name>Yelong Shen</name></author>
    <link title="pdf" href="https://arxiv.org/pdf/2106.09685v2" rel="related" type="application/pdf"/>
    <category term="cs.CL"/>
  </entry>
</feed>"""


def s2_paper(pid: str, title: str, year: int, arxiv: str = "", cites: int = 10) -> dict:
    return {
        "paperId": pid,
        "title": title,
        "year": year,
        "authors": [{"name": "Some Author"}],
        "abstract": f"Abstract of {title}.",
        "venue": "NeurIPS",
        "externalIds": {"ArXiv": arxiv} if arxiv else {"DOI": f"10.1000/{pid}"},
        "citationCount": cites,
        "url": f"https://www.semanticscholar.org/paper/{pid}",
    }


def make_handler(pdfs: dict[str, bytes], calls: list[str]):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        host, path = request.url.host, request.url.path
        if host == "export.arxiv.org":
            return httpx.Response(200, text=ARXIV_FEED)
        if host == "arxiv.org" and path.startswith("/pdf/2106.09685"):
            return httpx.Response(200, content=pdfs["lora"], headers={"content-type": "application/pdf"})
        if host == "example.org" and path == "/redirect.pdf":
            return httpx.Response(302, headers={"location": "https://example.org/prefix.pdf"})
        if host == "example.org" and path == "/prefix.pdf":
            return httpx.Response(200, content=pdfs["prefix"])
        if host == "example.org" and path == "/paywall":
            return httpx.Response(200, text="<html>login</html>")
        if host == "api.semanticscholar.org":
            if path.endswith("/paper/search"):
                return httpx.Response(200, json={"data": [
                    s2_paper("s2lora", "LoRA: Low-Rank Adaptation of Large Language Models", 2021, "2106.09685", 9000),
                    s2_paper("s2adapter", "Parameter-Efficient Transfer Learning for NLP", 2019, cites=3000),
                ]})
            if path.endswith("/references"):
                return httpx.Response(200, json={"data": [
                    {"citedPaper": s2_paper("s2prefix", "Prefix-Tuning: Optimizing Continuous Prompts for Generation", 2021, "2101.00190", 4000),
                     "isInfluential": True, "contexts": ["prefix tuning prepends vectors"]},
                    {"citedPaper": s2_paper("s2attn", "Attention Is All You Need", 2017, "1706.03762", 100000),
                     "isInfluential": False, "contexts": []},
                ]})
            if path.endswith("/citations"):
                return httpx.Response(200, json={"data": [
                    {"citingPaper": s2_paper("s2qlora", "QLoRA: Efficient Finetuning of Quantized LLMs", 2023, "2305.14314", 2000),
                     "isInfluential": True, "contexts": []},
                ]})
            if "/recommendations/" in path:
                return httpx.Response(200, json={"recommendedPapers": [
                    s2_paper("s2dora", "DoRA: Weight-Decomposed Low-Rank Adaptation", 2024, "2402.09353", 300),
                ]})
            return httpx.Response(200, json=s2_paper("s2x", "Some Paper", 2020))
        return httpx.Response(404)

    return handler


@pytest.fixture()
def calls() -> list[str]:
    return []


@pytest.fixture()
def library(tmp_path: Path, pdfs, calls) -> ResearchLibrary:
    settings = Settings(data_dir=tmp_path / "data", embedder="hash")
    settings.allowed_dirs = [settings.inbox_dir]
    client = httpx.Client(transport=httpx.MockTransport(make_handler(pdfs, calls)), follow_redirects=True)
    lib = ResearchLibrary(settings, embedder=HashingEmbedder(), sources=ScholarlySources(settings, client=client))
    yield lib
    lib.store.close()


@pytest.fixture()
def loaded(library: ResearchLibrary, pdfs) -> ResearchLibrary:
    for key in ("lora", "prefix", "gnn"):
        library.ingest_pdf_bytes(pdfs[key])
    return library


def dumps(obj) -> str:
    return json.dumps(obj, indent=2, default=str)
