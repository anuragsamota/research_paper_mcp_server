# Research Paper MCP Server

An MCP server that turns any LLM agent (Claude Desktop, Claude Code, IDEs, or the bundled
**Ollama-powered `research-agent`**) into a research assistant. You can run it fully self-hosted, with
an **Ollama server on your LAN** providing both the LLM and the embeddings. Agents can **search** arXiv and Semantic Scholar, **ingest** papers as PDFs,
**retrieve** grounded passages with a semantic search pipeline, and **analyze, compare and review**
papers. Every passage comes back with a citation like `[Hu2021, p. 4]`.

```
                    ┌──────────────────────── MCP host (Claude Desktop / Code / custom agent) ─────┐
                    │  LLM ── tools · resources · prompts ──┐                                      │
                    └───────────────────────────────────────┼──────────────────────────────────────┘
                                                            │ stdio or Streamable HTTP
┌───────────────────────────────────────────────────────────▼──────────────────────────────────────┐
│ research-mcp                                                                                     │
│                                                                                                  │
│  Discovery            Ingestion pipeline                     Retrieval            Analysis       │
│  arXiv API ──┐        PDF ─► text per page ─► sections ─►     dense vectors ┐     summaries      │
│  Semantic    ├─ add ─►      (pypdf)          (Abstract,       (embedder)    ├RRF► comparison     │
│  Scholar  ───┘                                Method, …)      BM25 ─────────┘     citations      │
│                       ─► chunks (page-tagged) ─► embeddings                       lit. review    │
│                       ─► reference list                                           BibTeX         │
│                                                                                                  │
│  SQLite library (metadata, chunks, vectors)  ·  PDFs on disk                                     │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Features

| Area | What it does |
|---|---|
| **Paper discovery** | Searches arXiv and Semantic Scholar together, merges duplicates, re-ranks results by semantic similarity to the query, and shows citation counts. |
| **PDF processing** | Extracts text page by page, fixes words hyphenated across lines, detects sections (Abstract, Introduction, Method, Experiments, Results, Discussion, Limitations, Conclusion, References, plus numbered headings), splits text into overlapping chunks tagged with their page, and parses the reference list. |
| **Embeddings** | Pluggable: `sentence-transformers` (local neural model), `ollama` (e.g. `nomic-embed-text`), or a dependency-free `hash` embedder that works offline. If you switch embedders, the library re-indexes itself. |
| **Vector search** | Hybrid retrieval. Cosine similarity over embeddings plus BM25 keyword scores, combined with reciprocal rank fusion. You can filter by paper and by section. Every chunk embedding also includes the paper's title for context. |
| **Analysis** | Extractive summaries (centrality, cue phrases and MMR de-duplication), TF-IDF keywords, highlights per section, and the limitations the authors state. |
| **Comparison** | For each aspect (problem, method, data, results, limitations, or any free-text aspect), the best evidence passage from each paper. Also shared and distinctive keywords and a pairwise similarity matrix. |
| **Citation discovery** | References and citing papers from Semantic Scholar, with influential citations and citation contexts first. Falls back to the reference list parsed from the PDF. Also finds which references are already in your library, and related papers (from your library plus Semantic Scholar recommendations). |
| **LLM backend (Ollama)** | Uses an Ollama server on this machine, your LAN or Ollama Cloud. The `research-agent` CLI is a tool-calling chat agent driven by your Ollama model. Server-side tools `ask_papers`, `synthesize_comparison` and `write_literature_review` generate cited answers, comparisons and full reviews, and flag any citation that isn't among the retrieved sources. |
| **Literature review** | Picks relevant papers (and can fetch new ones), clusters them into themes with spherical k-means, labels each theme, and extracts each paper's key contribution. Builds a timeline and a list of research gaps, and outputs evidence passages, a Markdown draft and BibTeX. |

## Quick start

```bash
git clone https://github.com/anuragsamota/research_paper_mcp_server
cd research_paper_mcp_server
python -m venv .venv && source .venv/bin/activate
pip install -e ".[st]"        # or `pip install -e .` for the lightweight hash embedder only
research-mcp --help
```

## Ollama on your LAN (self-hosted LLM backend)

```
 laptop / workstation                                  LAN GPU box (e.g. 192.168.1.50)
┌───────────────────────────────────────┐   HTTP     ┌─────────────────────────────┐
│ research-agent ─ stdio ─ research-mcp │──────────► │ Ollama :11434               │
│  (chat loop, tool calls)  (tools,     │ /api/chat  │  • llama3.1:8b / qwen2.5    │
│                            library)   │ /api/embed │  • nomic-embed-text         │
└───────────────────────────────────────┘            └─────────────────────────────┘
```

**1. On the Ollama machine**: listen on the network and pull the models:

```bash
# Linux (systemd): sudo systemctl edit ollama  → add:
#   [Service]
#   Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl restart ollama
# macOS: launchctl setenv OLLAMA_HOST 0.0.0.0:11434 and restart the Ollama app
# Windows: set the OLLAMA_HOST=0.0.0.0:11434 user environment variable and restart Ollama

ollama pull llama3.1:8b          # chat model with tool calling (or qwen2.5:7b/14b, mistral-nemo, llama3.3)
ollama pull nomic-embed-text     # embedding model
```

Allow TCP port 11434 through that machine's firewall for your LAN only. Ollama has no authentication,
so never expose it to the internet.

**2. On your machine**: point the project at it with a `.env` file in the working directory
(it's loaded automatically; see [`.env.example`](.env.example)):

```bash
RESEARCH_OLLAMA_URL=http://192.168.1.50:11434
RESEARCH_LLM_MODEL=llama3.1:8b
RESEARCH_EMBEDDER=ollama
RESEARCH_EMBEDDING_MODEL=nomic-embed-text
```

**3. Chat with your papers:**

```bash
research-agent                                   # interactive; spawns the MCP server automatically
research-agent "Find 3 recent papers on LoRA, add them, and compare their methods"
research-agent --ollama-url http://192.168.1.50:11434 --model qwen2.5:14b    # one-off overrides
research-agent --server-url http://server:8000/mcp --token change-me         # use a shared HTTP server
```

The agent checks at startup that Ollama is reachable and the model is installed, then prints each tool it calls
(`→ semantic_search({...})`). In-chat commands: `/tools`, `/reset`, `/quit`.

**Tips**
- Pick a model that supports tool calling. `llama3.1:8b` and `qwen2.5:7b` work on 8 GB of VRAM; `qwen2.5:14b`
  or larger follow multi-step plans more reliably.
- `RESEARCH_LLM_NUM_CTX` (default 8192) sets the context window. Raise it if the GPU has room.
- Use `RESEARCH_LLM_URL` to run the LLM on a different host from the embeddings.
- Requests to LAN and localhost Ollama addresses skip any `HTTP(S)_PROXY` set in the environment.
- Changing the embedding model triggers a one-time re-index of the library.

### Use with Claude Desktop

Add this to `claude_desktop_config.json` (see [`examples/claude_desktop_config.json`](examples/claude_desktop_config.json)):

```json
{
  "mcpServers": {
    "research-assistant": {
      "command": "/absolute/path/to/.venv/bin/research-mcp",
      "args": ["--data-dir", "/absolute/path/to/research-library"],
      "env": { "RESEARCH_ALLOWED_DIRS": "/absolute/path/to/your/pdfs" }
    }
  }
}
```

### Use with Claude Code

```bash
claude mcp add research-assistant -- /absolute/path/to/.venv/bin/research-mcp --data-dir ~/research-library
```

### Run as an HTTP service

```bash
RESEARCH_API_TOKEN=change-me research-mcp --transport http --host 0.0.0.0 --port 8000
# MCP endpoint: http://<host>:8000/mcp   (header: Authorization: Bearer change-me)
# Health check: http://<host>:8000/health

# or with Docker
docker build -t research-mcp .
docker run -p 8000:8000 -e RESEARCH_API_TOKEN=change-me -v research-data:/data research-mcp
```

## Example conversation

> **You:** Find recent work on parameter-efficient fine-tuning, add the 5 most relevant papers, and write me a
> literature review.
>
> **Agent:** calls `search_papers`, then `add_paper` ×5, then `literature_review`, then `semantic_search` for each theme,
> and returns a themed review with `[Hu2021, p. 3]`-style citations and BibTeX.

> **You:** How does LoRA's memory cost compare to prefix tuning?
>
> **Agent:** calls `compare_papers(["arxiv-2106.09685", "arxiv-2101.00190"], ["memory cost", "results"])`, then
> answers with a table grounded in the quoted passages.

## MCP interface

### Tools

| Tool | Purpose |
|---|---|
| `search_papers(query, source, limit, year)` | Search arXiv and Semantic Scholar |
| `add_paper(identifier, download_pdf, tags)` | Add a paper by arXiv id or URL, DOI, Semantic Scholar id, or PDF URL. Indexes the full text when an open-access PDF exists. |
| `ingest_pdf(path, title, tags)` | Index a local PDF from the inbox or an allowed directory |
| `list_papers(tag)` / `get_paper(paper_id, include_sections)` / `remove_paper(paper_id)` | Manage the library |
| `semantic_search(query, top_k, paper_ids, sections)` | Retrieve passages with citations |
| `summarize_paper(paper_id)` | Structured extractive summary |
| `compare_papers(paper_ids, aspects)` | Compare papers aspect by aspect, with evidence |
| `find_citations(paper_id, direction)` | References or citing papers |
| `extract_references(paper_id)` | Reference list parsed from the PDF, linked to papers in your library |
| `find_related_papers(paper_id)` | Similar papers in the library and from Semantic Scholar |
| `literature_review(topic, paper_ids, fetch_new, n_themes, min_year)` | Automated literature review scaffold |
| `export_bibtex(paper_ids)` | BibTeX entries |
| `ask_papers(question, paper_ids, top_k)` | **Ollama:** grounded answer with citations and sources |
| `synthesize_comparison(paper_ids, aspects)` | **Ollama:** comparison table and trade-off analysis |
| `write_literature_review(topic, …)` | **Ollama:** full review (intro, one section per theme, open problems, conclusion, BibTeX) |
| `llm_status()` | Check the Ollama connection and model |
| `library_stats()` / `reindex_library()` | Library status and re-embedding |

### Resources

| URI | Content |
|---|---|
| `papers://library` | All papers (JSON) |
| `papers://{paper_id}` | Metadata and abstract |
| `papers://{paper_id}/fulltext` | Full text by section (Markdown) |
| `papers://{paper_id}/bibtex` | BibTeX entry |

### Prompts

`review_literature(topic)`, `critique_paper(paper_id)`, `compare_approaches(paper_ids, focus)` and
`answer_question(question)` are ready-made agent workflows that chain the tools above.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `RESEARCH_ENV_FILE` | `./.env` | Settings file loaded at startup (real environment variables win) |
| `RESEARCH_DATA_DIR` | `./data` | Where the SQLite library, PDFs and `inbox/` live |
| `RESEARCH_EMBEDDER` | `auto` | `auto` (sentence-transformers if installed, otherwise hash), `hash`, `sentence-transformers` or `ollama` |
| `RESEARCH_EMBEDDING_MODEL` | backend default | e.g. `BAAI/bge-small-en-v1.5`, `nomic-embed-text` |
| `RESEARCH_OLLAMA_URL` | `http://localhost:11434` | Ollama server (LLM and embeddings), e.g. `http://192.168.1.50:11434` |
| `RESEARCH_LLM_MODEL` | `llama3.1:8b` | Ollama chat model used by `research-agent` and the generation tools |
| `RESEARCH_LLM_URL` | `RESEARCH_OLLAMA_URL` | A separate Ollama host for the LLM |
| `RESEARCH_LLM_NUM_CTX` / `RESEARCH_LLM_TEMPERATURE` / `RESEARCH_LLM_TIMEOUT` | `8192` / `0.2` / `300` | Generation options |
| `OLLAMA_API_KEY` | none | Only for Ollama Cloud or an authenticating reverse proxy |
| `RESEARCH_CHUNK_SIZE` / `RESEARCH_CHUNK_OVERLAP` | `220` / `40` | Chunk size and overlap, in words |
| `RESEARCH_ALLOWED_DIRS` | `<data>/inbox` | Folders that `ingest_pdf` may read from (separated by `os.pathsep`) |
| `RESEARCH_MAX_PDF_MB` | `50` | Maximum PDF size for downloads and local files |
| `SEMANTIC_SCHOLAR_API_KEY` | none | Raises the Semantic Scholar rate limit |
| `RESEARCH_API_TOKEN` | none | Bearer token required in HTTP mode |
| `RESEARCH_ALLOW_PRIVATE_URLS` | `0` | Allow PDF downloads from private or loopback addresses |

**Choosing an embedder.** For the best semantic retrieval, install the `st` extra, or point
`RESEARCH_EMBEDDER=ollama` at an Ollama server running `nomic-embed-text`. The `hash` embedder has no
dependencies and runs anywhere (including CI). It matches on vocabulary rather than meaning, but works
well in practice because BM25 fusion is always on.

## Security notes

- `ingest_pdf` only reads files under `RESEARCH_ALLOWED_DIRS`, so a model cannot read arbitrary files.
- URL downloads must return a real PDF (checked by its `%PDF` header) under the size limit. Every
  redirect hop is checked, and private or loopback addresses are refused (SSRF guard).
- HTTP mode supports a bearer token. Without a token, only `localhost` Host headers are accepted
  (DNS-rebinding protection).

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

The tests generate synthetic paper PDFs and mock arXiv, Semantic Scholar and Ollama, so they run offline.

### Project layout

```
research_mcp/
  server.py      MCP tools, resources and prompts
  service.py     ResearchLibrary: ingestion pipeline and high-level operations
  pdf.py         PDF text extraction, section detection, chunking
  embeddings.py  hash / sentence-transformers / Ollama embedders
  retrieval.py   hybrid dense + BM25 index with reciprocal rank fusion
  analysis.py    keywords, summaries, references, clustering, BibTeX
  sources.py     arXiv and Semantic Scholar clients, PDF download
  store.py       SQLite persistence
  http_app.py    Streamable HTTP app with token auth
  llm.py         Ollama client (chat + tool calling, health check, LAN-aware)
  generation.py  grounded Q&A, comparison synthesis, full literature reviews
  agent.py       research-agent CLI: Ollama tool-calling loop over the MCP tools
```
