# Agentic RAG with Claude

A production-quality Agentic RAG (Retrieval-Augmented Generation) system powered by Anthropic's Claude. Unlike basic RAG pipelines that blindly retrieve then generate, this system uses Claude's **tool-use API** to reason about *when* to search, *what* to search for, and *how* to chain multiple retrievals together to answer complex questions.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Claude](https://img.shields.io/badge/LLM-Claude%20Haiku-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-green)

---

## What makes this stand out

- **Agentic tool-use loop** — Claude decides whether to search, calculate, or answer directly, rather than always retrieving blindly
- **Multi-hop reasoning** — Claude chains multiple searches to answer complex questions (e.g. "compare Q2 and Q3 revenue and calculate the growth rate")
- **Hybrid retrieval** — combines semantic search (sentence-transformers embeddings) with BM25 keyword search, merged via Reciprocal Rank Fusion
- **Cross-encoder reranking** — a second-stage reranker boosts precision before generation
- **Step-by-step trace viewer** — every tool call is logged and displayed in the UI so you can inspect the agent's reasoning
- **Fully local embeddings** — no embedding API key needed; `all-MiniLM-L6-v2` runs on CPU

---

## Project structure

```
agentic-rag/
├── src/
│   ├── agent.py          # Claude tool-use loop (core of the system)
│   ├── retriever.py      # Hybrid BM25 + semantic search + reranking
│   ├── vectorstore.py    # ChromaDB wrapper, chunking, local embeddings
│   ├── tools.py          # Tool schemas passed to Claude's API
│   └── tracer.py         # Agent step logger
├── tests/
│   ├── test_agent.py     # safe_eval, ToolExecutor, AgentTracer tests
│   ├── test_retriever.py # BM25, RRF, format_context tests
│   └── test_vectorstore.py # Chunking and DocumentChunk tests
├── data/
│   └── docs/             # Drop your documents here
├── app.py                # Streamlit chat UI
├── ingest.py             # CLI: index documents into ChromaDB
├── requirements.txt
├── .env.example
├── Makefile
├── Dockerfile
└── docker-compose.yml
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/agentic-rag.git
cd agentic-rag
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First run downloads two models (~150MB total):
> - `all-MiniLM-L6-v2` — for embeddings (~80MB)
> - `cross-encoder/ms-marco-MiniLM-L-6-v2` — for reranking (~70MB)
>
> Both are cached after the first download.

### 4. Set your API key

```bash
cp .env.example .env
```

Open `.env` and add your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Get a key at [console.anthropic.com](https://console.anthropic.com). The project uses `claude-haiku-4-5-20251001` by default — the most cost-efficient model. A full demo session costs a few cents.

### 5. Add documents and index them

Drop any `.pdf`, `.txt`, or `.md` files into `data/docs/`, then run:

```bash
python ingest.py
```

You'll see a progress bar and a chunk count. To wipe and re-index from scratch:

```bash
python ingest.py --reset
```

### 6. Run the app

```bash
streamlit run app.py
```

Opens at **http://localhost:8501** automatically.

---

## Example questions to try

Once you have documents indexed, try questions like these to see the agentic behaviour in action:

- *"What documents do you have?"* — triggers `list_sources` tool
- *"Summarise the key points from all documents"* — triggers multiple `search_documents` calls
- *"What does the report say about Q3 revenue?"* — single focused retrieval
- *"Compare Q2 and Q3 figures and calculate the growth rate"* — multi-hop: search → search → calculate
- *"What risks are mentioned across the documents?"* — cross-document synthesis

Each answer shows an expandable **Agent trace** panel so you can see exactly which tools Claude called and what it retrieved.

---

## Running the tests

```bash
pytest tests/ -v
```

All tests are self-contained with mocks — no API keys or internet connection needed.

```
tests/test_agent.py        # 20 tests — safe_eval, ToolExecutor, AgentTracer
tests/test_retriever.py    # 9 tests  — BM25, RRF, format_context
tests/test_vectorstore.py  # 7 tests  — chunking, DocumentChunk
```

---

## Configuration

All settings are in `.env`:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | required | Your Anthropic API key |
| `CLAUDE_MODEL` | `claude-haiku-4-5-20251001` | Claude model to use |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | Where ChromaDB stores its data |
| `TOP_K_RETRIEVAL` | `10` | Candidates fetched in recall stage |
| `TOP_K_RERANK` | `4` | Final chunks passed to Claude after reranking |

To use a more powerful model, change `CLAUDE_MODEL` to `claude-sonnet-4-5-20251001`.

---

## Tech stack

| Component | Technology |
|---|---|
| LLM + tool-use | Anthropic Claude (Haiku) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector database | ChromaDB (local, persistent) |
| Keyword search | BM25 via `rank-bm25` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| UI | Streamlit |
| PDF parsing | pypdf |

---

## Docker (optional)

```bash
docker-compose up --build
```

The app will be available at `http://localhost:8501`. Mount your documents volume before building if you want them pre-indexed.

---

## How it works

```
User question
     │
     ▼
Claude (tool-use API)
     │
     ├─ search_documents("query") ──► BM25 + Semantic search
     │                                      │
     │                               Reciprocal Rank Fusion
     │                                      │
     │                               Cross-encoder rerank
     │                                      │
     │◄────────────── top-k chunks ─────────┘
     │
     ├─ calculate("expression") ──► safe AST evaluator ──► result
     │
     ├─ list_sources() ──► ChromaDB metadata scan ──► source list
     │
     └─ (repeat as needed for multi-hop)
          │
          ▼
     Final answer with inline citations
```

---

## License

MIT — free to use, modify, and include in your portfolio.
