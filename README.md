# Agentic Information Extraction System

Q&A over multi-modal, long-context PDF documents. A ReAct agent retrieves evidence from an indexed document, produces a cited draft answer, and a grounding validator checks every claim before the answer reaches the caller.

> **New here?** Read [`REPORT.md`](REPORT.md) first — it answers the assignment's design questions, lists what's implemented, and records the key decisions. [`DESIGN.md`](DESIGN.md) is the deep architecture dive; this README is setup and usage.

---

## Quickstart

Clone, install, add keys, index the sample PDF once, then ask — in order:

```bash
git clone https://github.com/hamitguner/agentic-rag-pdf-qa.git && cd agentic-rag-pdf-qa
uv sync
cp .env.example .env          # then add your ANTHROPIC_API_KEY and OPENAI_API_KEY

# 1) Index the sample PDF (one-time; builds the vector store under data/)
uv run main.py --pdf "data/llama3-eval.pdf" --collection llama3_herd \
  --description "Meta's Llama 3 model family technical report" --ingest-only

# 2) Ask a question (prints the grounded answer + confidence + citations)
uv run main.py --collection llama3_herd \
  --question "How was Llama 3 pre-trained?"

# 3) (Optional) Reproduce the committed 8-question demo report
uv run scripts/evaluate.py --collection llama3_herd \
  --questions eval/questions_llama3.json --out demo_output_llama3.md
```

> Requires an **Anthropic** key (the LLM) and an **OpenAI** key (query embeddings). The
> sections below are the full reference: every CLI flag, the REST API, the Streamlit UI,
> and the Turkish + vision showcase.

---

## Architecture

```
PDF ──► pdf_loader ──► chunker ──► indexer ──► ChromaDB
                                                   │
question ──► LangGraph graph ──────────────────────┘
               │
               ├─ prepare_node    (turn setup + memory reset)
               ├─ triage_node     (Claude Haiku — document vs chit-chat)
               ├─ research_node   (Claude Sonnet + 5 RAG tools)
               ├─ validate_node   (Claude Haiku — LLM-as-judge)
               └─ respond_node    (grounded answer, or abstain if unverified)
```

See [`REPORT.md`](REPORT.md) for the submission overview and [`DESIGN.md`](DESIGN.md) for full architecture decisions and trade-off arguments.

---

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Anthropic API key (Claude Sonnet for research, Claude Haiku for validation)
- OpenAI API key (text-embedding-3-small for vector search)
- LangSmith account (optional, for tracing)

---

## Setup

```bash
# 1. Clone and install
git clone https://github.com/hamitguner/agentic-rag-pdf-qa.git
cd agentic-rag-pdf-qa
uv sync

# 2. Configure environment
cp .env.example .env   # then fill in your keys
```

### Environment Variables (`.env`)

```env
# Required
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional — model override
MODEL_NAME=claude-sonnet-4-5-20250929

# Optional — conversation memory window (recent Q/A pairs kept per session)
MEMORY_MAX_PAIRS=5

# Optional — LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls__...
LANGSMITH_PROJECT=agentic-rag-pdf-qa
```

---

## Usage

### Ingest a PDF

```bash
# Ingest with auto-derived collection name (slug of filename → "llama3_eval")
uv run main.py --pdf "data/llama3-eval.pdf"

# Ingest with a custom collection name + description (description is injected
# into the agent prompts so triage and research know the corpus context).
# This is the exact command that produced the committed demo_output_llama3.md.
uv run main.py --pdf "data/llama3-eval.pdf" --collection "llama3_herd" \
  --description "Meta's Llama 3 model family technical report"

# (Illustrative) Add a SECOND, related PDF to the SAME collection — one shared
# search space. Point --pdf at any additional PDF you want to fold in.
uv run main.py --pdf "data/another-related.pdf" --collection "llama3_herd" --ingest-only

# Wipe just this collection, then ingest fresh
uv run main.py --pdf "data/llama3-eval.pdf" --collection "llama3_herd" --reset --ingest-only
```

> **Collections vs. documents.** A *collection* is a named group of one or more PDFs on a
> related topic, stored in one ChromaDB instance so a single search spans every PDF in the
> group. Each chunk keeps its own `doc_id`, so citations remain traceable to the source PDF.
> Collection names are slugified (spaces/Turkish → safe ASCII); the ingest log prints the
> canonical name to use when asking questions.

### Ask a Question

```bash
# Ingest + ask in one shot
uv run main.py \
  --pdf "data/llama3-eval.pdf" \
  --collection "llama3_herd" \
  --question "What is the parameter count of the flagship Llama 3 model, and how many tokens was it pre-trained on?"

# Ask only (document already indexed)
uv run main.py \
  --question "Which preference-optimization algorithm does Llama 3 use for alignment?" \
  --collection "llama3_herd"
```

### Conversation Memory (multi-turn)

Reuse a `--session-id` to keep context across questions, so follow-ups resolve against earlier turns:

```bash
uv run main.py --collection "llama3_herd" --session-id s1 \
  --question "What context length does Llama 3 support?"

uv run main.py --collection "llama3_herd" --session-id s1 \
  --question "And how is the model trained to reach it?"   # resolves via turn-1 context
```

Memory is session-scoped (LangGraph `InMemorySaver`, keyed on `thread_id`) and capped to the
most recent `MEMORY_MAX_PAIRS` Q/A pairs. Each turn still retrieves fresh evidence — memory
only carries conversational context, never replaces retrieval. See `DESIGN.md` §6.

**Sample output:**

```
============================================================
ANSWER:
The flagship Llama 3 model has 405B parameters and was pre-trained on
15.6T text tokens [llama3_eval_p1_c1].

------------------------------------------------------------
Grounded: True  |  Confidence: 100%
Citations: llama3_eval_p1_c1
============================================================
```

> Note how the collection is `llama3_herd` but the citation prefix is `llama3_eval`: the
> `doc_id` is the slug of the source file name (`llama3-eval.pdf`), independent of the collection
> name. Because every chunk id carries its own `doc_id`, a collection can hold several PDFs and
> each citation still traces back to the exact source document.

### CLI Reference

| Flag | Description |
|---|---|
| `--pdf PATH` | PDF to ingest |
| `--question TEXT` | Question to answer |
| `--collection NAME` | Collection name (default: slug of PDF filename) |
| `--description TEXT` | Collection description; stored in the registry and injected into the agent prompts |
| `--session-id ID` | Conversation session for memory (default: `default`) |
| `--reset` | Wipe the target collection's folder before ingesting (other collections survive) |
| `--ingest-only` | Ingest without running a question |

---

## LangGraph Studio (local debug UI)

```bash
uv run langgraph dev
```

Open the URL printed in the output (`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`).

Send input as:
```json
{
  "question": "Your question here",
  "collection": "llama3_herd",
  "session_id": "s1"
}
```

---

## REST API

A FastAPI surface exposes the same pipeline (the document must already be indexed):

```bash
uv run uvicorn src.api.main:app --reload
```

```bash
# Health check
curl http://127.0.0.1:8000/health

# List indexed collections with their descriptions (used by the Streamlit picker)
curl http://127.0.0.1:8000/collections

# Ask a question (reuse session_id for conversation memory)
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How was Llama 3 pre-trained?", "collection": "llama3_herd", "session_id": "s1"}'
```

Interactive docs at `http://127.0.0.1:8000/docs`. Errors degrade gracefully via the same
`invoke_safely` boundary as the CLI — a model/API failure returns a clean JSON body, not a 500.

---

## Web UI (Streamlit)

A minimal chat front-end that talks to the REST API over HTTP — pick a collection/session,
type a question, and watch the conversation unfold with grounding metadata under each answer.
Run the API and the UI in two terminals:

```bash
# Terminal 1 — start the API (graph built once, memory persists per session)
uv run uvicorn src.api.main:app --reload

# Terminal 2 — start the UI
uv run streamlit run ui/streamlit_app.py
```

The sidebar sets the API base URL and the session id, and offers a **collection picker**
populated live from `GET /collections` (it falls back to a free-text field if the API is
unreachable, or shows "no collections yet" if none are indexed). The selected collection's
description is shown beneath the picker. Reusing a session carries conversation memory
server-side, and **New session** starts fresh while keeping past chats listed for re-opening.

---

## Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run only agent-layer tests
uv run pytest tests/test_agent.py -v

# Run only preprocessing tests
uv run pytest tests/test_preprocessing.py -v
```

Tests are fully offline — no API keys required. External dependencies (LLM, ChromaDB) are either avoided with in-memory fakes or patched with `unittest.mock`.

---

## Project Structure

```
src/
├── preprocessing/
│   ├── pdf_loader.py      # PyMuPDF: text, PNG renders, TOC, table/figure detection
│   └── chunker.py         # Page-atomic chunking (512 tokens, 64 overlap)
├── retrieval/
│   ├── embedder.py        # OpenAI text-embedding-3-small
│   ├── indexer.py         # ChromaDB upsert, metadata
│   └── retriever.py       # Dense retrieval, RRF stub
├── agent/
│   ├── graph.py           # LangGraph StateGraph + invoke_safely (graceful failure)
│   ├── nodes.py           # prepare, triage, smalltalk, research, validate, respond
│   ├── state.py           # InputState / OutputState / AgentState
│   ├── tools.py           # 5 RAG tools (search, outline, section, index, vision)
│   ├── triage.py          # classify_intent (document vs chit-chat)
│   ├── validation.py      # grounding_check with structured output
│   └── prompts/           # evidence.txt, validation.txt, triage.txt
├── api/                   # FastAPI surface (POST /ask, GET /health)
├── registry.py            # Data-layout paths + collections.csv (collection→description)
├── slug.py                # Canonical doc/collection slug (shared)
├── cli.py                 # ingest() + ask() + argparse main
├── config.py              # Pydantic settings + load_dotenv
└── log.py                 # Colored logging, get_logger()

tests/
├── test_preprocessing.py  # pdf_loader + chunker
├── test_retrieval.py      # slugify + indexer + retriever + multi-PDF collection
├── test_registry.py       # path helpers + collections.csv round-trip (offline)
├── test_agent.py          # nodes, routing, memory, triage, description injection (offline)
├── test_tools.py          # analyze_image vision tool (offline, vision model mocked)
└── test_api.py            # FastAPI /health + /ask (offline)

eval/questions_llama3.json # Main eval set — Llama 3 paper (English, long doc)
eval/questions_fomc.json   # Showcase eval set — FOMC PDF (Turkish + vision)
scripts/evaluate.py        # Eval / demo-output harness
ui/streamlit_app.py        # Streamlit chat client over the REST API

# Generated data layout (gitignored) — see src/registry.py:
data/
├── collections.csv                  # collection,description registry
└── <collection>/
    ├── chroma/                      # one vector store per collection
    └── <doc_id>/                    # one folder per source PDF
        ├── <original_name>.pdf
        ├── pages/  page1.png ...
        └── outline.json             # TOC as {section, level, start_page, end_page}

main.py                    # CLI entry point (delegates to src/cli.py)
REPORT.md                  # Submission overview — answers the assignment + key decisions
DESIGN.md                  # Architecture decisions and trade-offs
langgraph.json             # LangGraph Studio config
```

---

## Evaluation / Demo Output

`scripts/evaluate.py` runs a labelled question set (each item has `question`, `expected_answer`,
and a `purpose` tag) through the full pipeline and prints each answer with its grounding status —
doubling as the demo output and the bonus evaluation harness. Two committed sets are provided:

- `eval/questions_llama3.json` — 8 questions on *The Llama 3 Herd of Models* (the main corpus: a
  long, ~92-page English paper), spanning early-to-late sections to exercise long-document navigation.
- `eval/questions_fomc.json` — 7 questions on a short Turkish FOMC report, kept as the
  **multilingual + vision showcase**: it exercises Turkish Q&A and reading an embedded chart via
  `analyze_image` (see the committed `demo_output_fomc.md`).

```bash
# Run the main long-document set against the Llama 3 collection (full 8 questions):
uv run scripts/evaluate.py --collection llama3_herd \
  --questions eval/questions_llama3.json --out demo_output_llama3.md

# Spot-check just the first 3 questions (prints to terminal; add --out to save):
uv run scripts/evaluate.py --collection llama3_herd \
  --questions eval/questions_llama3.json --limit 3

# Multilingual + vision showcase (Turkish report, produces demo_output_fomc.md):
uv run scripts/evaluate.py --collection fomc_june \
  --questions eval/questions_fomc.json --limit 3 --out demo_output_fomc.md
```

Each question runs in its own session (no memory bleed). The `--out` file is a markdown report
pairing every expected answer with the produced answer, grounding flag, confidence, and citations.

## Error Handling

The pipeline degrades gracefully on the two failure paths the assignment calls out:

- **Corrupt / unreadable PDF** — `load_document` raises `DocumentError`; the CLI logs a clear
  message and exits non-zero (covered by tests).
- **LLM / API failure** — `invoke_safely` wraps the graph run at the boundary and turns any
  terminal failure (a model/API error such as auth, quota or outage — or an unexpected internal
  error like a missing index) into a clean ungrounded answer instead of a traceback; the full
  error is always logged with its traceback, so a genuine bug stays visible rather than silently
  swallowed. The handling is **provider-agnostic**: because the model is chosen at runtime via
  `init_chat_model`, error handling is not bound to any one SDK's exception types, so swapping
  Anthropic → OpenAI/Gemini needs no change. Transient errors are already retried inside the
  chat model (`max_retries`); this boundary only catches terminal failures.
