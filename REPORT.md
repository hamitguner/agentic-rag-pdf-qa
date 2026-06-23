# Final Report — Agentic Information Extraction System

**Candidate take-home · Agentic Platform Team**
**Stack:** Python 3.11 · PyMuPDF · ChromaDB · OpenAI embeddings · LangChain / LangGraph · Anthropic Claude

This report is the submission front-door. It answers the six architecture questions from the
assignment, shows which mandatory and bonus requirements are implemented and where, justifies the
technology choices, and records the two most important design decisions I lived through. It is
intentionally concise; the deep architectural reasoning (with the ASCII diagram) lives in
[`DESIGN.md`](DESIGN.md), which this report links to section by section. Setup and run instructions
are in [`README.md`](README.md).

---

## A. What the system is — in one paragraph

The system answers questions about long, multi-modal PDFs (financial reports, academic papers) the
way a careful human reader would: it indexes the document once, then for each question an agent
**searches for evidence, reads charts/tables when needed, drafts a cited answer, and a second model
verifies every claim against the retrieved text before the answer is returned.** It is a
single-reasoning-agent RAG pipeline built as an explicit LangGraph state machine, with a grounding
validator that makes the system **abstain rather than hallucinate** when evidence is missing. The
committed demo over *The Llama 3 Herd of Models* (a ~92-page paper) answers 8/8 questions grounded
with correct, traceable citations spanning pages 1–57; a second demo over a Turkish FOMC report
showcases multilingual Q&A and chart reading via the vision tool.

---

## B. Architecture design — the six questions

### 1. Document pre-processing — how is text and visual content parsed? Which tools?

PDF parsing uses **PyMuPDF (`fitz`)** because it provides text extraction, table detection
(`find_tables()`), embedded-image enumeration (`get_images()`), TOC access (`get_toc()`), and page
rendering — all in one fast, native library. For each page the loader (`src/preprocessing/pdf_loader.py`)
produces: the extracted text, a **150-DPI PNG render** of the page, a TOC-derived **section
breadcrumb**, and three structural flags — `has_table`, `has_figure`, and `requires_vision`
(= table *or* figure). Figures are detected from embedded raster images only; `get_drawings()` is
deliberately skipped because vector primitives (cell borders, rules) cause heavy false positives.
The visual content is therefore not parsed eagerly — it is *flagged*, and the agent decides at query
time whether a page actually needs the vision tool. → *Detail: [`DESIGN.md` §1](DESIGN.md).*

### 2. Structural navigation — how does the agent find the right section in a long document?

At ingestion the TOC is collapsed into **fully-qualified breadcrumb paths** (e.g.
`Pre-Training > Model Architecture`) so repeated leaf titles stay unique, written to a JSON
`outline.json`, and stamped onto every chunk as metadata in ChromaDB. The agent navigates structure
with explicit tools: `get_outline()` (the section map, rebuilt from indexed metadata so it always
matches what is searchable), `search_semantic(query, section=…)` (search scoped to one section),
`fetch_section_content()` (all chunks of a section), and `fetch_chunks_by_index()` (neighbouring
chunks for context). The policy is **search-first, outline-on-demand**: the agent issues *one
focused* semantic query first and only escalates to the outline when the question is explicitly
structural or the first search came back scattered. If a PDF has no TOC, `section` is `none` and the
agent simply relies on unscoped semantic search. → *Detail: [`DESIGN.md` §2](DESIGN.md).*

### 3. Retrieval strategy — how are text-based and visual search integrated?

Text retrieval is **dense**: queries and chunks are embedded with OpenAI `text-embedding-3-small`
and stored in **ChromaDB** (cosine, HNSW, persistent), with metadata filtering for the structural
tools. Visual retrieval is integrated as a **tool, not a parallel index**: when a chunk is flagged
`requires_vision` (or the user asks about a chart/table), the agent calls `analyze_image`, which runs
a vision model on the page PNG *inside the tool* and returns a **concise text description** (titles,
axes, legends, exact values). Crucially the raw image never enters the agent's message history —
only the text does — which keeps the context bounded (a 150-DPI page as base64 is ~200k tokens and
would overflow the window). Text and vision thus share one reasoning loop and one citation scheme.
→ *Detail: [`DESIGN.md` §3](DESIGN.md).*

### 4. Agent architecture — how many agents, what roles, how do they communicate?

One **reasoning agent** inside a five-node **LangGraph state machine**, not a multi-agent crew.
Nodes: `prepare` (per-turn setup), `triage` (Claude Haiku — document question vs. chit-chat),
`research` (Claude Sonnet running a `create_agent` ReAct loop over the 5 RAG tools), `validate`
(Claude Haiku grounding judge), and `respond` (terminal: emit the grounded answer or abstain). Nodes
communicate through **typed shared state**, and routing between them is a set of **pure functions**
testable without an LLM. A multi-agent coordinator was prototyped and removed — its only jobs
(rephrasing the question, synthesising the output) the ReAct agent already does — which removed an
LLM call and a silent failure mode. → *Detail: [`DESIGN.md` §4](DESIGN.md).*

### 5. Validation and reliability — how are wrong answers prevented?

Every answer passes an **LLM-as-judge grounding check** (`src/agent/validation.py`): Claude Haiku
receives the question, the draft, and the retrieved chunks, and returns a **provider-native
structured** `ValidationResult` (is_grounded, confidence, unverified_claims, cleaned answer). The
agent must cite chunk IDs inline (`[doc_p4_c2]`); the validator checks those claims against the
chunks. If grounding fails, the unverified claims are injected back as a targeted critique and the
agent retries once; if it still fails, `respond` **abstains** — returns the draft flagged
`is_grounded=False` rather than fabricating confidence. Vision facts are made verifiable via a
`vision_pageN` pseudo-chunk so chart-derived claims are grounded like text. Honest "the document does
not cover X" answers are treated as grounded. → *Detail: [`DESIGN.md` §5](DESIGN.md).*

### 6. Memory management — is cross-task learning possible? How?

Memory runs at two timescales. **Within a turn**, the ReAct scratchpad (tool calls, critiques,
prior draft) persists across the retry so the agent knows what it already tried. **Across turns**, a
LangGraph `InMemorySaver` keyed on `--session-id` lets a terse follow-up ("and how is it
quantized?") resolve against the previous answer — the agent rewrites it into a fresh standalone
query, so memory carries *context* but never replaces retrieval. The turn is compacted to a clean
Q→A pair (via `RemoveMessage`) and the history is capped to a bounded window. **Long-term
cross-session learning** (a durable store of past answers) is deliberately deferred: the MVP keeps a
correct bounded short-term memory rather than a half-working cache that could serve stale answers
after a re-index. → *Detail: [`DESIGN.md` §6](DESIGN.md).*

---

## C. MVP requirements — what is implemented

**Mandatory (5/5):**

| Requirement | Status | Where |
|---|---|---|
| PDF input + text extraction | ✅ | `src/preprocessing/pdf_loader.py` (PyMuPDF) |
| Retrieval layer | ✅ | dense embeddings + ChromaDB — `src/retrieval/` |
| Agent loop with tool-calling | ✅ | `research_node` + `create_agent` + 5 tools — `src/agent/` |
| Validation layer | ✅ | LLM-as-judge grounding check — `src/agent/validation.py` |
| CLI (PDF path + question as args) | ✅ | `main.py` / `src/cli.py` — `--pdf`, `--question` |

**Bonus (all five addressed):**

| Bonus | Status | Notes |
|---|---|---|
| Visual content support | ✅ | `analyze_image` vision tool + 150-DPI page renders + `requires_vision` flag |
| Multiple / specialized agents | ✅ | specialized **nodes** (triage, research, validate) each a focused LLM role; a single *reasoning* agent is a deliberate scope choice (see §G) |
| Memory module | ✅ | session memory (`InMemorySaver`) + bounded window |
| Structured outline (XML/JSON) | ✅ | hierarchical `outline.json` per document |
| Evaluation code (≥3 Q&A) | ✅ | `scripts/evaluate.py` + `eval/questions_llama3.json` (8) and `eval/questions_fomc.json` (7) |

---

## D. Technology choices

| Layer | Choice | Why this one |
|---|---|---|
| PDF processing | **PyMuPDF** | text + tables + images + TOC + rendering in one fast native library (vs. pdfplumber/Adobe) |
| Embeddings | **OpenAI `text-embedding-3-small`** | strong multilingual (incl. Turkish) with no GPU; ~5× cheaper than `-large` |
| Vector store | **ChromaDB** | persistent on disk, metadata filtering for the structural tools, idempotent upsert — preferred over FAISS for an MVP |
| Agent framework | **LangChain + LangGraph** | explicit, testable state machine with first-class tracing (see §G) |
| LLM | **Anthropic Claude** (Sonnet research / Haiku triage+validate+vision) | strong tool-use + native structured output; cheap Haiku for the cheap roles |
| Tokenization | **tiktoken** | token-accurate chunk sizing |
| Tests | **pytest** | 84 offline tests, no API keys required |
| Interfaces (extra) | **FastAPI + Streamlit** | REST surface + minimal chat UI over the same pipeline |

All of these are on the assignment's "viewed favourably" list.

---

## E. Code quality

- **Modular by component** — `preprocessing/`, `retrieval/`, `agent/` (with validation as its own module), exactly the separation the rubric asks for.
- **Type hints + docstrings** on public functions throughout.
- **`pyproject.toml`** for dependencies; `requires-python = ">=3.11"`.
- **README** with setup, env vars, and run commands for CLI / API / UI / Studio.
- **84 pytest tests**, fully offline (LLM and ChromaDB faked/mocked) — across preprocessing, retrieval, registry, agent nodes/routing/memory, tools, and the API.
- **Centralised logging** via `get_logger` (no `print`, no bare `logging`); UTF-8-safe console.

---

## F. Deliverables

| # | Deliverable | Artifact |
|---|---|---|
| 1 | GitHub repo | this repository |
| 2 | Architecture design document | [`DESIGN.md`](DESIGN.md) (6 questions + ASCII diagram) + this report |
| 3 | Working MVP code | `src/` + `main.py` (meets all mandatory requirements) |
| 4 | Demo output (≥3 Q&A on ≥1 PDF) | [`demo_output_llama3.md`](demo_output_llama3.md) (8 Q&A) and [`demo_output_fomc.md`](demo_output_fomc.md) (multilingual + vision) |
| 5 | Short technical note (200–400 words) | §G below |

---

## G. Technical note — the two key decisions

### Decision 1 — Migrating from the Claude Agent SDK to LangGraph

I first built the pipeline on the Claude Agent SDK. It worked and was quick to stand up, but as the
system grew it became hard to reason about: control flow lived implicitly inside prompts, and
inspecting or steering the agent's state mid-run felt like fighting a black box. Because this task is
graded on *reliability and justified design*, not raw capability, I migrated to a **LangGraph state
machine**. Each step — triage, research, validation, response — is now an explicit node; routing is
a set of pure functions over typed state that I unit-test without calling an LLM; and every run is
traceable in LangSmith. I traded a little initial velocity for control, observability, and
testability — the right trade for a system whose entire job is to be trustworthy.

### Decision 2 — Token-bounded, page-atomic chunks over whole-page chunks

The obvious first design was one chunk per page: trivial to build and it keeps a page intact. But it
is expensive and blunt at query time — every retrieval hit drags back a whole page of tokens, most
of it irrelevant, which raises cost and dilutes the grounding signal the validator depends on. I
instead split each page into **512-token chunks (64 overlap)** while keeping chunks **page-atomic**,
and attach each page's **section breadcrumb and image path as metadata**. Retrieval stays precise and
cheap; the section metadata still gives the agent whole-section access on demand
(`fetch_section_content`, section-scoped search); and page-atomicity means each chunk maps to exactly
one rendered image, which is what makes the `requires_vision` → `analyze_image` path unambiguous.

Three smaller decisions reinforce these: a **single reasoning agent plus an LLM judge** rather than a
multi-agent crew (scope discipline); **provider-native structured output** for the validator instead
of `json.loads` (no silent parse-failure retries); and a **search-first** navigation policy (one
focused query, escalate to the outline only when needed) that keeps tool usage tight.

---

## H. Key trade-offs and reliability

The trade-offs that shaped the system — the heart of the architecture grade:

- **Single agent vs. multi-agent.** Document Q&A is linear-with-feedback; a deterministic state
  machine is easier to test and debug than emergent multi-agent coordination. *"A polished
  single-agent system beats a half-finished multi-agent one."*
- **Dense-only vs. hybrid retrieval.** BM25 + reciprocal-rank-fusion are left as explicit, documented
  stubs in `retriever.py`. Hybrid retrieval without tuned fusion weights often *hurts* precision, so
  for the MVP dense retrieval alone is the honest choice.
- **Whole-page vs. token-bounded chunks.** Chose 512-token page-atomic chunks for retrieval precision
  and bounded token cost (see §G).
- **Vision as a text-returning tool.** Keeps the reasoning loop bounded and within the context window;
  the alternative (raw image in history) does not scale.
- **Search-first vs. always-outline.** Avoids a wasted outline call on the common case and, in
  practice, avoided an over-retrieval failure mode where speculative fan-out pulled in tangential
  chunks and broke grounding.
- **Short-term vs. long-term memory.** A correct bounded session memory over a half-working durable
  cache that risks stale answers after re-indexing.
- **Structured output vs. JSON parsing.** Provider-native structured output removes a class of silent
  validator failures.

**Error handling — the two failure paths the assignment calls out:**

- **Corrupt / unreadable PDF** → `load_document` raises `DocumentError`; the CLI logs a clear message
  and exits non-zero (covered by tests).
- **LLM / API failure** → caught at the run boundary by `invoke_safely`, which returns a clean
  ungrounded answer instead of a traceback. The catch is **provider-agnostic by design**: the model
  is selected at runtime via `init_chat_model`, so error handling is not bound to one SDK's exception
  types and survives a provider swap. Transient errors are already retried inside the chat model; the
  boundary only converts *terminal* failures.

**What I attribute retrieval quality to:** token-bounded page-atomic chunks (precision without losing
a complete idea), a multilingual embedding model, section metadata for scoped search, a search-first
policy that avoids noisy over-retrieval, and — most importantly — the grounding validator, which
turns "retrieved the wrong thing" into an honest abstention instead of a confident wrong answer.

---

## I. Pointers

- **Deep architecture + diagram:** [`DESIGN.md`](DESIGN.md)
- **Setup, env vars, run commands (CLI / REST API / Streamlit / LangGraph Studio):** [`README.md`](README.md)
- **Demo outputs:** [`demo_output_llama3.md`](demo_output_llama3.md), [`demo_output_fomc.md`](demo_output_fomc.md)
- **Quick run:**
  ```bash
  uv run main.py --pdf "data/llama3-eval.pdf" --collection llama3_herd \
    --description "Meta's Llama 3 model family technical report"
  uv run main.py --collection llama3_herd --question "How was Llama 3 pre-trained?"
  uv run pytest tests/        # 84 tests, offline
  ```
