# Agentic Information Extraction System — Design Document

**Position:** Software Developer — Agentic Platform Team  
**Stack:** Python 3.11 · PyMuPDF · ChromaDB · OpenAI Embeddings · LangChain · LangGraph · Anthropic Claude

---

## Architecture Overview

```
                        ┌─────────────────────────────────────────┐
                        │           INGESTION PIPELINE             │
                        │  (runs once per document, offline)       │
                        │                                          │
  PDF File ─────────── ▶│  pdf_loader.py                          │
                        │  · PyMuPDF text extraction               │
                        │  · Page PNG renders (150 DPI)            │
                        │  · TOC → section labels                  │
                        │  · Table / figure detection              │
                        │           │                              │
                        │  chunker.py                              │
                        │  · RecursiveCharacterTextSplitter        │
                        │  · 512-token page-atomic chunks          │
                        │           │                              │
                        │  indexer.py                              │
                        │  · text-embedding-3-small (OpenAI)       │
                        │  · ChromaDB upsert (cosine, HNSW)        │
                        └────────────────┬────────────────────────┘
                                         │
                              ChromaDB (persisted to disk)
                                         │
                        ┌────────────────▼────────────────────────┐
                        │           QUERY PIPELINE                 │
                        │         (per user question)              │
                        │                                          │
 Question + ──────────▶│  graph.py  (LangGraph StateGraph)       │
 session_id            │   + InMemorySaver checkpointer (memory)  │
                        │                                          │
                        │  ┌──────────────────────────────────┐   │
                        │  │  prepare_node  (once per turn)   │   │
                        │  │  append question · reset retries │   │
                        │  └──────────────┬───────────────────┘   │
                        │                 ▼                        │
                        │  ┌──────────────────────────────────┐   │
                        │  │  triage_node  (Claude Haiku)     │   │
                        │  │  document? or chit-chat?         │   │
                        │  └───┬──────────────────────┬───────┘   │
                        │      │ chit-chat            │ document  │
                        │      ▼                      ▼           │
                        │  smalltalk_node ─▶ END   (continue ↓)   │
                        │  ┌──────────────────────────────────┐   │
                        │  │  research_node                   │   │
                        │  │  create_agent (Claude)           │   │
                        │  │  ┌──────────────────────────┐    │   │
                        │  │  │  ReAct Loop              │    │   │
                        │  │  │  · search_semantic       │    │   │
                        │  │  │  · get_outline           │    │   │
                        │  │  │  · fetch_section_content │    │   │
                        │  │  │  · fetch_chunks_by_index │    │   │
                        │  │  │  · analyze_image         │    │   │
                        │  │  └──────────────────────────┘    │   │
                        │  └──────────────┬───────────────────┘   │
                        │                 │                        │
                        │  ┌──────────────▼───────────────────┐   │
                        │  │  validate_node                   │   │
                        │  │  grounding_check (Claude Haiku)  │   │
                        │  │  structured output → grounded?   │   │
                        │  └───┬──────────────────────────────┘   │
                        │      │                                   │
                        │   grounded / spent ─▶ respond ─▶ END     │
                        │   retry    ────────▶ research (max 1x)  │
                        │   (respond emits the answer or abstains; │
                        │    prunes turn → compact Q→A memory)     │
                        └──────────────────────────────────────────┘
                                         │
                              final_answer + is_grounded
                              + confidence + citations
```

---

## 1. Document Pre-processing

**Library choice: PyMuPDF (fitz)**

PyMuPDF was chosen over pdfplumber and Adobe PDF Extract for three reasons:

- **Richer API in one package**: text extraction, table detection (`page.find_tables()`), embedded image enumeration (`page.get_images()`), TOC access (`doc.get_toc()`), and page rendering (`page.get_pixmap()`) — all without additional dependencies.
- **Performance**: PyMuPDF processes pages via native C bindings (MuPDF), which matters for long financial or academic PDFs.
- **Vision support**: rendering pages to 150 DPI PNG gives the agent a visual fallback for tables and figures without a separate PDF-to-image tool.

**What the loader produces per page (`PageData`):**
- Raw extracted text
- PNG image path (150 DPI render)
- Section title (derived from TOC — see §2)
- Structural flags: `has_table`, `has_figure`, `requires_vision`

Figure detection uses `page.get_images()` (embedded raster images) and intentionally skips `get_drawings()` — vector primitives (lines, borders, decorations) would generate false positives. Tables are detected via PyMuPDF's built-in `find_tables()`.

**Trade-off acknowledged**: Section assignment is page-level. A page spanning two TOC sections is labeled with the first one. Fixing this would require character-offset intersection with the TOC, adding complexity without measurably changing retrieval quality for the MVP.

---

## 2. Structural Navigation

The document's table of contents is extracted at load time and saved as a JSON outline (`data/<collection>/<doc_id>/outline.json`). The TOC is collapsed into a **fully-qualified breadcrumb path** per heading (e.g. `Pre-Training > Model Architecture`), so leaf titles that repeat across a long paper (multiple "Model Architecture" sections) stay unique. The file is written in the same `{section, level, start_page, end_page}` shape the agent sees at query time, so it is human-readable and needs no translation. During ingestion, every page is assigned the breadcrumb of the most recent TOC entry at or before its page number — an O(n) sorted lookup — and that label is stored as metadata on every chunk and indexed in ChromaDB.

The agent navigates structure through four explicit tools:

1. **`get_outline()`** — returns the document's section map (breadcrumb, level, page span), derived from the **indexed metadata** rather than the raw file, so it always reflects exactly what is searchable. The returned `section` strings are the exact keys to scope searches with.
2. **`search_semantic(query, section=...)`** — semantic search, optionally scoped to one section via a metadata filter.
3. **`fetch_section_content(section_name)`** — retrieves all chunks in a named section directly, bypassing ranking. Useful when the answer is spread across an entire section.
4. **`fetch_chunks_by_index(chunk_indexes)`** — retrieves specific chunks by their global index, e.g. to fetch neighbours (index ± 1, ± 2) and recover context severed at a chunk boundary.

**Search-first, outline-on-demand.** The agent is instructed to start with `search_semantic` for essentially every question — semantic search finds most answers directly and cheaply — and to escalate to `get_outline` only when the question is explicitly about the document's structure, or when a first search came back scattered and the agent needs to scope into a section. This keeps navigation explicit and tool-driven rather than embedded in retrieval heuristics, while avoiding a wasted outline call on the common case. The agent reasons about which access pattern fits the question, which is more transparent and testable than a hybrid strategy that conflates structure and semantics.

**Known limitation (multi-PDF collections).** `fetch_section_content` and `fetch_chunks_by_index` filter only by `section` / `chunk_index`, not by `doc_id`. In a collection holding two or more PDFs, two documents can share a section path or a global chunk index, so these direct-lookup tools could return rows from both. `search_semantic` is unaffected (it ranks by embedding and each hit carries its own `doc_id`/citation). Single-PDF collections — including the demo corpora — are fully correct today; the fix is a per-call `doc_id` filter on the two lookup tools, deferred as it is not exercised by the current MVP.

---

## 3. Retrieval Strategy

**Primary: Dense retrieval (semantic search)**

Queries and document chunks are embedded with **OpenAI text-embedding-3-small**. Chosen over local alternatives (sentence-transformers) because:
- Native multilingual support including Turkish, without fine-tuning
- API-based — no GPU requirement, consistent latency
- Cost-effective at $0.02/1M tokens — several times cheaper than text-embedding-3-large

Vectors are stored in **ChromaDB** (persistent, cosine similarity, HNSW index). Preferred over FAISS because:
- Persistence without serialization code — vectors survive process restarts
- Metadata filtering (`where={"section": ...}`, `where={"chunk_index": {"$in": [...]}}`) enables structural tools without a separate lookup table
- Idempotent upsert via deterministic chunk IDs (`{doc_id}_p{page}_c{chunk_index}`, the same form the agent cites) — re-indexing the same document is safe

**Retrieval quality levers:**
- **Chunk size (512 tokens, 64 overlap)**: precise enough for retrieval hits, large enough to contain a complete argument or table row. The 64-token overlap ensures sentences split at boundaries appear fully in at least one chunk.
- **Page-atomic chunking**: chunks never cross page boundaries, so every chunk maps to exactly one rendered PNG. This makes `requires_vision` actionable — `analyze_image` can be called on the chunk's image without ambiguity.
- **Focused, iterative search**: the evidence prompt instructs the agent to issue *one* focused query first and answer as soon as it can, only searching again to fill a specific gap — it does not fan out many speculative queries at once. On a long document, speculative multi-query fan-out tended to pull in tangential chunks and weaken grounding, so the policy is deliberately conservative (see §2, *search-first, outline-on-demand*).

**Visual retrieval:** When `requires_vision == true`, the agent calls `analyze_image`, which runs a vision model on the page PNG **inside the tool** and returns a concise **text** description (chart titles, axes, legends, table values). Crucially, the raw image never enters the research agent's message history — only the text does. This is deliberate: a 150-DPI page encoded as base64 *text* is ~200k tokens and overflows the model's context window in a single call. Following LangChain's guidance for multimodal tool content ("have the tool return a concise text description instead of base64 blocks"), the research loop stays text-only and bounded, while the vision work is encapsulated as a self-contained capability. The agent is instructed to use this tool only after text retrieval is insufficient.

**Sparse retrieval (BM25)** and **reciprocal rank fusion** are stubbed as extension points in `retriever.py`. Not implemented for MVP: dense retrieval alone is sufficient for Turkish financial/policy documents where vocabulary is consistent, and hybrid retrieval without tuned fusion weights often degrades precision.

---

## 4. Agent Architecture

**Single ReAct agent with a grounding validation loop**

The system uses one reasoning agent rather than a multi-agent architecture. This was a deliberate scope decision:

> *"Mükemmel bir tek-ajan sistemi, yarım kalmış çok-ajan sistemden değerlidir."* — Assignment spec

Document Q&A is a **linear-with-feedback** task. A state machine captures this better than emergent multi-agent coordination, which adds communication overhead and is harder to test. An initial coordinator agent was prototyped and removed — its only jobs were rephrasing the question as an investigation brief and synthesising the evidence agent's output, both of which the ReAct agent handles directly.

**LangGraph StateGraph** (`src/agent/graph.py`) wires these nodes:

| Node | Model | Role |
|---|---|---|
| `prepare_node` | — | One-shot turn setup: append question, reset per-turn state |
| `triage_node` | Claude Haiku | Intent router: document question vs. chit-chat |
| `smalltalk_node` | — | Direct reply for chit-chat; skips retrieval entirely |
| `research_node` | Claude Sonnet (configurable) | ReAct loop: calls retrieval tools, produces cited draft |
| `validate_node` | Claude Haiku | LLM-as-judge: checks every claim against retrieved chunks |
| `respond_node` | — | Single terminal: emits the grounded answer, or abstains (`is_grounded=False`, agent's own draft) when the retry budget is spent |

**Intent triage**: Not every message is a document question. A greeting ("hey there"), a thank-you, or gibberish ("asdfjkl") should not embed a query, search the index, run the ReAct agent, and grounding-check the result — ~3 LLM calls plus retrieval to conclude "not in the document". `triage_node` makes one cheap Haiku call (structured output, like the validator) to classify the turn; chit-chat routes to `smalltalk_node` and returns directly. The classifier is **biased toward `document` when unsure**, so a borderline case degrades to a normal search and a genuine question is never refused. It also receives the previous answer as context, so terse follow-ups enabled by session memory ("and how is it quantized?") classify correctly. Triage is language-agnostic — it replies in the user's language, so the Turkish showcase corpus is handled the same way (e.g. a follow-up like "peki ya çekirdek PCE?").

**Why `create_agent`**: LangChain's agent runtime handles the tool-call / tool-result loop internally. The graph has no separate `tools` node — execution is encapsulated inside `research_node`, keeping the retrieval path conceptually clean.

**Why `init_chat_model`**: The research model is injected via `config={"configurable": {"model": ...}}` at invoke time, making it swappable through configuration without touching node code.

**State design:**
```
InputState:  { question, collection, session_id }
OutputState: { final_answer, is_grounded, confidence, citations }
AgentState:  InputState
             + messages          (add_messages reducer — ReAct scratchpad + memory)
             + retrieved_chunks  (re-parsed from the turn's tool messages each
                                  validation pass — no reducer; the messages list
                                  is already the single source of truth)
             + citations         (chunk ids cited in the draft)
             + intent, draft_answer, retries, validation, final_answer
```

**Graph routing:**
```
START → prepare → triage
triage → smalltalk → END           (chit-chat)
triage → research → validate
validate → respond    (grounded, or budget exhausted → abstain)
validate → research   (retry, ≤ MAX_RETRIES=1)
respond / smalltalk → END
```

---

## 5. Validation and Reliability

**Intent triage as a first guard.** Reliability starts before retrieval: `triage_node` keeps non-document turns (greetings, gibberish) out of the grounding path entirely, so the validator only ever judges answers that actually make document claims. This avoids spurious "ungrounded" verdicts on inputs that were never questions, and the safe-default-to-`document` bias ensures the guard never silently drops a real question.

**LLM-as-judge grounding check**

The validation layer answers: *does every claim in the draft exist in the retrieved chunks?* This is a **faithfulness check**, not a correctness check — at runtime there is no ground truth to compare against.

LLM-as-judge was chosen over rule-based extraction because sentence-level entailment from chunks to claims requires natural language reasoning. Regex or keyword matching misses paraphrases and flags legitimate inferences.

**How it works** (`src/agent/validation.py`):
1. All retrieved chunks are formatted with chunk IDs and page numbers
2. Claude Haiku receives question + draft + chunks via `init_chat_model(...).with_structured_output(ValidationResult)`
3. Provider-native structured output returns a guaranteed-valid `ValidationResult` (Pydantic) — no JSON parsing, no fallback hacks
4. `is_grounded=True` → `respond_node` (emits the validated answer)
5. `is_grounded=False` → unverified claims injected as targeted critique, `retries++`, route back to `research_node`

**Citation enforcement**: The evidence prompt requires the agent to cite chunk IDs inline (`[chunk_id]`) for every factual claim. The validator checks whether cited chunks contain the claimed facts — a tractable, specific check rather than open-ended claim validation.

**Grounding vision-derived facts**: `analyze_image` returns a text description, not an `evidence_chunks` array, so its output would otherwise be invisible to the validator — every fact read off a chart or table would be flagged unverified and trigger a wasted retry. To prevent this, the chunk-parser synthesises a **vision pseudo-chunk** (`vision_pageN`, carrying the tool's text result and source page) from each `analyze_image` call. The agent cites these as `[vision_pageN]`, and the validator grounds them exactly like text chunks — so the multimodal path is verifiable end-to-end.

**Honest absence handling**: If the agent declares "the document does not cover X," the validation prompt treats this as `is_grounded=True, confidence=1.0` — absence of evidence is a valid answer. Without this rule, natural absence declarations trigger unnecessary retries.

**Abstaining**: After `MAX_RETRIES=1`, if grounding still fails, the single terminal `respond_node` takes its abstain branch — it returns the agent's own last draft flagged `is_grounded=False` rather than a hardcoded string or a fabricated confident answer. The caller detects this via `is_grounded=False` in `OutputState`. Keeping this as one branch of the terminal (rather than a separate node) keeps the graph honest: routing only decides *answer vs retry*, and the terminal decides *confident vs abstained*. The worst case is always an honest, clearly-flagged attempt.

**Failure paths**: The two failure modes the assignment calls out are handled so the user never sees a traceback. A corrupt or unreadable PDF raises `DocumentError` during loading; the CLI logs it and exits non-zero. An LLM/API failure is caught at the run boundary by `invoke_safely`, which returns a clean ungrounded `OutputState` instead of crashing mid-graph. The catch is intentionally broad — it also converts an unexpected internal error (e.g. a missing index) into the same clean response — but the full error is always logged with its traceback, so a genuine bug stays visible rather than silently swallowed. It is **provider-agnostic by design**: since the model is selected at runtime via `init_chat_model` (Anthropic now, OpenAI/Gemini later), binding error handling to one SDK's exception types would silently break on a provider swap. Transient errors are already retried inside the chat model; the boundary only converts *terminal* failures.

---

## 6. Memory Management

Memory operates at two timescales: **within a turn** (the ReAct scratchpad) and **across turns in a session** (conversation memory). Both are implemented; long-term cross-session learning is deliberately deferred.

**Within a turn:** The `messages` field accumulates the full ReAct loop across retries via `add_messages` — tool results, critique messages, and previous draft attempts stay visible, so on retry the agent knows what it already retrieved and what was flagged.

**Across turns (implemented):** The graph compiles with a LangGraph `InMemorySaver` checkpointer keyed on `thread_id`. The CLI exposes this as `--session-id`: reusing a session lets a follow-up like *"and how is the model trained to reach it?"* resolve against the previous question. Three design points make this correct rather than just "turning on a checkpointer":

1. **Per-turn reset (`prepare_node`).** A checkpointer persists the *entire* state, so without intervention turn 2 would inherit turn 1's `retries` count (abstaining too early) and the new question would never enter `messages` (the agent would re-answer turn 1). A one-shot `prepare_node` at `START` — which the retry loop bypasses, re-entering at `research` — appends the new question and resets `retries`, `validation`, and `draft_answer`.

2. **Conversation, not scratchpad.** Persisting the raw ReAct trace would carry tool dumps and base64 page images forward forever. At turn end, `respond_node` (and `smalltalk_node`) build a list of `RemoveMessage`s via `_conversation_cleanup`, which the `add_messages` reducer applies to collapse the turn to a clean `Q→A` pair. This runs *after* validation, so grounding still sees the live tool messages; and because prior turns' tool messages are already gone, the validator is **auto-scoped** to the current turn's chunks. `RemoveMessage` is chosen over `trim_messages` (which can orphan a tool-call/tool-result pair) and over invoke-time filtering (which never cleans persisted state) precisely because of this timing requirement.

3. **Bounded window.** History is capped to `settings.memory_max_pairs` recent `Q→A` pairs (default 5, configurable via `.env`), keeping token cost and latency flat regardless of session length.

**Across sessions / long-term (deferred):** True cross-task learning — a persistent store of Q&A pairs or chunk-utility scores that informs *future, unrelated* sessions — is **not** built. It is a clean extension: swap `InMemorySaver` for `SqliteSaver` for durable threads, and add a semantic store that retrieves similar past answers. It was deferred because the failure modes are hard to defend in an MVP: **stale cached answers** when the document is re-indexed, and **cache-invalidation** complexity. Per the "scope discipline" rule, a correct bounded short-term memory beats a half-working long-term one. The agent always retrieves fresh evidence; memory only carries conversational context, never substitutes for retrieval.

---

## Appendix: Additional Implementation Decisions

> The two *headline* design decisions (the Claude Agent SDK → LangGraph migration and the
> token-bounded page-atomic chunking strategy) are written up as the official technical note in
> [`REPORT.md` §G](REPORT.md). The two decisions below are additional, reinforcing choices.

### Decision A: Single-agent over multi-agent

An early prototype included a coordinator agent that delegated to an evidence agent and a separate validation agent. This was removed for the following reasons:

The coordinator's only jobs were (a) rephrasing the user question as an investigation brief and (b) synthesising the evidence agent's output into a draft. The ReAct evidence agent can perform both directly. Removing the coordinator eliminated one LLM call per question — along with its latency and failure surface — and removed a layer whose failure mode (the coordinator generating an overly narrow brief) could silently degrade retrieval without surfacing an error.

The remaining pipeline is a state machine with deterministic routing. Each routing function is a pure function on state, testable without an LLM. This is a stronger reliability argument than "we have more agents."

### Decision B: Provider-native structured output for validation

The original validation layer parsed the LLM response with `json.loads()` and fell back to `is_grounded=False` on parse failure. This was a silent failure mode: a malformed response was indistinguishable from a genuinely ungrounded answer, causing an unnecessary retry.

Replacing this with `init_chat_model(...).with_structured_output(ValidationResult)` moves schema enforcement to Anthropic's API. The response is guaranteed to deserialise into a valid `ValidationResult` Pydantic model. The validation prompt no longer needs to specify JSON format. False negatives from parse failures are eliminated.

The same pattern applies across the codebase: where the LLM must produce structured data, use structured output. Where it must produce natural language (the research agent), do not constrain it.

---

*All architectural decisions described above are reflected in the codebase under `src/`. See `README.md` for setup and usage instructions.*
