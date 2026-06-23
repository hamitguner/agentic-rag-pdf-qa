"""Streamlit chat UI — a thin client over the FastAPI ``POST /ask`` endpoint.

Run the API and this UI in two separate processes:

    uv run uvicorn src.api.main:app --reload      # terminal 1
    uv run streamlit run ui/streamlit_app.py      # terminal 2

The UI keeps its own per-session transcripts in ``st.session_state.conversations``
(a dict keyed by session id) so past chats stay listed in the sidebar and can be
re-opened. The *backend* conversation memory lives in the graph checkpointer keyed
by the same session id, so re-opening a chat also resumes its server-side context.
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx
import streamlit as st

DEFAULT_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 120.0


def _new_session_id() -> str:
    """Short, human-readable session id for a fresh conversation."""
    return f"ui-{uuid.uuid4().hex[:8]}"


def _fetch_collections(base_url: str) -> list[dict[str, str]] | None:
    """GET /collections — returns the list, or None if the API is unreachable."""
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/collections", timeout=5.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError:
        return None


def _ask(base_url: str, question: str, collection: str, session_id: str) -> dict[str, Any]:
    """POST one question to the API and return the parsed JSON response.

    Raises httpx.HTTPError on connection failure or non-2xx status so the caller
    can render a friendly inline error instead of crashing.
    """
    resp = httpx.post(
        f"{base_url.rstrip('/')}/ask",
        json={"question": question, "collection": collection, "session_id": session_id},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _meta_caption(meta: dict[str, Any]) -> str:
    """Render grounding metadata as a compact one-line caption."""
    grounded = "✓ grounded" if meta.get("is_grounded") else "✗ not grounded"
    confidence = f"{meta.get('confidence', 0.0):.0%}"
    citations = meta.get("citations") or []
    cites = f" · citations: {', '.join(citations)}" if citations else ""
    return f"{grounded} · confidence {confidence}{cites}"


def _conversation_title(messages: list[dict[str, Any]], session_id: str) -> str:
    """Label a conversation by its first user question, falling back to its id."""
    for msg in messages:
        if msg["role"] == "user":
            text = msg["content"].strip().replace("\n", " ")
            return text[:40] + "…" if len(text) > 40 else text
    return f"(empty) {session_id}"


# --- Session-state bootstrap -------------------------------------------------
# conversations: {session_id -> list[message]}; active_session: the open one.
if "conversations" not in st.session_state:
    first = _new_session_id()
    st.session_state.conversations = {first: []}
    st.session_state.active_session = first


def _start_new_session() -> None:
    """Archive the current chat (kept in the dict) and open a fresh one."""
    new_id = _new_session_id()
    st.session_state.conversations[new_id] = []
    st.session_state.active_session = new_id


def _open_session(session_id: str) -> None:
    """Switch the active conversation to a past one."""
    st.session_state.active_session = session_id


st.set_page_config(page_title="Agentic RAG Chat", page_icon="📄")
st.title("📄 Agentic RAG — Document Chat")

# --- Sidebar: settings + conversation list ----------------------------------
with st.sidebar:
    st.header("Settings")
    base_url = st.text_input("API base URL", value=DEFAULT_BASE_URL)

    collections_data = _fetch_collections(base_url)
    if collections_data is None:
        collection = st.text_input("Collection (indexed doc_id)", value="")
        st.caption("⚠️ Could not reach the API.")
    elif collections_data:
        options = [c["collection"] for c in collections_data]
        descriptions = {c["collection"]: c["description"] for c in collections_data}
        selected = st.selectbox("Collection", options=options)
        collection = selected or ""
        if collection and descriptions.get(collection):
            st.caption(descriptions[collection])
    else:
        collection = st.text_input("Collection (indexed doc_id)", value="")
        st.caption("No collections yet — ingest a PDF first.")

    st.button("➕ New session", use_container_width=True, on_click=_start_new_session)

    st.divider()
    st.subheader("Conversations")
    # Most recent first.
    for sid in reversed(list(st.session_state.conversations)):
        is_active = sid == st.session_state.active_session
        st.button(
            ("🟢 " if is_active else "💬 ") + _conversation_title(
                st.session_state.conversations[sid], sid
            ),
            key=f"open_{sid}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            on_click=_open_session,
            args=(sid,),
        )

    st.caption("The API must be running and the collection already ingested.")

# --- Active conversation -----------------------------------------------------
active = st.session_state.active_session
messages = st.session_state.conversations[active]
st.caption(f"Session: `{active}`")

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            st.caption(_meta_caption(msg["meta"]))

# --- Input + API call -------------------------------------------------------
if question := st.chat_input("Ask about the document…"):
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not collection.strip():
            st.error("Please set a collection in the sidebar first.")
            messages.append({"role": "assistant", "content": "⚠️ No collection set."})
        else:
            try:
                with st.spinner("Thinking…"):
                    data = _ask(base_url, question, collection, active)
            except httpx.HTTPStatusError as exc:
                err = f"⚠️ API returned {exc.response.status_code}: {exc.response.text[:300]}"
                st.error(err)
                messages.append({"role": "assistant", "content": err})
            except httpx.HTTPError as exc:
                err = f"⚠️ Could not reach the API at {base_url}. Is it running? ({exc})"
                st.error(err)
                messages.append({"role": "assistant", "content": err})
            else:
                answer = data.get("final_answer", "")
                meta = {
                    "is_grounded": data.get("is_grounded", False),
                    "confidence": data.get("confidence", 0.0),
                    "citations": data.get("citations", []),
                }
                st.markdown(answer)
                st.caption(_meta_caption(meta))
                messages.append({"role": "assistant", "content": answer, "meta": meta})

    # Refresh so the sidebar title (derived from the first question) updates.
    st.rerun()
