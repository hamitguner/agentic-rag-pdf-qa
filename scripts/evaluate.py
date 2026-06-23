"""Run a question set through the pipeline and report answers.

Serves two purposes at once:
  - Demo output (Deliverable #4): a terminal log / markdown of N example Q&A.
  - Bonus evaluation: runs a labelled set (question + expected answer) so answers
    can be compared against ground truth.

The question file is JSON: a list of objects with ``question`` and
``expected_answer``; ``purpose`` (what it probes) is shown if present. Defaults
to the main committed set, ``eval/questions_llama3.json``.

Each question runs in its own session so they stay independent (no memory bleed).

Usage:
    # Main long-document set against the Llama 3 collection (already indexed):
    uv run scripts/evaluate.py --collection llama3_herd --out demo_output_llama3.md

    # Spot-check with a limit (prints to terminal unless --out is given):
    uv run scripts/evaluate.py --collection llama3_herd --limit 3

    # Multilingual + vision showcase (Turkish FOMC report):
    uv run scripts/evaluate.py --collection fomc_june \
        --questions eval/questions_fomc.json --limit 3 --out demo_output_fomc.md
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.log import get_logger, setup_logging

logger = get_logger(__name__)


def _default_questions_path() -> str | None:
    """Locate a default question set.

    Prefers the main committed set ``eval/questions_llama3.json``; falls back to a
    local ``gemini-code-*.json`` (gitignored scratch set) if present.
    """
    root = Path(__file__).parent.parent
    committed = root / "eval" / "questions_llama3.json"
    if committed.exists():
        return str(committed)
    matches = sorted(glob.glob(str(root / "gemini-code-*.json")))
    return matches[0] if matches else None


def load_questions(path: str) -> list[dict]:
    """Load and validate the question set."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit(f"Question file {path!r} must be a non-empty JSON list.")
    return data


def _format_report(rows: list[dict]) -> str:
    """Render the eval rows as a readable markdown report."""
    lines = ["# Evaluation / Demo Output\n", f"Document collection: `{rows[0]['collection']}`\n"]
    for r in rows:
        lines.append(f"\n## Q{r['id']}: {r['question']}\n")
        if r.get("purpose"):
            lines.append(f"_Probes:_ {r['purpose']}\n")
        if r.get("expected"):
            lines.append(f"**Expected:** {r['expected']}\n")
        lines.append(f"**Answer:** {r['answer']}\n")
        lines.append(
            f"**Grounded:** {r['is_grounded']} · "
            f"**Confidence:** {r['confidence']:.0%} · "
            f"**Citations:** {', '.join(r['citations']) or '—'}\n"
        )
    return "\n".join(lines)


def run_eval(collection: str, questions: list[dict], limit: int | None) -> list[dict]:
    """Run each question through the pipeline and collect results."""
    from src.agent.graph import build_graph, invoke_safely

    # research_node initializes the retriever from each turn's collection.
    graph = build_graph()

    selected = questions[:limit] if limit else questions
    rows: list[dict] = []

    for q in selected:
        qid = q.get("id", len(rows) + 1)
        question = q["question"]
        session = f"eval-{qid}"  # isolated session per question

        result = invoke_safely(
            graph,
            {"question": question, "collection": collection, "session_id": session},
            config={"configurable": {"thread_id": session}},
        )

        row = {
            "id": qid,
            "collection": collection,
            "question": question,
            "expected": q.get("expected_answer", ""),
            "purpose": q.get("purpose", ""),
            "answer": result.get("final_answer", ""),
            "is_grounded": result.get("is_grounded", False),
            "confidence": result.get("confidence", 0.0),
            "citations": result.get("citations", []),
        }
        rows.append(row)

        print(f"\n{'=' * 70}\nQ{qid}: {question}")
        if row["expected"]:
            print(f"{'-' * 70}\nEXPECTED: {row['expected']}")
        print(f"{'-' * 70}\nANSWER:   {row['answer']}")
        print(
            f"{'-' * 70}\nGrounded: {row['is_grounded']} | "
            f"Confidence: {row['confidence']:.0%} | "
            f"Citations: {', '.join(row['citations']) or '—'}\n{'=' * 70}"
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate / demo the RAG pipeline.")
    parser.add_argument("--collection", help="Collection to query (or to create from --pdf).")
    parser.add_argument("--pdf", help="Optional PDF to ingest before evaluating.")
    parser.add_argument("--description", help="Collection description (stored + injected).")
    parser.add_argument(
        "--reset", action="store_true", help="Wipe the target collection before ingesting --pdf."
    )
    parser.add_argument(
        "--questions",
        default=_default_questions_path(),
        help="Path to the question JSON (default: bundled gemini-code-*.json).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N questions.")
    parser.add_argument("--out", help="Optional path to write a markdown report.")
    args = parser.parse_args()

    setup_logging(settings.log_level)

    if not args.questions:
        raise SystemExit("No question file found. Pass --questions PATH.")

    collection = args.collection
    if args.pdf:
        from src.cli import ingest

        collection = ingest(
            args.pdf,
            collection=args.collection,
            description=args.description,
            reset=args.reset,
        )
    if not collection:
        raise SystemExit("--collection is required when --pdf is not given.")

    questions = load_questions(args.questions)
    rows = run_eval(collection, questions, args.limit)

    if args.out:
        Path(args.out).write_text(_format_report(rows), encoding="utf-8")
        logger.info("Wrote report to %s", args.out)


if __name__ == "__main__":
    main()
