"""Load prompt strings from .txt files at import time."""

from pathlib import Path

_DIR = Path(__file__).parent


def _load(filename: str) -> str:
    return (_DIR / filename).read_text(encoding="utf-8").strip()


EVIDENCE_AGENT_PROMPT = _load("evidence.txt")
VALIDATION_SYSTEM_PROMPT = _load("validation.txt")
TRIAGE_SYSTEM_PROMPT = _load("triage.txt")

__all__ = ["EVIDENCE_AGENT_PROMPT", "VALIDATION_SYSTEM_PROMPT", "TRIAGE_SYSTEM_PROMPT"]
