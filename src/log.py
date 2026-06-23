"""Logging configuration for the agentic RAG system."""

import logging
import sys

from colorama import Fore, Style, init

init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Colorize log levels for terminal readability."""

    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        color = self.COLORS.get(levelname, Fore.WHITE)
        record.levelname = f"{color}{levelname}{Style.RESET_ALL}"
        return super().format(record)


_NOISY_LIBS = [
    "httpcore", "httpx", "openai", "chromadb", "urllib3",
    "langsmith", "anthropic", "hpack", "h2",
]


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger once at application startup."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Force UTF-8 on the console so logging non-ASCII text (Turkish section
    # titles, queries) never crashes with UnicodeEncodeError on a default
    # Windows (cp1252) terminal. Best-effort: reconfigure exists on 3.7+ TextIO.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

    handler = logging.StreamHandler(sys.stdout)
    formatter = ColoredFormatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Keep third-party HTTP/vector-store libraries quiet regardless of app level.
    for lib in _NOISY_LIBS:
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger for a module.

    Usage: logger = get_logger(__name__)
    """
    return logging.getLogger(name)
