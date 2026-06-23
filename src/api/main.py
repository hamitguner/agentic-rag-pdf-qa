"""FastAPI application: a REST surface over the agentic RAG pipeline.

Run with:
    uv run uvicorn src.api.main:app --reload

The compiled graph is built once at startup and shared across requests, so the
in-memory conversation checkpointer persists per session_id between calls.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import router
from src.config import settings
from src.log import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    setup_logging(settings.log_level)
    from src.agent.graph import build_graph

    app.state.graph = build_graph()
    logger.info("API ready — log_level=%s", settings.log_level)
    yield
    logger.info("API shutting down")


app = FastAPI(title="Agentic RAG API", lifespan=lifespan)
app.include_router(router)
