"""FastAPI application: wiring, lifespan, and router registration."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.routes import podcast, speech
from app.config import get_settings
from app.core.engine import TTSEngine
from app.core.jobs import JobStore
from app.core.providers.registry import build_registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    providers = build_registry(settings)
    app.state.engine = TTSEngine(
        providers=providers,
        default_provider=settings.default_provider,
        default_voice=settings.default_voice,
    )

    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    app.state.output_dir = output_dir

    jobs = JobStore(settings.jobs_db)
    await jobs.connect()
    app.state.jobs = jobs
    try:
        yield
    finally:
        await jobs.close()


app = FastAPI(title="TTS Engine", version="0.1.0", lifespan=lifespan)
app.include_router(speech.router)
app.include_router(podcast.router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
