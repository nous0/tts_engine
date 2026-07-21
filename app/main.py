"""FastAPI application: wiring, lifespan, and router registration."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import speech
from app.config import get_settings
from app.core.engine import TTSEngine
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
    yield


app = FastAPI(title="TTS Engine", version="0.1.0", lifespan=lifespan)
app.include_router(speech.router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
