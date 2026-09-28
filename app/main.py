"""FastAPI application: wiring, lifespan, and router registration."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import podcast, speech, voices
from app.config import get_settings
from app.core.cache import FileCache
from app.core.engine import InvalidRequest, ProviderNotConfigured, TTSEngine
from app.core.jobs import JobStore
from app.core.providers.base import ProviderError, UnsupportedFormat
from app.core.providers.registry import build_registry
from app.logging_setup import configure_logging, new_request_id, request_id_var

log = logging.getLogger("app.http")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    providers = build_registry(settings)
    cache = None
    if settings.cache_enabled:
        cache = FileCache(settings.cache_dir, settings.cache_max_mb * 1024 * 1024)
        await cache.prune()
    app.state.engine = TTSEngine(
        providers=providers,
        default_provider=settings.default_provider,
        default_voice=settings.default_voice,
        cache=cache,
    )

    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    app.state.output_dir = output_dir

    jobs = JobStore(settings.jobs_db, max_concurrent=settings.max_concurrent_jobs)
    await jobs.connect()
    await jobs.recover_stale()
    jobs.start_maintenance(settings.jobs_retention_days, output_dir)
    app.state.jobs = jobs
    logging.getLogger("app").info(
        "started: providers=%s default=%s", sorted(providers), settings.default_provider
    )
    try:
        yield
    finally:
        await jobs.close()


app = FastAPI(title="TTS Engine", version="0.1.0", lifespan=lifespan)
app.include_router(speech.router)
app.include_router(podcast.router)
app.include_router(voices.router)


# How each ProviderError kind reaches the client.
_PROVIDER_ERROR_STATUS = {
    "bad_request": 400,
    "auth": 502,
    "upstream": 502,
    "rate_limited": 503,
    "timeout": 504,
}


@app.exception_handler(ProviderError)
async def provider_error_handler(request: Request, exc: ProviderError) -> JSONResponse:
    headers = {}
    if exc.retry_after is not None:
        headers["Retry-After"] = str(max(1, round(exc.retry_after)))
    return JSONResponse(
        status_code=_PROVIDER_ERROR_STATUS.get(exc.kind, 502),
        content={"detail": str(exc), "kind": exc.kind},
        headers=headers,
    )


@app.exception_handler(ProviderNotConfigured)
@app.exception_handler(UnsupportedFormat)
@app.exception_handler(InvalidRequest)
async def bad_request_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Tag the request with an id, echo it back, and log one line per request."""
    rid = new_request_id(request.headers.get("x-request-id"))
    token = request_id_var.set(rid)
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        # For streaming responses this is time-to-headers, not the full stream.
        log.info(
            "%s %s -> %s in %.0f ms",
            request.method,
            request.url.path,
            status,
            (time.perf_counter() - started) * 1000,
        )
        request_id_var.reset(token)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
