"""WebSocket /v1/speech/ws: speak LLM output while it is still being generated.

Protocol (JSON text messages, plus binary audio from the server):

client -> server
    {"type": "start", "provider"?, "voice"?, "speed"?, "instructions"?,
     "fast_start"?: true, "cache"?: true, "normalize"?: true,
     "max_sentence_length"?: 200}                                      (first message)
    {"type": "text", "text": "..."}     LLM token deltas, any size
    {"type": "flush"}                   end of the turn: speak what's left
    {"type": "cancel"}                  barge-in: stop speaking now

server -> client
    {"type": "ready", "sample_rate": 24000, "channels": 1, "sample_width": 2,
     "format": "pcm", "provider", "voice"}
    {"type": "sentence", "index", "turn", "chars", "cached"}   before each sentence's audio
    <binary>                            raw PCM, s16le mono at sample_rate
    {"type": "done", "turn"}            the turn's last audio has been sent
    {"type": "cancelled"}               after a cancel; all pending turns are dropped
    {"type": "error", "detail", "status", "kind"?, "turn"?}

A bad ``start`` (unknown provider/voice, malformed) gets an error and the connection
closes with code 1008. Other errors keep the connection open: a failing sentence ends
its turn with an ``error`` that carries the turn number, and a bad message just gets an
``error`` reply. Browser connections (with an ``Origin`` header) are refused unless the
origin is listed in ``WS_ALLOWED_ORIGINS``, so a random web page can't drive a local
server. Server-side clients send no Origin and are allowed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError

from app.api.errors import CLIENT_ERRORS, error_payload
from app.config import Settings, get_settings
from app.core.engine import InvalidRequest, TTSEngine
from app.core.providers._audio import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH
from app.core.providers.base import SynthOpts
from app.core.session import (
    Audio,
    Cancelled,
    SentenceStarted,
    SpeechSession,
    TurnDone,
    TurnFailed,
)
from app.logging_setup import new_request_id, request_id_var
from app.models.schemas import WsStart

router = APIRouter(prefix="/v1", tags=["speech"])
log = logging.getLogger("app.routes.realtime")

START_TIMEOUT_S = 30.0

SendJson = Callable[[dict], Awaitable[None]]


@router.websocket("/speech/ws")
async def speech_ws(ws: WebSocket) -> None:
    settings = get_settings()
    request_id_var.set(new_request_id(ws.headers.get("x-request-id")))

    origin = ws.headers.get("origin")
    if origin and origin not in _allowed_origins(settings):
        log.warning("websocket refused: origin %s is not in WS_ALLOWED_ORIGINS", origin)
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await ws.accept()

    send_lock = asyncio.Lock()  # the receiver and the event forwarder both send

    async def send_json(payload: dict) -> None:
        async with send_lock:
            await ws.send_json(payload)

    async def send_bytes(data: bytes) -> None:
        async with send_lock:
            await ws.send_bytes(data)

    try:
        session = await _open_session(ws, send_json, settings)
    except WebSocketDisconnect:
        return
    if session is None:
        return

    log.info("websocket session started")
    forwarder = asyncio.create_task(_forward_events(session, send_json, send_bytes))
    try:
        await _handle_commands(ws, session, send_json)
    except WebSocketDisconnect:
        pass
    finally:
        forwarder.cancel()
        await asyncio.wait({forwarder})
        await session.close()
        log.info("websocket session closed after %d turn(s)", session.turn)


async def _open_session(
    ws: WebSocket, send_json: SendJson, settings: Settings
) -> SpeechSession | None:
    """Wait for ``start``, validate it, and open a session (None if refused)."""
    try:
        raw = await asyncio.wait_for(_receive_text(ws), START_TIMEOUT_S)
    except TimeoutError:
        await _refuse(ws, send_json, "No 'start' message received in time.")
        return None
    if raw is None:
        await _refuse(ws, send_json, "The first message must be a JSON 'start' message.")
        return None
    try:
        start = WsStart.model_validate_json(raw)
    except ValidationError as exc:
        await _refuse(ws, send_json, f"Invalid 'start' message: {_first_error(exc)}")
        return None

    engine: TTSEngine = ws.app.state.engine
    try:
        provider = engine.get_provider(start.provider)
        voice = await engine.resolve_voice(provider, start.voice)
    except CLIENT_ERRORS as exc:
        await _refuse(ws, send_json, str(exc))
        return None

    session = SpeechSession(
        engine,
        provider,
        voice,
        SynthOpts(format="pcm", speed=start.speed, instructions=start.instructions),
        fast_start=start.fast_start,
        use_cache=start.cache,
        normalize=start.normalize,
        max_turn_chars=settings.ws_max_turn_chars,
        max_sentence_length=start.max_sentence_length,
    )
    session.start()
    await send_json(
        {
            "type": "ready",
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "sample_width": SAMPLE_WIDTH,
            "format": "pcm",
            "provider": provider.name,
            "voice": voice,
        }
    )
    return session


async def _handle_commands(ws: WebSocket, session: SpeechSession, send_json: SendJson) -> None:
    while True:
        raw = await _receive_text(ws)
        if raw is None:
            await _error(send_json, "Binary messages aren't accepted; send JSON commands.")
            continue
        try:
            message = json.loads(raw)
            kind = message["type"]
        except (ValueError, TypeError, KeyError):
            await _error(send_json, 'Expected a JSON object with a "type" field.')
            continue

        if kind == "text":
            text = message.get("text")
            if not isinstance(text, str):
                await _error(send_json, "'text' messages need a string 'text' field.")
                continue
            try:
                session.feed(text)
            except InvalidRequest as exc:
                await _error(send_json, str(exc))
        elif kind == "flush":
            session.flush()
        elif kind == "cancel":
            await session.cancel()
        elif kind == "start":
            await _error(send_json, "The session has already started.")
        else:
            await _error(
                send_json, f"Unknown message type {kind!r}; expected text, flush or cancel."
            )


async def _forward_events(
    session: SpeechSession,
    send_json: SendJson,
    send_bytes: Callable[[bytes], Awaitable[None]],
) -> None:
    with contextlib.suppress(WebSocketDisconnect, RuntimeError):  # client already gone
        async for event in session.events():
            if isinstance(event, Audio):
                await send_bytes(event.pcm)
            elif isinstance(event, SentenceStarted):
                await send_json(
                    {
                        "type": "sentence",
                        "index": event.index,
                        "turn": event.turn,
                        "chars": event.chars,
                        "cached": event.cached,
                    }
                )
            elif isinstance(event, TurnDone):
                await send_json({"type": "done", "turn": event.turn})
            elif isinstance(event, TurnFailed):
                await send_json({"type": "error", "turn": event.turn,
                                 **error_payload(event.error)})
            elif isinstance(event, Cancelled):
                await send_json({"type": "cancelled"})


async def _receive_text(ws: WebSocket) -> str | None:
    """Next text message (None for a binary one); raises WebSocketDisconnect on close."""
    message = await ws.receive()
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", status.WS_1000_NORMAL_CLOSURE))
    return message.get("text")


async def _error(send_json: SendJson, detail: str, status_code: int = 400) -> None:
    await send_json({"type": "error", "detail": detail, "status": status_code})


async def _refuse(ws: WebSocket, send_json: SendJson, detail: str) -> None:
    log.info("websocket start refused: %s", detail)
    await _error(send_json, detail)
    await ws.close(code=status.WS_1008_POLICY_VIOLATION)


def _allowed_origins(settings: Settings) -> set[str]:
    return {o.strip().rstrip("/") for o in settings.ws_allowed_origins.split(",") if o.strip()}


def _first_error(exc: ValidationError) -> str:
    err = exc.errors()[0]
    where = ".".join(str(p) for p in err.get("loc", ())) or "message"
    return f"{where}: {err.get('msg', 'invalid')}"
