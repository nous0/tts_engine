"""Speech synthesis routes.

POST /v1/speech         one-shot synthesis.
POST /v1/speech/stream  chunked audio, sentence-by-sentence.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

from app.core.engine import TTSEngine
from app.core.providers.base import SynthOpts, UnsupportedFormat
from app.models.schemas import SpeechRequest, SpeechStreamRequest

router = APIRouter(prefix="/v1", tags=["speech"])
log = logging.getLogger("app.routes.speech")

_MEDIA_TYPES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "pcm": "audio/L16",
}


def _engine(request: Request) -> TTSEngine:
    return request.app.state.engine


@router.post(
    "/speech",
    responses={200: {"content": {"audio/mpeg": {}}, "description": "Synthesized audio."}},
)
async def create_speech(req: SpeechRequest, request: Request) -> Response:
    # Provider/format/voice/upstream errors are mapped to statuses in app/main.py.
    opts = SynthOpts(format=req.format, speed=req.speed, instructions=req.instructions)
    audio = await _engine(request).synthesize(
        req.text, voice=req.voice, provider=req.provider, opts=opts
    )

    media_type = _MEDIA_TYPES.get(req.format, "application/octet-stream")
    return Response(
        content=audio,
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="speech.{req.format}"'},
    )


async def _stream_body(first: bytes, rest: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Send the already-produced first chunk, then the rest of the stream.

    A failure after the 200 went out can't change the status code. Writing an error
    message into the stream would be played as noise, so the error is logged and
    re-raised instead: the server drops the connection and the client sees an
    incomplete response.
    """
    try:
        if first:
            yield first
        async for chunk in rest:
            yield chunk
    except Exception:
        log.warning("speech stream aborted after it started", exc_info=True)
        raise
    finally:
        await rest.aclose()


@router.post(
    "/speech/stream",
    responses={
        200: {
            "content": {"audio/mpeg": {}},
            "description": "Chunked audio stream synthesized sentence-by-sentence.",
        }
    },
)
async def create_speech_stream(
    req: SpeechStreamRequest, request: Request
) -> StreamingResponse:
    engine = _engine(request)
    # Validate up-front so an unconfigured provider or unsupported format
    # returns a clean 400 before we commit to a chunked 200 response.
    provider = engine.get_provider(req.provider)
    can_produce = getattr(provider, "can_produce", None)
    if can_produce is not None and not can_produce(req.format):
        raise UnsupportedFormat(req.format, provider.name)

    opts = SynthOpts(format=req.format, speed=req.speed, instructions=req.instructions)
    chunks = engine.synthesize_stream(
        req.text,
        voice=req.voice,
        provider=req.provider,
        opts=opts,
        max_sentence_length=req.max_sentence_length,
    )
    # Wait for the first audio chunk before sending headers, so the most common
    # failures (bad voice or format, provider down or timing out) still get a real
    # error status (mapped in app/main.py) instead of a 200 and a broken stream.
    try:
        first = await anext(chunks)
    except StopAsyncIteration:
        first = b""
    except BaseException:
        await chunks.aclose()
        raise

    media_type = _MEDIA_TYPES.get(req.format, "application/octet-stream")
    return StreamingResponse(
        _stream_body(first, chunks),
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="speech_stream.{req.format}"',
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering for low latency
        },
    )
