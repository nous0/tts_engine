"""Speech synthesis routes.

Phase 1: POST /v1/speech (one-shot).
Phase 2: POST /v1/speech/stream (chunked audio, sentence-by-sentence).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from app.core.engine import ProviderNotConfigured, TTSEngine
from app.core.providers.base import SynthOpts, UnsupportedFormat
from app.models.schemas import SpeechRequest, SpeechStreamRequest

router = APIRouter(prefix="/v1", tags=["speech"])

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
    opts = SynthOpts(format=req.format, speed=req.speed, instructions=req.instructions)
    try:
        audio = await _engine(request).synthesize(
            req.text, voice=req.voice, provider=req.provider, opts=opts
        )
    except (ProviderNotConfigured, UnsupportedFormat) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    media_type = _MEDIA_TYPES.get(req.format, "application/octet-stream")
    return Response(
        content=audio,
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="speech.{req.format}"'},
    )


async def _stream_audio(
    engine: TTSEngine, req: SpeechStreamRequest
) -> AsyncIterator[bytes]:
    opts = SynthOpts(format=req.format, speed=req.speed, instructions=req.instructions)
    try:
        async for chunk in engine.synthesize_stream(
            req.text,
            voice=req.voice,
            provider=req.provider,
            opts=opts,
            max_sentence_length=req.max_sentence_length,
        ):
            yield chunk
    except ProviderNotConfigured as exc:
        # Streaming has already started (200 + headers) so we cannot turn this
        # into a clean 400; surface the message as a trailing error chunk and
        # stop. Configured providers are validated up-front in the route below
        # to make this path effectively unreachable.
        yield str(exc).encode()
        return


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
    # Validate up-front so an unconfigured provider or unsupported format
    # returns a clean 400 before we commit to a chunked 200 response.
    try:
        provider = _engine(request).get_provider(req.provider)
    except ProviderNotConfigured as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    can_produce = getattr(provider, "can_produce", None)
    if can_produce is not None and not can_produce(req.format):
        raise HTTPException(
            status_code=400, detail=str(UnsupportedFormat(req.format, provider.name))
        )

    media_type = _MEDIA_TYPES.get(req.format, "application/octet-stream")
    return StreamingResponse(
        _stream_audio(_engine(request), req),
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="speech_stream.{req.format}"',
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering for low latency
        },
    )
