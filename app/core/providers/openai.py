"""OpenAI TTS provider.

Uses the official async SDK's ``audio.speech.create`` endpoint, and its streaming
response variant for ``synthesize_stream``.

Retries come from the SDK itself (connection errors, 408/409/429/5xx, with backoff);
this module only sets a sane timeout (the SDK default is 600 s) and translates SDK
errors into ``ProviderError`` so the API can answer with a meaningful status.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import httpx
import openai
from openai import AsyncOpenAI

from .base import Gender, ProviderError, ProviderErrorKind, SynthOpts, Voice

log = logging.getLogger("app.providers.openai")

# OpenAI does not expose a "list voices" endpoint, so we ship a static catalog.
_OPENAI_VOICES = [
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
]

# OpenAI publishes no gender metadata; these are perceived genders, used only to cast
# podcast speakers. ``alloy`` sounds neutral and is left untagged.
_OPENAI_GENDERS: dict[str, Gender] = {
    "coral": "female",
    "nova": "female",
    "sage": "female",
    "shimmer": "female",
    "ash": "male",
    "ballad": "male",
    "echo": "male",
    "fable": "male",
    "onyx": "male",
}


class OpenAIProvider:
    """TTSProvider implementation backed by the OpenAI audio API."""

    name = "openai"

    def __init__(
        self,
        api_key: str,
        default_model: str = "gpt-4o-mini-tts",
        default_voice: str = "alloy",
        timeout: float = 30.0,
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            timeout=httpx.Timeout(timeout, connect=min(5.0, timeout)),
            max_retries=max_retries,
            http_client=http_client,
        )
        self._default_model = default_model
        self.default_voice = default_voice

    async def synthesize(self, text: str, voice: str, opts: SynthOpts) -> bytes:
        model = opts.model or self._default_model
        kwargs: dict[str, object] = {}
        if opts.instructions:
            kwargs["instructions"] = opts.instructions
        # `speed` is only accepted by the tts-1 family, not gpt-4o-mini-tts.
        if opts.speed != 1.0 and model.startswith("tts-1"):
            kwargs["speed"] = opts.speed

        try:
            response = await self._client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format=opts.format,
                **kwargs,
            )
        except (openai.APIError, httpx.HTTPError) as exc:
            raise self._translate(exc) from exc
        return response.content

    async def synthesize_stream(
        self, text: str, voice: str, opts: SynthOpts
    ) -> AsyncIterator[bytes]:
        """Stream the synthesized audio in chunks as the API produces them."""
        model = opts.model or self._default_model
        kwargs: dict[str, object] = {}
        if opts.instructions:
            kwargs["instructions"] = opts.instructions
        if opts.speed != 1.0 and model.startswith("tts-1"):
            kwargs["speed"] = opts.speed

        try:
            async with self._client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice,
                input=text,
                response_format=opts.format,
                **kwargs,
            ) as response:
                async for chunk in response.iter_bytes():
                    if chunk:
                        yield chunk
        except (openai.APIError, httpx.HTTPError) as exc:
            # httpx errors can surface directly while the body is being read.
            raise self._translate(exc) from exc

    async def list_voices(self) -> list[Voice]:
        return [
            Voice(
                id=v,
                name=v.capitalize(),
                provider=self.name,
                gender=_OPENAI_GENDERS.get(v),
            )
            for v in _OPENAI_VOICES
        ]

    def _translate(self, exc: Exception) -> ProviderError:
        """Map an SDK/httpx error to a ProviderError with the right ``kind``."""
        kind: ProviderErrorKind = "upstream"
        retry_after: float | None = None
        if isinstance(exc, (openai.APITimeoutError, httpx.TimeoutException)):
            kind = "timeout"
        elif isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            kind = "auth"
        elif isinstance(exc, openai.RateLimitError):
            kind = "rate_limited"
            retry_after = _retry_after(exc.response)
        elif isinstance(
            exc,
            (openai.BadRequestError, openai.NotFoundError, openai.UnprocessableEntityError),
        ):
            kind = "bad_request"
        message = getattr(exc, "message", None) or str(exc) or type(exc).__name__
        err = ProviderError(self.name, message[:300], kind=kind, retry_after=retry_after)
        log.warning("%s", err)
        return err


def _retry_after(response: httpx.Response | None) -> float | None:
    if response is None:
        return None
    try:
        return float(response.headers["retry-after"])
    except (KeyError, ValueError):
        return None
