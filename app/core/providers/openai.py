"""OpenAI TTS provider.

Uses the official async SDK's ``audio.speech.create`` endpoint, and its streaming
response variant for ``synthesize_stream``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from .base import Gender, SynthOpts, Voice

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

    def __init__(self, api_key: str, default_model: str = "gpt-4o-mini-tts") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._default_model = default_model

    async def synthesize(self, text: str, voice: str, opts: SynthOpts) -> bytes:
        model = opts.model or self._default_model
        kwargs: dict[str, object] = {}
        if opts.instructions:
            kwargs["instructions"] = opts.instructions
        # `speed` is only accepted by the tts-1 family, not gpt-4o-mini-tts.
        if opts.speed != 1.0 and model.startswith("tts-1"):
            kwargs["speed"] = opts.speed

        response = await self._client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format=opts.format,
            **kwargs,
        )
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
