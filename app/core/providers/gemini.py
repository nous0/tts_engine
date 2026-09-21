"""Google Gemini TTS provider (Gemini 2.5 Flash TTS).

Gemini's TTS models return **raw PCM** audio (24 kHz, 16-bit, mono). PCM/WAV/ffmpeg
handling lives in ``_audio`` and is shared with the local Kokoro provider. Style/tone
is steered via natural language in the prompt (``instructions``), e.g. "Say cheerfully".

Docs: https://ai.google.dev/gemini-api/docs/speech-generation
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from google import genai
from google.genai import types

from ._audio import (
    CHANNELS,
    HAS_FFMPEG,
    NATIVE_FORMATS,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    ffmpeg_encode,
    pcm_to_wav,
    wav_stream_header,
)
from .base import SynthOpts, UnsupportedFormat, Voice

# Re-exported so existing imports (and tests) can reach these via this module.
__all__ = [
    "GeminiProvider",
    "GEMINI_VOICES",
    "extract_pcm",
    "pcm_to_wav",
    "wav_stream_header",
    "ffmpeg_encode",
    "HAS_FFMPEG",
    "NATIVE_FORMATS",
    "SAMPLE_RATE",
    "CHANNELS",
    "SAMPLE_WIDTH",
]

# Prebuilt Gemini voices; the API expects these exact names.
GEMINI_VOICES = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
    "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
    "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
]


def extract_pcm(response: object, required: bool = True) -> bytes:
    """Pull raw PCM bytes out of a Gemini response (or stream chunk)."""
    parts = None
    try:
        parts = response.candidates[0].content.parts  # type: ignore[attr-defined]
    except (AttributeError, IndexError, TypeError):
        parts = None
    if parts:
        for part in parts:
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None)
            if data:
                return data
    if required:
        raise RuntimeError("Gemini returned no audio data.")
    return b""


class GeminiProvider:
    """TTSProvider backed by the Gemini TTS models."""

    name = "gemini"
    native_formats = NATIVE_FORMATS

    def __init__(
        self, api_key: str, default_model: str = "gemini-2.5-flash-preview-tts"
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._default_model = default_model

    def can_produce(self, fmt: str) -> bool:
        """Whether this provider can emit ``fmt`` in the current environment."""
        return fmt in NATIVE_FORMATS or HAS_FFMPEG

    def _config(self, voice: str) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        )

    def _contents(self, text: str, opts: SynthOpts) -> str:
        # Gemini steers style through the prompt itself, e.g. "Say cheerfully: ...".
        return f"{opts.instructions}: {text}" if opts.instructions else text

    async def synthesize(self, text: str, voice: str, opts: SynthOpts) -> bytes:
        model = opts.model or self._default_model
        response = await self._client.aio.models.generate_content(
            model=model, contents=self._contents(text, opts), config=self._config(voice)
        )
        return await self._encode(extract_pcm(response), opts.format)

    async def synthesize_stream(
        self, text: str, voice: str, opts: SynthOpts
    ) -> AsyncIterator[bytes]:
        fmt = opts.format
        if fmt not in NATIVE_FORMATS:
            if not HAS_FFMPEG:
                raise UnsupportedFormat(fmt, self.name)
            # Compressed formats can't be framed incrementally here; synthesize
            # the whole clip and emit it once (still one network round-trip).
            yield await self.synthesize(text, voice, opts)
            return

        model = opts.model or self._default_model
        if fmt == "wav":
            yield wav_stream_header()
        stream = await self._client.aio.models.generate_content_stream(
            model=model, contents=self._contents(text, opts), config=self._config(voice)
        )
        async for chunk in stream:
            pcm = extract_pcm(chunk, required=False)
            if pcm:
                yield pcm

    async def list_voices(self) -> list[Voice]:
        return [Voice(id=v, name=v, provider=self.name) for v in GEMINI_VOICES]

    async def _encode(self, pcm: bytes, fmt: str) -> bytes:
        if fmt == "pcm":
            return pcm
        if fmt == "wav":
            return pcm_to_wav(pcm)
        if not HAS_FFMPEG:
            raise UnsupportedFormat(fmt, self.name)
        return await ffmpeg_encode(pcm, fmt)
