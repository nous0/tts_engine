"""Provider abstraction: the common interface every TTS backend implements.

Cloud providers (OpenAI, ElevenLabs, ...) and, later, local backends
(Piper/XTTS) implement this Protocol so the engine treats them uniformly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

# Audio container formats the engine understands.
AudioFormat = str  # "mp3" | "wav" | "opus" | "aac" | "flac" | "pcm"
Gender = Literal["female", "male"]
ProviderErrorKind = Literal["timeout", "rate_limited", "auth", "bad_request", "upstream"]


class UnsupportedFormat(Exception):
    """Raised when a provider cannot produce the requested audio format.

    Typically because a compressed format (mp3/opus/...) was requested from a
    PCM-native provider (e.g. Kokoro) and ffmpeg is not installed for transcoding.
    """

    def __init__(self, fmt: str, provider: str) -> None:
        self.fmt = fmt
        self.provider = provider
        super().__init__(
            f"Format '{fmt}' is not supported by provider '{provider}' without "
            f"ffmpeg. Use 'wav' or 'pcm', or install ffmpeg."
        )


class ProviderError(Exception):
    """A provider call failed: timeout, rate limit, bad credentials, bad input, or
    an upstream/network error. The API maps ``kind`` to an HTTP status (see
    ``app/main.py``), so a flaky backend never surfaces as an opaque 500.
    """

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        kind: ProviderErrorKind = "upstream",
        retry_after: float | None = None,
    ) -> None:
        self.provider = provider
        self.kind = kind
        self.retry_after = retry_after
        super().__init__(f"Provider '{provider}' {_KIND_TEXT[kind]}: {message}")


_KIND_TEXT: dict[str, str] = {
    "timeout": "timed out",
    "rate_limited": "is rate limiting requests",
    "auth": "rejected the server's credentials",
    "bad_request": "rejected the request",
    "upstream": "failed",
}


@dataclass(slots=True)
class SynthOpts:
    """Per-request synthesis options, provider-agnostic."""

    format: AudioFormat = "wav"
    speed: float = 1.0
    model: str | None = None  # override the provider's default model
    instructions: str | None = None  # style/tone hint (e.g. OpenAI gpt-4o-mini-tts)


@dataclass(slots=True)
class Voice:
    """A voice offered by a provider."""

    id: str
    name: str
    provider: str
    language: str | None = None
    gender: Gender | None = None  # perceived voice gender; None when neutral/unknown
    tags: list[str] = field(default_factory=list)


@runtime_checkable
class TTSProvider(Protocol):
    """Common interface for all TTS providers."""

    name: str

    async def synthesize(self, text: str, voice: str, opts: SynthOpts) -> bytes:
        """Synthesize the full clip and return encoded audio bytes."""
        ...

    def synthesize_stream(
        self, text: str, voice: str, opts: SynthOpts
    ) -> AsyncIterator[bytes]:
        """Stream encoded audio chunks as they are produced."""
        ...

    async def list_voices(self) -> list[Voice]:
        """Return the voices this provider offers."""
        ...
