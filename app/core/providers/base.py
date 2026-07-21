"""Provider abstraction: the common interface every TTS backend implements.

Cloud providers (OpenAI, ElevenLabs, ...) and, later, local backends
(Piper/XTTS) implement this Protocol so the engine treats them uniformly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# Audio container formats the engine understands.
AudioFormat = str  # "mp3" | "wav" | "opus" | "aac" | "flac" | "pcm"


@dataclass(slots=True)
class SynthOpts:
    """Per-request synthesis options, provider-agnostic."""

    format: AudioFormat = "mp3"
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
