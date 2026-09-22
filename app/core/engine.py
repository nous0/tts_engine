"""Engine orchestrator: routes requests to a provider and applies defaults.

Covers one-shot synthesis, sentence-by-sentence streaming, and voice listing.
Podcast assembly lives in ``podcast.py`` and uses the engine for each turn.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from .chunker import split_sentences
from .providers.base import SynthOpts, TTSProvider, Voice


class ProviderNotConfigured(Exception):
    """Raised when a requested provider has no configured credentials."""

    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = available
        hint = ", ".join(available) if available else "none configured"
        super().__init__(f"Provider '{name}' is not configured (available: {hint}).")


class TTSEngine:
    def __init__(
        self,
        providers: dict[str, TTSProvider],
        default_provider: str,
        default_voice: str,
    ) -> None:
        self._providers = providers
        self._default_provider = default_provider
        self._default_voice = default_voice

    @property
    def default_provider(self) -> str:
        return self._default_provider

    def provider_names(self) -> list[str]:
        """Names of the registered (usable) providers, sorted."""
        return sorted(self._providers)

    def get_provider(self, name: str | None = None) -> TTSProvider:
        name = name or self._default_provider
        provider = self._providers.get(name)
        if provider is None:
            raise ProviderNotConfigured(name, sorted(self._providers))
        return provider

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        provider: str | None = None,
        opts: SynthOpts | None = None,
    ) -> bytes:
        p = self.get_provider(provider)
        return await p.synthesize(text, voice or self._default_voice, opts or SynthOpts())

    async def synthesize_stream(
        self,
        text: str,
        voice: str | None = None,
        provider: str | None = None,
        opts: SynthOpts | None = None,
        max_sentence_length: int = 200,
    ) -> AsyncIterator[bytes]:
        """Stream audio by synthesizing the text sentence-by-sentence.

        The text is split into sentence-ish chunks; each chunk is streamed
        through the provider's ``synthesize_stream`` so audio bytes are yielded
        as soon as they are produced. This lets a chatbot start playing the
        first sentence while later sentences are still being synthesized.
        """
        p = self.get_provider(provider)
        eff_voice = voice or self._default_voice
        eff_opts = opts or SynthOpts()
        for sentence in split_sentences(text, max_length=max_sentence_length):
            async for chunk in p.synthesize_stream(sentence, eff_voice, eff_opts):
                yield chunk

    async def list_voices(self, provider: str | None = None) -> list[Voice]:
        p = self.get_provider(provider)
        return await p.list_voices()
