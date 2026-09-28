"""Engine orchestrator: routes requests to a provider and applies defaults.

Covers one-shot synthesis, sentence-by-sentence streaming, and voice listing.
Podcast assembly lives in ``podcast.py`` and uses the engine for each turn.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import replace

from .cache import FileCache
from .chunker import split_sentences
from .providers._audio import wav_stream_header
from .providers.base import SynthOpts, TTSProvider, Voice

log = logging.getLogger("app.engine")

# Formats whose per-sentence files can't simply be concatenated into one stream.
STREAM_UNSUPPORTED_FORMATS = frozenset({"flac"})


class ProviderNotConfigured(Exception):
    """Raised when a requested provider has no configured credentials."""

    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = available
        hint = ", ".join(available) if available else "none configured"
        super().__init__(f"Provider '{name}' is not configured (available: {hint}).")


class InvalidRequest(ValueError):
    """The request itself can't be served (maps to HTTP 400)."""


class InvalidVoice(InvalidRequest):
    """The voice doesn't exist for the chosen provider."""

    def __init__(self, voice: str, provider: str) -> None:
        self.voice = voice
        self.provider = provider
        super().__init__(
            f"Unknown voice '{voice}' for provider '{provider}'. "
            f"See GET /v1/voices?provider={provider}."
        )


class TTSEngine:
    def __init__(
        self,
        providers: dict[str, TTSProvider],
        default_provider: str,
        default_voice: str,
        cache: FileCache | None = None,
    ) -> None:
        self._providers = providers
        self._default_provider = default_provider
        self._default_voice = default_voice
        self._cache = cache

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

    async def resolve_voice(self, provider: TTSProvider, voice: str | None) -> str:
        """The voice to use: the requested one (validated), else a sensible default.

        An explicit voice must exist in the provider's catalog; this also keeps
        arbitrary strings (paths, unknown names) away from backends that would try
        to load or download them. Without one, the first valid choice of: the global
        DEFAULT_VOICE (only for the default provider), the provider's own default
        voice, the first voice in its catalog.
        """
        catalog = [v.id for v in await provider.list_voices()]
        if voice:
            if catalog and voice not in catalog:
                raise InvalidVoice(voice, provider.name)
            return voice
        candidates = [
            self._default_voice if provider.name == self._default_provider else None,
            getattr(provider, "default_voice", None),
            *catalog[:1],
        ]
        for candidate in candidates:
            if candidate and (not catalog or candidate in catalog):
                return candidate
        raise InvalidRequest(f"Provider '{provider.name}' has no voice to use; pass 'voice'.")

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        provider: str | None = None,
        opts: SynthOpts | None = None,
    ) -> bytes:
        audio, _hit = await self.synthesize_cached(text, voice, provider, opts)
        return audio

    async def synthesize_cached(
        self,
        text: str,
        voice: str | None = None,
        provider: str | None = None,
        opts: SynthOpts | None = None,
    ) -> tuple[bytes, bool]:
        """Like ``synthesize``, also saying whether the audio came from the cache.

        The cache key uses the resolved voice and the provider's ``cache_tag`` (its
        real model/version), so defaults and model changes can't serve stale audio.
        Streaming is never cached.
        """
        p = self.get_provider(provider)
        eff_voice = await self.resolve_voice(p, voice)
        eff_opts = opts or SynthOpts()
        key = None
        if self._cache is not None:
            key = self._cache.make_key(
                provider=p.name,
                tag=_cache_tag(p, eff_opts),
                voice=eff_voice,
                format=eff_opts.format,
                speed=eff_opts.speed,
                instructions=eff_opts.instructions,
                text=text,
            )
            cached = await self._cache.get(key, eff_opts.format)
            if cached is not None:
                log.debug("cache hit: provider=%s voice=%s chars=%d", p.name, eff_voice,
                          len(text))
                return cached, True

        started = time.perf_counter()
        audio = await p.synthesize(text, eff_voice, eff_opts)
        log.info(
            "synthesized: provider=%s voice=%s format=%s chars=%d in %.0f ms",
            p.name,
            eff_voice,
            eff_opts.format,
            len(text),
            (time.perf_counter() - started) * 1000,
        )
        if self._cache is not None and key is not None:
            await self._cache.put(key, eff_opts.format, audio)
        return audio, False

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

        For ``wav`` the provider is asked for raw PCM and the engine sends a single
        streaming WAV header, just before the first audio chunk. Asking providers for
        WAV per sentence would put a header between sentences, which players render
        as a click. Nothing is yielded until the provider produces real audio, so a
        caller can wait for the first chunk to catch early failures.
        """
        p = self.get_provider(provider)
        eff_voice = await self.resolve_voice(p, voice)
        eff_opts = opts or SynthOpts()
        if eff_opts.format in STREAM_UNSUPPORTED_FORMATS:
            raise InvalidRequest(
                f"Format '{eff_opts.format}' can't be streamed; use /v1/speech for it, "
                f"or stream wav, pcm, mp3, opus or aac."
            )
        wav = eff_opts.format == "wav"
        provider_opts = replace(eff_opts, format="pcm") if wav else eff_opts
        header_sent = False
        for sentence in split_sentences(text, max_length=max_sentence_length):
            async for chunk in p.synthesize_stream(sentence, eff_voice, provider_opts):
                if wav and not header_sent:
                    header_sent = True
                    yield wav_stream_header()
                yield chunk

    async def list_voices(self, provider: str | None = None) -> list[Voice]:
        p = self.get_provider(provider)
        return await p.list_voices()


def _cache_tag(provider: TTSProvider, opts: SynthOpts) -> str:
    """What besides the inputs determines the audio: model name, package version."""
    tag = getattr(provider, "cache_tag", None)
    return tag(opts) if callable(tag) else provider.name
