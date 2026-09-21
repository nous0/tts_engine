"""Kokoro TTS provider — local, CPU-friendly (Kokoro-82M).

Runs fully offline on **CPU** (no GPU/CUDA). The model (~330 MB) is downloaded from
Hugging Face on first use and cached under ``~/.cache/huggingface``. English only,
with smooth prosody that suits narration.

Kokoro produces 24 kHz float32 audio per sentence-ish segment, so it maps cleanly onto
the shared PCM/WAV helpers and streams segment-by-segment. Inference is synchronous and
CPU-bound, so it is run in a worker thread to avoid blocking the event loop.

Install:  pip install -e ".[local]"
Docs:     https://github.com/hexgrad/kokoro
"""

from __future__ import annotations

import asyncio
import importlib.util
import queue
import threading
from collections.abc import AsyncIterator, Callable, Iterator

from ._audio import (
    HAS_FFMPEG,
    NATIVE_FORMATS,
    ffmpeg_encode,
    pcm_to_wav,
    wav_stream_header,
)
from .base import SynthOpts, UnsupportedFormat, Voice

# American (af_/am_) and British (bf_/bm_) English voices shipped with Kokoro.
KOKORO_VOICES = [
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]


def is_installed() -> bool:
    """True if the ``kokoro`` package is importable (no heavy import performed)."""
    return importlib.util.find_spec("kokoro") is not None


def _lang_code_for(voice: str) -> str:
    # Voice prefix: 'b' => British English, otherwise American English.
    return "b" if voice[:1] == "b" else "a"


def _floats_to_pcm16(audio: object) -> bytes:
    """Convert a float32 waveform in [-1, 1] to little-endian 16-bit PCM bytes."""
    import numpy as np

    arr = np.asarray(audio, dtype="float32")
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype("<i2").tobytes()


async def _aiter_blocking(gen_factory: Callable[[], Iterator[bytes]]) -> AsyncIterator[bytes]:
    """Bridge a blocking generator to an async iterator via a worker thread."""
    q: queue.Queue = queue.Queue(maxsize=8)
    done = object()

    def run() -> None:
        try:
            for item in gen_factory():
                q.put(item)
        except Exception as exc:  # noqa: BLE001 - propagate to the consumer
            q.put(exc)
        finally:
            q.put(done)

    threading.Thread(target=run, daemon=True).start()
    loop = asyncio.get_running_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is done:
            return
        if isinstance(item, Exception):
            raise item
        yield item


class KokoroProvider:
    """TTSProvider backed by the local Kokoro-82M model (CPU)."""

    name = "kokoro"
    native_formats = NATIVE_FORMATS

    def __init__(self, default_voice: str = "af_heart") -> None:
        self._default_voice = default_voice
        self._pipelines: dict[str, object] = {}  # lang_code -> KPipeline (lazy)

    def can_produce(self, fmt: str) -> bool:
        return fmt in NATIVE_FORMATS or HAS_FFMPEG

    def _pipeline(self, lang_code: str) -> object:
        pipe = self._pipelines.get(lang_code)
        if pipe is None:
            try:
                from kokoro import KPipeline
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise RuntimeError(
                    'Kokoro is not installed. Run: pip install -e ".[local]"'
                ) from exc
            pipe = KPipeline(lang_code=lang_code)
            self._pipelines[lang_code] = pipe
        return pipe

    def _segments(self, text: str, voice: str, speed: float) -> Iterator[bytes]:
        """Yield PCM bytes for each Kokoro output segment (blocking)."""
        pipeline = self._pipeline(_lang_code_for(voice))
        for result in pipeline(text, voice=voice, speed=speed):
            audio = result[-1]  # (graphemes, phonemes, audio)
            if audio is None:
                continue
            pcm = _floats_to_pcm16(audio)
            if pcm:
                yield pcm

    def _collect_pcm(self, text: str, voice: str, speed: float) -> bytes:
        return b"".join(self._segments(text, voice, speed))

    async def synthesize(self, text: str, voice: str, opts: SynthOpts) -> bytes:
        v = voice or self._default_voice
        pcm = await asyncio.to_thread(self._collect_pcm, text, v, opts.speed)
        return await self._encode(pcm, opts.format)

    async def synthesize_stream(
        self, text: str, voice: str, opts: SynthOpts
    ) -> AsyncIterator[bytes]:
        fmt = opts.format
        v = voice or self._default_voice
        if fmt not in NATIVE_FORMATS:
            if not HAS_FFMPEG:
                raise UnsupportedFormat(fmt, self.name)
            yield await self.synthesize(text, v, opts)
            return

        if fmt == "wav":
            yield wav_stream_header()
        async for pcm in _aiter_blocking(lambda: self._segments(text, v, opts.speed)):
            yield pcm

    async def list_voices(self) -> list[Voice]:
        return [
            Voice(id=v, name=v, provider=self.name, language="en") for v in KOKORO_VOICES
        ]

    async def _encode(self, pcm: bytes, fmt: str) -> bytes:
        if fmt == "pcm":
            return pcm
        if fmt == "wav":
            return pcm_to_wav(pcm)
        if not HAS_FFMPEG:
            raise UnsupportedFormat(fmt, self.name)
        return await ffmpeg_encode(pcm, fmt)
