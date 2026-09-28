"""Kokoro TTS provider — local, CPU-friendly (Kokoro-82M).

Runs fully offline on **CPU** (no GPU/CUDA). The model (~330 MB) is downloaded from
Hugging Face on first use and cached under ``~/.cache/huggingface``. English only,
with smooth prosody that suits narration.

Kokoro produces 24 kHz float32 audio per sentence-ish segment, so it maps cleanly onto
the shared PCM/WAV helpers and streams segment-by-segment. Inference is synchronous and
CPU-bound, so it runs on a dedicated thread pool sized by ``max_concurrency`` (default
1): extra requests wait their turn instead of all fighting over the CPU, and they
never occupy threads of the default executor that the job store's SQLite calls use.

Install:  pip install -e ".[local]"
Docs:     https://github.com/hexgrad/kokoro
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import importlib.metadata
import importlib.util
import logging
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import Executor, ThreadPoolExecutor

from ._audio import (
    HAS_FFMPEG,
    NATIVE_FORMATS,
    ffmpeg_encode,
    pcm_to_wav,
    wav_stream_header,
)
from .base import Gender, ProviderError, SynthOpts, UnsupportedFormat, Voice

log = logging.getLogger("app.providers.kokoro")

# American (a) and British (b) English voices shipped with Kokoro; the second letter
# is the voice's gender (f/m). Within each gender, voices are ordered roughly by the
# upstream quality grades so automatic podcast casting picks the best ones first.
KOKORO_VOICES = [
    "af_heart", "af_bella", "af_nicole", "bf_emma", "af_aoede", "af_kore", "af_sarah",
    "af_alloy", "af_nova", "af_sky", "af_jessica", "af_river",
    "bf_isabella", "bf_alice", "bf_lily",
    "am_michael", "am_fenrir", "am_puck", "bm_george", "bm_fable", "am_echo", "am_eric",
    "am_liam", "am_onyx", "am_adam", "bm_lewis", "bm_daniel", "am_santa",
]


@functools.cache
def _kokoro_version() -> str:
    try:
        return importlib.metadata.version("kokoro")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def is_installed() -> bool:
    """True if the ``kokoro`` package is importable (no heavy import performed)."""
    return importlib.util.find_spec("kokoro") is not None


def voice_gender(voice: str) -> Gender | None:
    """Gender encoded in a Kokoro voice id (``af_*`` female, ``am_*`` male)."""
    return {"f": "female", "m": "male"}.get(voice[1:2])


def voice_language(voice: str) -> str:
    """Locale encoded in a Kokoro voice id (``b*`` British, otherwise American)."""
    return "en-GB" if voice[:1] == "b" else "en-US"


def _lang_code_for(voice: str) -> str:
    # Voice prefix: 'b' => British English, otherwise American English.
    return "b" if voice[:1] == "b" else "a"


def _floats_to_pcm16(audio: object) -> bytes:
    """Convert a float32 waveform in [-1, 1] to little-endian 16-bit PCM bytes."""
    import numpy as np

    arr = np.asarray(audio, dtype="float32")
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype("<i2").tobytes()


async def _aiter_blocking(
    gen_factory: Callable[[], Iterator[bytes]], executor: Executor
) -> AsyncIterator[bytes]:
    """Bridge a blocking generator to an async iterator.

    The producer runs on ``executor`` and hands items to the event loop with
    ``call_soon_threadsafe``, so waiting for the next item costs no thread. When the
    consumer stops early (client disconnected, task cancelled), a stop flag makes the
    producer quit at its next item instead of blocking forever.
    """
    loop = asyncio.get_running_loop()
    items: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()
    done = object()

    def push(item: object) -> None:
        try:
            loop.call_soon_threadsafe(items.put_nowait, item)
        except RuntimeError:  # event loop already closed; nobody is listening
            stop.set()

    def run() -> None:
        try:
            for item in gen_factory():
                if stop.is_set():
                    log.debug("producer stopped: consumer went away")
                    return
                push(item)
        except Exception as exc:  # noqa: BLE001 - propagate to the consumer
            push(exc)
        finally:
            push(done)

    ctx = contextvars.copy_context()  # keep the request id on the worker's log lines
    loop.run_in_executor(executor, ctx.run, run)
    try:
        while True:
            item = await items.get()
            if item is done:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop.set()


class KokoroProvider:
    """TTSProvider backed by the local Kokoro-82M model (CPU)."""

    name = "kokoro"
    native_formats = NATIVE_FORMATS

    def __init__(self, default_voice: str = "af_heart", max_concurrency: int = 1) -> None:
        self._default_voice = default_voice
        self._pipelines: dict[str, object] = {}  # lang_code -> KPipeline (lazy)
        self._pipeline_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, max_concurrency), thread_name_prefix="kokoro"
        )

    @property
    def default_voice(self) -> str:
        return self._default_voice

    def cache_tag(self, opts: SynthOpts) -> str:
        """Kokoro's package version: upgrading the model/code misses the cache."""
        return f"kokoro:{_kokoro_version()}"

    def can_produce(self, fmt: str) -> bool:
        return fmt in NATIVE_FORMATS or HAS_FFMPEG

    def _pipeline(self, lang_code: str) -> object:
        pipe = self._pipelines.get(lang_code)
        if pipe is not None:
            return pipe
        # Loading takes tens of seconds; the lock stops two first requests from each
        # loading their own copy of the model.
        with self._pipeline_lock:
            pipe = self._pipelines.get(lang_code)
            if pipe is None:
                try:
                    from kokoro import KPipeline
                except ImportError as exc:  # pragma: no cover - env-dependent
                    raise RuntimeError(
                        'Kokoro is not installed. Run: pip install -e ".[local]"'
                    ) from exc
                log.info("loading Kokoro pipeline (lang=%s)", lang_code)
                try:
                    pipe = KPipeline(lang_code=lang_code)
                except Exception as exc:  # download/load failure: report, don't crash
                    log.exception("Kokoro model failed to load")
                    raise ProviderError(
                        self.name, f"could not load the model ({exc})", kind="upstream"
                    ) from exc
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
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        pcm = await loop.run_in_executor(
            self._executor, ctx.run, self._collect_pcm, text, v, opts.speed
        )
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
        async for pcm in _aiter_blocking(
            lambda: self._segments(text, v, opts.speed), self._executor
        ):
            yield pcm

    async def list_voices(self) -> list[Voice]:
        return [
            Voice(
                id=v,
                name=v,
                provider=self.name,
                language=voice_language(v),
                gender=voice_gender(v),
            )
            for v in KOKORO_VOICES
        ]

    async def _encode(self, pcm: bytes, fmt: str) -> bytes:
        if fmt == "pcm":
            return pcm
        if fmt == "wav":
            return pcm_to_wav(pcm)
        if not HAS_FFMPEG:
            raise UnsupportedFormat(fmt, self.name)
        return await ffmpeg_encode(pcm, fmt)
