"""Tests for the Kokoro provider using a fake pipeline (no model download).

A fake KPipeline is injected into the provider's cache so the real ``kokoro``
package and its model are never needed.
"""

from __future__ import annotations

import numpy as np

from app.core.providers import kokoro as kk
from app.core.providers.base import SynthOpts
from app.core.providers.kokoro import KokoroProvider


class FakePipeline:
    """Mimics kokoro.KPipeline: callable yielding (graphemes, phonemes, audio)."""

    def __init__(self, segments: list[np.ndarray]):
        self._segments = segments
        self.calls: list[dict] = []

    def __call__(self, text, voice, speed):
        self.calls.append({"text": text, "voice": voice, "speed": speed})
        for seg in self._segments:
            yield ("g", "p", seg)


def _provider_with(segments, lang="a") -> KokoroProvider:
    p = KokoroProvider()
    p._pipelines[lang] = FakePipeline(segments)  # bypass real model load
    return p


def test_lang_code_for():
    assert kk._lang_code_for("af_heart") == "a"
    assert kk._lang_code_for("am_adam") == "a"
    assert kk._lang_code_for("bf_emma") == "b"
    assert kk._lang_code_for("bm_george") == "b"


def test_floats_to_pcm16_scaling():
    pcm = kk._floats_to_pcm16(np.array([0.0, 1.0, -1.0], dtype="float32"))
    samples = np.frombuffer(pcm, dtype="<i2")
    assert list(samples) == [0, 32767, -32767]


def test_floats_to_pcm16_clips_out_of_range():
    pcm = kk._floats_to_pcm16(np.array([2.0, -2.0], dtype="float32"))
    samples = np.frombuffer(pcm, dtype="<i2")
    assert list(samples) == [32767, -32767]


def test_voices_are_english():
    provider = KokoroProvider()
    import asyncio

    voices = asyncio.run(provider.list_voices())
    assert len(voices) == len(kk.KOKORO_VOICES)
    assert any(v.id == "af_heart" for v in voices)
    assert all(v.provider == "kokoro" for v in voices)
    by_id = {v.id: v for v in voices}
    assert (by_id["af_heart"].gender, by_id["af_heart"].language) == ("female", "en-US")
    assert (by_id["am_michael"].gender, by_id["am_michael"].language) == ("male", "en-US")
    assert (by_id["bf_emma"].gender, by_id["bf_emma"].language) == ("female", "en-GB")
    assert (by_id["bm_george"].gender, by_id["bm_george"].language) == ("male", "en-GB")
    # Every voice id encodes a gender, so none is left untagged.
    assert all(v.gender in ("female", "male") for v in voices)
    assert len(set(kk.KOKORO_VOICES)) == len(kk.KOKORO_VOICES)


def test_can_produce_matches_environment():
    provider = KokoroProvider()
    assert provider.can_produce("wav") is True
    assert provider.can_produce("pcm") is True
    assert provider.can_produce("mp3") is kk.HAS_FFMPEG


async def test_synthesize_pcm_concatenates_segments():
    seg1 = np.array([0.0, 1.0], dtype="float32")
    seg2 = np.array([-1.0], dtype="float32")
    provider = _provider_with([seg1, seg2])

    pcm = await provider.synthesize("hello", "af_heart", SynthOpts(format="pcm"))
    samples = np.frombuffer(pcm, dtype="<i2")
    assert list(samples) == [0, 32767, -32767]


async def test_synthesize_wav_wraps_pcm():
    provider = _provider_with([np.array([0.5], dtype="float32")])
    wav = await provider.synthesize("hi", "af_heart", SynthOpts(format="wav"))
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"


async def test_stream_wav_emits_header_then_segments():
    provider = _provider_with(
        [np.array([0.1], dtype="float32"), np.array([0.2], dtype="float32")]
    )
    chunks = [
        chunk
        async for chunk in provider.synthesize_stream(
            "hi there", "af_heart", SynthOpts(format="wav")
        )
    ]
    # First chunk is the streaming WAV header; then one PCM chunk per segment.
    assert chunks[0][:4] == b"RIFF"
    assert len(chunks) == 3
