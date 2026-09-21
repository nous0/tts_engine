"""Tests for the Gemini provider: PCM/WAV helpers, format gating, mocked synth.

No network calls — the genai client is replaced with a fake after construction.
"""

from __future__ import annotations

import io
import wave

import pytest

from app.core.providers import gemini as gm
from app.core.providers.base import SynthOpts, UnsupportedFormat
from app.core.providers.gemini import GeminiProvider

# A short slice of fake PCM (4 samples, 16-bit mono).
FAKE_PCM = b"\x01\x00\x02\x00\x03\x00\x04\x00"


def test_pcm_to_wav_roundtrips():
    data = gm.pcm_to_wav(FAKE_PCM)
    assert data[:4] == b"RIFF"
    assert data[8:12] == b"WAVE"
    with wave.open(io.BytesIO(data), "rb") as wf:
        assert wf.getnchannels() == gm.CHANNELS
        assert wf.getframerate() == gm.SAMPLE_RATE
        assert wf.getsampwidth() == gm.SAMPLE_WIDTH
        assert wf.readframes(wf.getnframes()) == FAKE_PCM


def test_wav_stream_header_is_44_bytes():
    header = gm.wav_stream_header()
    assert len(header) == 44
    assert header[:4] == b"RIFF"
    assert header[36:40] == b"data"


def test_extract_pcm_reads_inline_data():
    class _Inline:
        data = FAKE_PCM

    class _Part:
        inline_data = _Inline()

    class _Content:
        parts = [_Part()]

    class _Cand:
        content = _Content()

    class _Resp:
        candidates = [_Cand()]

    assert gm.extract_pcm(_Resp()) == FAKE_PCM
    # A chunk with no audio yields b"" when not required.
    assert gm.extract_pcm(object(), required=False) == b""


def test_can_produce_matches_environment():
    provider = GeminiProvider(api_key="x")
    assert provider.can_produce("wav") is True
    assert provider.can_produce("pcm") is True
    assert provider.can_produce("mp3") is gm.HAS_FFMPEG


class _FakeModels:
    def __init__(self, pcm: bytes):
        self._pcm = pcm

    async def generate_content(self, *, model, contents, config):
        class _Inline:
            data = self._pcm

        class _Part:
            inline_data = _Inline()

        class _Content:
            parts = [_Part()]

        class _Cand:
            content = _Content()

        class _Resp:
            candidates = [_Cand()]

        return _Resp()


class _FakeClient:
    def __init__(self, pcm: bytes):
        class _Aio:
            models = _FakeModels(pcm)

        self.aio = _Aio()


async def test_synthesize_pcm_and_wav():
    provider = GeminiProvider(api_key="x")
    provider._client = _FakeClient(FAKE_PCM)

    raw = await provider.synthesize("hi", "Kore", SynthOpts(format="pcm"))
    assert raw == FAKE_PCM

    wav = await provider.synthesize("hi", "Kore", SynthOpts(format="wav"))
    assert wav[:4] == b"RIFF"


async def test_synthesize_mp3_without_ffmpeg_raises(monkeypatch):
    monkeypatch.setattr(gm, "HAS_FFMPEG", False)
    provider = GeminiProvider(api_key="x")
    provider._client = _FakeClient(FAKE_PCM)
    with pytest.raises(UnsupportedFormat):
        await provider.synthesize("hi", "Kore", SynthOpts(format="mp3"))
