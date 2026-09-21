"""Tests for the PCM audio pipeline (concat / pause / normalize / encode)."""

from __future__ import annotations

import array
import io
import math
import wave

import pytest

from app.core import audio
from app.core.audio import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH


def tone(duration_ms: int, amplitude: int = 8000, freq: float = 220.0) -> bytes:
    """A sine tone as little-endian 16-bit PCM, for length/gain assertions."""
    n_frames = int(SAMPLE_RATE * duration_ms / 1000)
    samples = array.array(
        "h",
        (
            int(amplitude * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))
            for i in range(n_frames)
        ),
    )
    return samples.tobytes()


def test_silence_length_matches_duration():
    pcm = audio.silence(100)
    assert len(pcm) == SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH // 10
    assert set(pcm) == {0}
    assert audio.duration_ms(pcm) == pytest.approx(100.0)


def test_silence_is_frame_aligned():
    pcm = audio.silence(1)
    assert len(pcm) % (CHANNELS * SAMPLE_WIDTH) == 0


def test_silence_non_positive_is_empty():
    assert audio.silence(0) == b""
    assert audio.silence(-5) == b""


def test_concat_length_is_sum_plus_pauses():
    a, b, c = tone(100), tone(200), tone(50)
    merged = audio.concat([a, b, c], pause_ms=250)
    # Three segments -> two gaps.
    expected = audio.duration_ms(a + b + c) + 2 * 250
    assert audio.duration_ms(merged) == pytest.approx(expected, abs=1.0)


def test_concat_without_pause_is_plain_join():
    a, b = tone(10), tone(20)
    assert audio.concat([a, b]) == a + b


def test_concat_skips_empty_segments():
    a, b = tone(10), tone(20)
    merged = audio.concat([a, b"", b], pause_ms=100)
    assert audio.duration_ms(merged) == pytest.approx(
        audio.duration_ms(a + b) + 100, abs=1.0
    )


def test_concat_of_nothing_is_empty():
    assert audio.concat([]) == b""
    assert audio.concat([b"", b""], pause_ms=500) == b""


def test_peak_reports_loudest_sample():
    assert audio.peak(b"") == 0
    assert audio.peak(audio.silence(10)) == 0
    assert audio.peak(tone(50, amplitude=8000)) == pytest.approx(8000, abs=5)


def test_normalize_raises_quiet_audio_to_target():
    quiet = tone(200, amplitude=1000)
    normalized = audio.normalize(quiet, target_dbfs=-1.0)
    target = 32767 * (10 ** (-1.0 / 20))
    assert audio.peak(normalized) == pytest.approx(target, rel=0.02)
    assert len(normalized) == len(quiet)


def test_normalize_lowers_hot_audio_and_never_clips():
    hot = tone(100, amplitude=32000)
    normalized = audio.normalize(hot, target_dbfs=-6.0)
    assert audio.peak(normalized) < audio.peak(hot)
    assert audio.peak(normalized) <= 32767


def test_normalize_leaves_silence_untouched():
    pcm = audio.silence(50)
    assert audio.normalize(pcm) == pcm


def test_apply_gain_stdlib_matches_numpy_path():
    pcm = tone(20, amplitude=5000)
    fast = audio.apply_gain(pcm, 2.0)
    slow = audio._apply_gain_stdlib(pcm, 2.0)
    assert fast == slow


def test_apply_gain_clips_at_full_scale():
    pcm = tone(20, amplitude=30000)
    boosted = audio.apply_gain(pcm, 10.0)
    assert audio.peak(boosted) == 32767


async def test_encode_wav_round_trips_to_same_pcm():
    pcm = tone(120)
    data = await audio.encode(pcm, "wav")
    assert data[:4] == b"RIFF"

    with wave.open(io.BytesIO(data), "rb") as wf:
        assert wf.getnchannels() == CHANNELS
        assert wf.getsampwidth() == SAMPLE_WIDTH
        assert wf.getframerate() == SAMPLE_RATE
    assert audio.wav_to_pcm(data) == pcm


async def test_encode_pcm_is_passthrough():
    pcm = tone(10)
    assert await audio.encode(pcm, "pcm") is pcm


async def test_encode_compressed_without_ffmpeg_raises(monkeypatch):
    monkeypatch.setattr(audio, "HAS_FFMPEG", False)
    with pytest.raises(audio.AudioError, match="ffmpeg"):
        await audio.encode(tone(10), "mp3")


def test_to_pcm_unwraps_wav_and_passes_raw_through():
    pcm = tone(30)
    wav = audio.pcm_to_wav(pcm)
    assert audio.to_pcm(wav) == pcm
    assert audio.to_pcm(pcm) == pcm


def test_wav_to_pcm_rejects_mismatched_format():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(44_100)
        wf.writeframes(b"\x00\x00" * 100)
    with pytest.raises(audio.AudioError):
        audio.wav_to_pcm(buf.getvalue())
