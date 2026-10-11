"""Tests for ffmpeg encoding options and loudness normalization (EBU R128).

The subprocess is faked for argument/parsing checks; tests marked ``real_ffmpeg`` run
the actual binary and are skipped when it isn't installed.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import wave

import pytest

from app.core import audio
from app.core.audio import SAMPLE_RATE, SAMPLE_WIDTH
from app.core.providers import _audio as _ffmpeg
from tests.test_audio import tone


def _analysis(i: str, tp: str) -> str:
    """What ffmpeg prints for loudnorm=print_format=json (the fields we read)."""
    fields = {"input_i": i, "input_tp": tp, "input_lra": "2.40", "normalization_type": "dynamic"}
    return "[Parsed_loudnorm_0 @ 0x1]\n" + json.dumps(fields, indent="\t") + "\n"


def _fake_ffmpeg(monkeypatch, *responses: tuple[bytes, str]) -> list[list[str]]:
    calls: list[list[str]] = []
    queue = list(responses)

    async def run(args, data, verbose=False):
        calls.append(list(args))
        return queue.pop(0)

    monkeypatch.setattr(_ffmpeg, "_run_ffmpeg", run)
    return calls


async def test_ffmpeg_encode_mp3_uses_lame_with_a_real_bitrate(monkeypatch):
    calls = _fake_ffmpeg(monkeypatch, (b"MP3", ""))
    assert await _ffmpeg.ffmpeg_encode(tone(10), "mp3") == b"MP3"
    args = calls[0]
    assert args[args.index("-c:a") + 1] == "libmp3lame"
    assert args[args.index("-b:a") + 1] == "64k"
    assert "-metadata" not in args
    assert args[-1] == "pipe:1"


async def test_ffmpeg_encode_passes_rate_bitrate_and_tags(monkeypatch):
    calls = _fake_ffmpeg(monkeypatch, (b"MP3", ""))
    await _ffmpeg.ffmpeg_encode(
        tone(10), "mp3", sample_rate=44100, bitrate="96k", metadata={"title": "Ep 1\nPilot"}
    )
    args = calls[0]
    # The output rate comes after the input, so it applies to the output.
    assert args[args.index("-i") + 1 :].count("44100") == 1
    assert args[args.index("-b:a") + 1] == "96k"
    assert args[args.index("-metadata") + 1] == "title=Ep 1 Pilot"


async def test_ffmpeg_encode_unknown_format_raises_before_running(monkeypatch):
    calls = _fake_ffmpeg(monkeypatch)
    with pytest.raises(KeyError):
        await _ffmpeg.ffmpeg_encode(tone(10), "ogg")
    assert calls == []


async def test_loudness_uses_one_gain_when_the_peaks_fit(monkeypatch):
    calls = _fake_ffmpeg(
        monkeypatch,
        (b"", _analysis("-25.68", "-9.11")),
        (b"PCM", ""),
        (b"", _analysis("-19.01", "-2.43")),
    )
    out, stats = await _ffmpeg.ffmpeg_loudness_normalize(tone(10), -19.0, -1.5)

    assert out == b"PCM"
    measure, apply, remeasure = calls
    assert "loudnorm=print_format=json" in measure
    assert measure[-3:] == ["-f", "null", "-"]
    # -25.68 -> -19 is +6.68 dB, and the peak (-9.11 + 6.68 = -2.43) stays under -1.5.
    assert apply[apply.index("-af") + 1] == "volume=6.68dB"
    assert apply[-3:] == ["-f", "s16le", "pipe:1"]
    assert remeasure == measure
    assert stats == {
        "input_i": -25.68,
        "input_tp": -9.11,
        "gain_db": 6.68,
        "output_i": -19.01,
        "output_tp": -2.43,
        "normalization_type": "linear",
    }


async def test_loudness_limits_only_peaks_the_gain_would_push_too_high(monkeypatch):
    calls = _fake_ffmpeg(
        monkeypatch,
        (b"", _analysis("-25.82", "-5.25")),
        (b"PCM", ""),
        (b"", _analysis("-19.09", "-1.96")),
    )
    _out, stats = await _ffmpeg.ffmpeg_loudness_normalize(tone(10), -19.0, -1.5)

    chain = calls[1][calls[1].index("-af") + 1]
    # Oversampled so inter-sample peaks are caught, then back to the canonical rate.
    assert chain.startswith(f"aresample={SAMPLE_RATE * 4},volume=6.82dB,alimiter=")
    assert chain.endswith(f"aresample={SAMPLE_RATE}")
    limit = float(re.search(r"limit=([\d.]+)", chain).group(1))
    assert limit == pytest.approx(10 ** (-2.0 / 20), abs=1e-4)  # -1.5 dBTP minus margin
    assert "level=disabled" in chain  # no auto gain on top
    assert "latency=true" in chain  # no shift in timing
    assert stats["normalization_type"] == "linear+limiter"
    assert stats["output_tp"] == -1.96


async def test_loudness_corrects_the_gain_once_when_limiting_falls_short(monkeypatch):
    calls = _fake_ffmpeg(
        monkeypatch,
        (b"", _analysis("-31.06", "-4.28")),
        (b"FIRST", ""),
        (b"", _analysis("-19.99", "-1.93")),  # the limited peaks carried ~1 LU
        (b"SECOND", ""),
        (b"", _analysis("-19.05", "-1.95")),
    )
    out, stats = await _ffmpeg.ffmpeg_loudness_normalize(tone(10), -19.0, -1.5)

    assert out == b"SECOND"
    assert len(calls) == 5
    assert "volume=12.06dB" in calls[1][calls[1].index("-af") + 1]
    assert "volume=13.05dB" in calls[3][calls[3].index("-af") + 1]
    assert stats["gain_db"] == 13.05
    assert stats["output_i"] == -19.05


async def test_loudness_without_a_limiter_never_takes_a_second_pass(monkeypatch):
    calls = _fake_ffmpeg(
        monkeypatch,
        (b"", _analysis("-25.0", "-20.0")),
        (b"PCM", ""),
        (b"", _analysis("-19.8", "-14.0")),
    )
    await _ffmpeg.ffmpeg_loudness_normalize(tone(10), -19.0, -1.5)
    assert len(calls) == 3


async def test_loudness_leaves_silence_alone(monkeypatch):
    calls = _fake_ffmpeg(monkeypatch, (b"", _analysis("-inf", "-inf")))
    pcm = audio.silence(100)
    out, stats = await _ffmpeg.ffmpeg_loudness_normalize(pcm, -19.0, -1.5)
    assert out is pcm
    assert stats["normalization_type"].startswith("skipped")
    assert len(calls) == 1


async def test_loudness_without_measurements_raises(monkeypatch):
    _fake_ffmpeg(monkeypatch, (b"", "no json here"))
    with pytest.raises(RuntimeError, match="no measurements"):
        await _ffmpeg.ffmpeg_loudness_normalize(tone(10), -19.0, -1.5)


async def test_loudness_normalize_without_ffmpeg_raises(monkeypatch):
    monkeypatch.setattr(audio, "HAS_FFMPEG", False)
    with pytest.raises(audio.AudioError, match="requires ffmpeg"):
        await audio.loudness_normalize(tone(10))


async def test_loudness_normalize_passes_the_target_and_wraps_failures(monkeypatch):
    monkeypatch.setattr(audio, "HAS_FFMPEG", True)
    seen: list[tuple] = []

    async def broken(pcm, i, tp):
        seen.append((i, tp))
        raise RuntimeError("ffmpeg failed: boom")

    monkeypatch.setattr(audio, "ffmpeg_loudness_normalize", broken)
    with pytest.raises(audio.AudioError, match="boom"):
        await audio.loudness_normalize(tone(10), audio.LoudnessTarget(integrated=-16))
    assert seen == [(-16, -1.5)]


async def test_encode_forwards_compressed_options(monkeypatch):
    monkeypatch.setattr(audio, "HAS_FFMPEG", True)
    seen: dict = {}

    async def fake_encode(pcm, fmt, **kwargs):
        seen.update(kwargs, fmt=fmt)
        return b"MP3"

    monkeypatch.setattr(audio, "ffmpeg_encode", fake_encode)
    out = await audio.encode(
        tone(10), "mp3", sample_rate=44100, bitrate="64k", metadata={"title": "T"}
    )
    assert out == b"MP3"
    assert seen == {
        "fmt": "mp3",
        "sample_rate": 44100,
        "bitrate": "64k",
        "metadata": {"title": "T"},
    }


async def test_encode_wav_ignores_compressed_options():
    data = await audio.encode(tone(10), "wav", sample_rate=44100, metadata={"title": "T"})
    with wave.open(io.BytesIO(data), "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE


# --- the real ffmpeg binary ---

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


def _speechlike(seconds: float) -> bytes:
    """Bursts of tone at varying levels with short gaps, so loudness gating has work."""
    parts = []
    for i in range(int(seconds * 2)):
        parts.append(tone(400, amplitude=2000 + (i % 4) * 1500, freq=180 + 40 * (i % 5)))
        parts.append(audio.silence(100))
    return b"".join(parts)


def measure_loudness(source: bytes | str) -> tuple[float, float]:
    """Integrated loudness (LUFS) and true peak (dBTP) of PCM or a file, via ebur128."""
    if isinstance(source, bytes):
        inputs = ["-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0"]
        data: bytes | None = source
    else:
        inputs, data = ["-i", source], None
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", *inputs, "-af", "ebur128=peak=true",
         "-f", "null", "-"],
        input=data,
        capture_output=True,
        check=True,
    )
    summary = proc.stderr.decode(errors="ignore").rsplit("Summary:", 1)[1]
    integrated = float(re.search(r"I:\s+(-?[\d.]+) LUFS", summary).group(1))
    true_peak = float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", summary).group(1))
    return integrated, true_peak


@pytest.mark.parametrize("target", [-19.0, -23.0])
@needs_ffmpeg
@pytest.mark.real_ffmpeg
async def test_real_loudness_normalize_hits_the_target(target):
    pcm, stats = await audio.loudness_normalize(
        _speechlike(6), audio.LoudnessTarget(integrated=target)
    )
    integrated, true_peak = measure_loudness(pcm)
    assert integrated == pytest.approx(target, abs=1.0)
    assert true_peak <= -1.0
    assert stats["normalization_type"] == "linear"
    assert stats["output_i"] == pytest.approx(target, abs=1.0)
    assert len(pcm) % SAMPLE_WIDTH == 0


@needs_ffmpeg
@pytest.mark.real_ffmpeg
async def test_real_rare_peaks_are_limited_without_missing_the_target():
    # Quiet speech with a few sharp clicks, like plosives: one gain to -19 LUFS would
    # push those clicks over full scale.
    click = tone(3, amplitude=20000, freq=1500)
    body = audio.apply_gain(_speechlike(6), 0.3)  # about -10 dB, so the gain is large
    pcm = body[: len(body) // 2] + click + body[len(body) // 2 :] + click
    out, stats = await audio.loudness_normalize(pcm)

    assert stats["normalization_type"] == "linear+limiter"
    integrated, true_peak = measure_loudness(out)
    assert integrated == pytest.approx(-19.0, abs=1.0)
    assert true_peak <= -1.5
    assert len(out) == len(pcm)  # the limiter's lookahead doesn't shift or pad


@needs_ffmpeg
@pytest.mark.real_ffmpeg
async def test_real_mp3_is_44k_with_the_bitrate_and_title(tmp_path):
    data = await audio.encode(
        _speechlike(3), "mp3", sample_rate=44100, bitrate="64k", metadata={"title": "Pilot"}
    )
    path = tmp_path / "ep.mp3"
    path.write_bytes(data)
    probe = json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_name,sample_rate,channels,bit_rate:format_tags=title",
             "-of", "json", str(path)],
            capture_output=True,
            check=True,
        ).stdout
    )
    stream = probe["streams"][0]
    assert stream["codec_name"] == "mp3"
    assert stream["sample_rate"] == "44100"
    assert stream["channels"] == 1
    assert int(stream["bit_rate"]) == 64000
    assert probe["format"]["tags"]["title"] == "Pilot"
