"""Shared audio helpers for providers that emit raw PCM (Kokoro, OpenAI ``pcm``).

The canonical format is 24 kHz, 16-bit, mono PCM.
These helpers wrap PCM into WAV natively (stdlib ``wave``, no external tools) and,
when ``ffmpeg`` is on the PATH, transcode to compressed formats (mp3/opus/...).

The podcast pipeline in ``app.core.audio`` builds on these same primitives, so
provider output and stitched output always share one canonical PCM format.
"""

from __future__ import annotations

import asyncio
import io
import json
import math
import re
import shutil
import struct
import wave
from typing import Any

# PCM parameters shared by the PCM-emitting providers.
SAMPLE_RATE = 24_000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes per sample (16-bit)

# Formats emittable with no external tooling.
NATIVE_FORMATS = frozenset({"wav", "pcm"})

HAS_FFMPEG = shutil.which("ffmpeg") is not None

# ffmpeg output-format flags per container (input is always raw s16le PCM).
_FFMPEG_FMT = {
    # Without an explicit codec and bitrate, ffmpeg wrote 24 kHz mono mp3 at 32 kbps.
    "mp3": ["-c:a", "libmp3lame", "-f", "mp3"],
    "opus": ["-f", "opus"],
    "aac": ["-f", "adts"],
    "flac": ["-f", "flac"],
}

# Bitrate used when the caller doesn't pass one (mono speech).
DEFAULT_BITRATE = {"mp3": "64k"}

_PCM_INPUT = ["-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "-i", "pipe:0"]


def pcm_to_wav(
    pcm: bytes,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
    sample_width: int = SAMPLE_WIDTH,
) -> bytes:
    """Wrap raw PCM in a correct, finite WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def wav_stream_header(
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
    sample_width: int = SAMPLE_WIDTH,
) -> bytes:
    """A 44-byte WAV header for streaming, where the total length is unknown.

    Uses a max placeholder for the RIFF/data sizes; players read PCM until EOF.
    """
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    bits = sample_width * 8
    data_size = 0xFFFFFFFF - 36
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", data_size + 36, b"WAVE",
        b"fmt ", 16, 1, channels, sample_rate, byte_rate, block_align, bits,
        b"data", data_size,
    )


async def ffmpeg_encode(
    pcm: bytes,
    fmt: str,
    *,
    sample_rate: int | None = None,
    bitrate: str | None = None,
    metadata: dict[str, str] | None = None,
) -> bytes:
    """Transcode raw s16le PCM to ``fmt`` via ffmpeg (must be on PATH).

    ``sample_rate`` resamples the output (e.g. 44100 for podcast platforms), ``bitrate``
    overrides ``DEFAULT_BITRATE``, and ``metadata`` becomes container tags (ID3 for mp3).
    Raises ``KeyError`` for an unknown format and ``RuntimeError`` if ffmpeg fails.
    """
    args = [*_PCM_INPUT, *_FFMPEG_FMT[fmt]]
    if sample_rate:
        args += ["-ar", str(sample_rate)]
    if rate := bitrate or DEFAULT_BITRATE.get(fmt):
        args += ["-b:a", rate]
    for key, value in (metadata or {}).items():
        args += ["-metadata", f"{key}={' '.join(value.split())}"]
    out, _err = await _run_ffmpeg([*args, "pipe:1"], pcm)
    return out


# Peaks are limited this far below the true-peak target, since the limiter works on
# samples (at 4x oversampling) and mp3 encoding can add a little.
LIMITER_MARGIN_DB = 0.5
_OVERSAMPLE = 4
# When limiting lowers the loudness by more than this, the gain is corrected once.
LOUDNESS_TOLERANCE_LU = 0.5


async def ffmpeg_loudness_normalize(
    pcm: bytes, target_i: float, target_tp: float
) -> tuple[bytes, dict[str, Any]]:
    """Bring PCM to an integrated loudness (EBU R128) with one gain for the whole file.

    The audio is measured, then scaled by a single gain to ``target_i``. If that gain
    would push the true peak over ``target_tp``, a fast limiter (4x oversampled) catches
    just those peaks. In TTS speech they are rare plosives (about 0.01% of samples on a
    Kokoro episode), so this keeps the voice untouched, where ffmpeg's ``loudnorm``
    would fall back to dynamic mode and ride the gain over the whole file.

    The result is measured again. Loud peaks count toward the measured loudness, so
    limiting them can leave the file short of the target; then the gain is corrected
    by the shortfall and applied once more. Silent audio is returned as is.
    """
    before = await _measure(pcm)
    if not math.isfinite(before["i"]):
        return pcm, {"normalization_type": "skipped (silent)"}

    gain = target_i - before["i"]
    limited = before["tp"] + gain > target_tp
    out = await _apply_gain(pcm, gain, target_tp if limited else None)
    after = await _measure(out)
    shortfall = target_i - after["i"]
    if limited and abs(shortfall) > LOUDNESS_TOLERANCE_LU:
        gain += shortfall
        out = await _apply_gain(pcm, gain, target_tp)
        after = await _measure(out)
    stats = {
        "input_i": before["i"],
        "input_tp": before["tp"],
        "gain_db": round(gain, 2),
        "output_i": after["i"],
        "output_tp": after["tp"],
        "normalization_type": "linear+limiter" if limited else "linear",
    }
    return out, stats


async def _apply_gain(pcm: bytes, gain_db: float, peak_ceiling: float | None) -> bytes:
    """Scale PCM by ``gain_db``; with a ceiling (dBTP), limit the peaks that exceed it."""
    chain = f"volume={gain_db:.2f}dB"
    if peak_ceiling is not None:
        limit = 10 ** ((peak_ceiling - LIMITER_MARGIN_DB) / 20)
        chain = (
            f"aresample={SAMPLE_RATE * _OVERSAMPLE},{chain},"
            f"alimiter=limit={limit:.4f}:attack=1:release=50:level=disabled:latency=true,"
            f"aresample={SAMPLE_RATE}"
        )
    out, _err = await _run_ffmpeg(
        [
            *_PCM_INPUT, "-af", chain,
            "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
            "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1",
        ],
        pcm,
    )
    return out


async def _measure(pcm: bytes) -> dict[str, float]:
    """Integrated loudness (LUFS) and true peak (dBTP), read from loudnorm's analysis."""
    _out, err = await _run_ffmpeg(
        [*_PCM_INPUT, "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        pcm,
        verbose=True,
    )
    data = _loudnorm_json(err)
    return {"i": float(data["input_i"]), "tp": float(data["input_tp"])}


async def _run_ffmpeg(args: list[str], data: bytes, verbose: bool = False) -> tuple[bytes, str]:
    """Run ffmpeg with ``data`` on stdin; return (stdout, stderr text).

    ``verbose`` keeps info-level logging, which is where filters such as loudnorm print
    their results.
    """
    level = ["-nostats", "-loglevel", "info"] if verbose else ["-loglevel", "error"]
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", *level, *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(data)
    text = err.decode(errors="ignore")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {text.strip()[-500:]}")
    return out, text


def _loudnorm_json(stderr: str) -> dict[str, Any]:
    """The JSON block loudnorm prints at the end of its run."""
    blocks = re.findall(r"\{[^{}]*\}", stderr)
    try:
        return json.loads(blocks[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("ffmpeg loudnorm printed no measurements") from exc
