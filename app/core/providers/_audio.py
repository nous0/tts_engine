"""Shared audio helpers for providers that emit raw PCM (Gemini, Kokoro, ...).

Both cloud (Gemini) and local (Kokoro) backends produce 24 kHz, 16-bit, mono PCM.
These helpers wrap PCM into WAV natively (stdlib ``wave``, no external tools) and,
when ``ffmpeg`` is on the PATH, transcode to compressed formats (mp3/opus/...).

The podcast pipeline in ``app.core.audio`` builds on these same primitives, so
provider output and stitched output always share one canonical PCM format.
"""

from __future__ import annotations

import asyncio
import io
import shutil
import struct
import wave

# PCM parameters shared by the PCM-emitting providers.
SAMPLE_RATE = 24_000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes per sample (16-bit)

# Formats emittable with no external tooling.
NATIVE_FORMATS = frozenset({"wav", "pcm"})

HAS_FFMPEG = shutil.which("ffmpeg") is not None

# ffmpeg output-format flags per container (input is always raw s16le PCM).
_FFMPEG_FMT = {
    "mp3": ["-f", "mp3"],
    "opus": ["-f", "opus"],
    "aac": ["-f", "adts"],
    "flac": ["-f", "flac"],
}


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


async def ffmpeg_encode(pcm: bytes, fmt: str) -> bytes:
    """Transcode raw s16le PCM to ``fmt`` via ffmpeg (must be on PATH)."""
    args = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "-i", "pipe:0",
        *_FFMPEG_FMT[fmt], "pipe:1",
    ]
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(pcm)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {err.decode(errors='ignore')}")
    return out
