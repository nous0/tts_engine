"""Audio pipeline: concat / pause / normalize / encode for stitched output.

Every provider here can emit raw PCM at 24 kHz, 16-bit, mono — Kokoro and Gemini
natively, OpenAI via ``response_format="pcm"`` — so the podcast pipeline works in
that single canonical format: synthesize each turn as PCM, stitch the turns with
silence, normalize the mix, then encode once at the end.

The PCM/WAV primitives are shared with the providers (``providers/_audio.py``), so
provider output and assembled output never disagree about sample rate or width.
WAV needs no external tooling; compressed formats require ``ffmpeg`` on the PATH.
"""

from __future__ import annotations

import io
import sys
import wave
from array import array
from collections.abc import Iterable

from .providers._audio import (
    CHANNELS,
    HAS_FFMPEG,
    NATIVE_FORMATS,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    ffmpeg_encode,
    pcm_to_wav,
)

__all__ = [
    "AudioError",
    "CHANNELS",
    "HAS_FFMPEG",
    "NATIVE_FORMATS",
    "SAMPLE_RATE",
    "SAMPLE_WIDTH",
    "apply_gain",
    "concat",
    "duration_ms",
    "encode",
    "normalize",
    "pcm_to_wav",
    "peak",
    "silence",
    "to_pcm",
    "wav_to_pcm",
]

FULL_SCALE = 32767
BYTES_PER_FRAME = CHANNELS * SAMPLE_WIDTH

# Headroom left by the default normalization pass, in dB below full scale.
DEFAULT_TARGET_DBFS = -1.0


class AudioError(Exception):
    """Raised when audio cannot be decoded, mixed, or encoded as requested."""


def duration_ms(pcm: bytes) -> float:
    """Duration of a PCM buffer in milliseconds."""
    return len(pcm) / BYTES_PER_FRAME * 1000 / SAMPLE_RATE


def silence(ms: int) -> bytes:
    """A frame-aligned run of digital silence ``ms`` milliseconds long."""
    if ms <= 0:
        return b""
    frames = round(SAMPLE_RATE * ms / 1000)
    return bytes(frames * BYTES_PER_FRAME)


def concat(segments: Iterable[bytes], pause_ms: int = 0) -> bytes:
    """Join PCM segments, inserting ``pause_ms`` of silence between them.

    Empty segments are skipped so a provider returning nothing for one turn
    does not leave a double pause in the middle of the episode.
    """
    gap = silence(pause_ms)
    parts: list[bytes] = []
    for segment in segments:
        if not segment:
            continue
        if parts and gap:
            parts.append(gap)
        parts.append(segment)
    return b"".join(parts)


def peak(pcm: bytes) -> int:
    """Largest absolute sample value in ``pcm`` (0 for empty or silent audio)."""
    if not pcm:
        return 0
    np = _numpy()
    if np is not None:
        samples = np.frombuffer(_whole_samples(pcm), dtype="<i2")
        if samples.size == 0:
            return 0
        return int(np.abs(samples.astype("int32")).max())
    samples = _samples(pcm)
    if not samples:
        return 0
    return max(max(samples), -min(samples))


def apply_gain(pcm: bytes, factor: float) -> bytes:
    """Scale every sample by ``factor``, clipping at full scale rather than wrapping."""
    if not pcm or factor == 1.0:
        return pcm
    np = _numpy()
    if np is None:
        return _apply_gain_stdlib(pcm, factor)
    samples = np.frombuffer(_whole_samples(pcm), dtype="<i2").astype("float64") * factor
    clipped = np.clip(np.rint(samples), -FULL_SCALE, FULL_SCALE)
    return clipped.astype("<i2").tobytes()


def _apply_gain_stdlib(pcm: bytes, factor: float) -> bytes:
    """Pure-stdlib fallback for ``apply_gain`` (and the reference implementation)."""
    scaled = array(
        "h",
        (
            max(-FULL_SCALE, min(FULL_SCALE, _round_half_even(s * factor)))
            for s in _samples(pcm)
        ),
    )
    return _to_bytes(scaled)


def normalize(pcm: bytes, target_dbfs: float = DEFAULT_TARGET_DBFS) -> bytes:
    """Peak-normalize ``pcm`` so its loudest sample sits at ``target_dbfs``.

    Peak normalization is what keeps a stitched episode at a consistent level
    without ever clipping: quiet turns come up, hot turns come down, and the
    small headroom below full scale survives later encoding. (Perceptual
    loudness matching — EBU R128 / LUFS — needs ffmpeg and is a later concern.)
    """
    current = peak(pcm)
    if current <= 0:
        return pcm
    target = FULL_SCALE * (10 ** (target_dbfs / 20))
    return apply_gain(pcm, target / current)


async def encode(pcm: bytes, fmt: str) -> bytes:
    """Encode raw PCM into ``fmt``, shelling out to ffmpeg for compressed containers."""
    if fmt == "pcm":
        return pcm
    if fmt == "wav":
        return pcm_to_wav(pcm)
    if not HAS_FFMPEG:
        raise AudioError(
            f"Format '{fmt}' requires ffmpeg on the PATH. Use 'wav' or 'pcm', "
            f"or install ffmpeg."
        )
    try:
        return await ffmpeg_encode(pcm, fmt)
    except KeyError as exc:
        raise AudioError(f"Unsupported output format '{fmt}'.") from exc
    except RuntimeError as exc:
        raise AudioError(str(exc)) from exc


def wav_to_pcm(data: bytes) -> bytes:
    """Extract raw frames from a WAV container, rejecting foreign PCM parameters.

    Mixing 44.1 kHz stereo into a 24 kHz mono timeline would silently change
    pitch and duration, so a mismatch is an error rather than a best effort.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            params = (wf.getnchannels(), wf.getsampwidth(), wf.getframerate())
            frames = wf.readframes(wf.getnframes())
    except wave.Error as exc:
        raise AudioError(f"Not a readable WAV stream: {exc}") from exc

    expected = (CHANNELS, SAMPLE_WIDTH, SAMPLE_RATE)
    if params != expected:
        raise AudioError(
            f"WAV is {params[0]}ch/{params[1] * 8}-bit/{params[2]} Hz; "
            f"the pipeline needs {expected[0]}ch/{expected[1] * 8}-bit/{expected[2]} Hz."
        )
    return frames


def to_pcm(data: bytes) -> bytes:
    """Return raw PCM for ``data``, unwrapping a WAV container when present.

    Providers are asked for ``pcm``, but tolerating WAV keeps the mixer working
    with any backend that only speaks WAV.
    """
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return wav_to_pcm(data)
    return data


def _numpy():
    """Return numpy if it is installed, else ``None``.

    numpy arrives with the local Kokoro backend and makes the sample-level
    passes fast enough for hour-long episodes; the stdlib fallback keeps the
    pipeline working on a cloud-only install.
    """
    return sys.modules.get("numpy") or _import_numpy()


def _import_numpy():
    try:
        import numpy
    except ImportError:  # pragma: no cover - depends on the install extras
        return None
    return numpy


def _whole_samples(pcm: bytes) -> bytes:
    """Drop a trailing partial sample so the buffer is a whole number of samples."""
    extra = len(pcm) % SAMPLE_WIDTH
    return pcm[: len(pcm) - extra] if extra else pcm


def _samples(pcm: bytes) -> array:
    samples = array("h")
    samples.frombytes(_whole_samples(pcm))
    if sys.byteorder != "little":  # pragma: no cover - x86/ARM are little-endian
        samples.byteswap()
    return samples


def _to_bytes(samples: array) -> bytes:
    if sys.byteorder != "little":  # pragma: no cover - x86/ARM are little-endian
        samples = array("h", samples)
        samples.byteswap()
    return samples.tobytes()


def _round_half_even(value: float) -> int:
    """Match numpy's ``rint`` so both gain paths produce identical bytes."""
    return int(round(value))
