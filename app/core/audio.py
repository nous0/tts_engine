"""Audio pipeline: concat / pause / normalize / encode for stitched output.

Every provider here can emit raw PCM at 24 kHz, 16-bit, mono — Kokoro natively,
OpenAI via ``response_format="pcm"`` — so the podcast pipeline works in
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
    "PAUSE_AFTER_MS",
    "AudioError",
    "CHANNELS",
    "EdgeTrimmer",
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

# Joining streamed chunks: samples quieter than this count as silence (about -44 dBFS).
EDGE_THRESHOLD = 200
EDGE_KEEP_MS = 30  # silence kept next to speech, so soft onsets/endings survive
EDGE_FADE_MS = 5  # ramp at the new edges, so a cut never clicks

# Pause inserted after a streamed chunk, by how it ended (see chunker.Boundary).
# Kokoro pads every chunk with ~0.3 s of silence in front and ~0.5 s behind; trimming
# that and adding these instead gives natural gaps instead of ~0.8 s of dead air.
PAUSE_AFTER_MS = {"sentence": 300, "clause": 150, "line": 450, "word": 40, "end": 0}


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


class EdgeTrimmer:
    """Trim the silence around one streamed chunk without delaying its speech.

    ``push`` audio as it arrives and send what it returns; call ``finish`` at the end of
    the chunk. Leading silence is dropped (except ``keep_ms`` before the first sound,
    faded in). Audio is passed on as soon as it contains sound, and only a quiet stretch
    after the last sound is held back, because it might be the chunk's trailing
    silence. ``finish`` keeps ``keep_ms`` of it, faded out. Speech is never delayed,
    even for providers that stream many small pieces (OpenAI).
    """

    def __init__(
        self,
        threshold: int = EDGE_THRESHOLD,
        keep_ms: int = EDGE_KEEP_MS,
        fade_ms: int = EDGE_FADE_MS,
    ) -> None:
        self._threshold = threshold
        self._keep = _frames(keep_ms)
        self._fade = _frames(fade_ms)
        self._started = False
        self._held = array("h")  # leading silence before speech, or quiet tail after it
        self._carry = b""  # half a sample left over from an odd-sized piece

    def push(self, pcm: bytes) -> bytes:
        data = self._carry + pcm
        whole = len(data) - len(data) % SAMPLE_WIDTH
        self._carry = data[whole:]
        buf = self._held + _samples(data[:whole])

        if not self._started:
            first = _first_loud(buf, self._threshold)
            if first is None:
                self._held = buf[-self._keep :] if self._keep else array("h")
                return b""
            buf = buf[max(0, first - self._keep) :]
            _fade(buf, self._fade, fade_in=True)
            self._started = True

        last = _last_loud(buf, self._threshold)
        if last is None:
            self._held = buf
            return b""
        self._held = buf[last + 1 :]
        return _to_bytes(buf[: last + 1])

    def finish(self) -> bytes:
        """The rest of the chunk: a short, faded-out bit of its trailing silence."""
        if not self._started:
            return b""  # the chunk was silent
        tail = self._held[: self._keep]
        self._held = array("h")
        self._carry = b""
        _fade(tail, self._fade, fade_in=False)
        return _to_bytes(tail)


def _frames(ms: int) -> int:
    return max(0, round(SAMPLE_RATE * ms / 1000))


def _first_loud(samples: array, threshold: int) -> int | None:
    np = _numpy()
    if np is not None and len(samples):
        hits = np.flatnonzero(np.abs(np.frombuffer(samples, dtype="<i2").astype("int32"))
                              >= threshold)
        return int(hits[0]) if hits.size else None
    for i, s in enumerate(samples):
        if abs(s) >= threshold:
            return i
    return None


def _last_loud(samples: array, threshold: int) -> int | None:
    np = _numpy()
    if np is not None and len(samples):
        hits = np.flatnonzero(np.abs(np.frombuffer(samples, dtype="<i2").astype("int32"))
                              >= threshold)
        return int(hits[-1]) if hits.size else None
    for i in range(len(samples) - 1, -1, -1):
        if abs(samples[i]) >= threshold:
            return i
    return None


def _fade(samples: array, frames: int, fade_in: bool) -> None:
    """Linear ramp over the first (fade in) or last (fade out) ``frames`` samples."""
    n = min(frames, len(samples))
    for k in range(n):
        gain = (k + 1) / (n + 1)
        i = k if fade_in else len(samples) - 1 - k
        samples[i] = int(samples[i] * gain)


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
