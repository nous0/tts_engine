"""What the real-time scheduler learns about a provider: how fast it makes and speaks.

Kept per provider (and speed) on the engine, so every WebSocket session starts from
what earlier sessions measured instead of from the provider's built-in guess. That
matters because the CPU's speed varies (load, power saving), and a wrong guess sizes the
first chunks of every session badly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_LATENCY_PRIOR = (0.8, 0.02)  # (a, b) for providers that don't declare one
AUDIO_PER_CHAR_PRIOR = 0.062  # seconds of speech per character (Kokoro, speed 1)


@dataclass
class LatencyModel:
    """Time to a chunk's first audio as ``a + b * chars`` seconds, learned as it goes.

    Exponentially weighted least squares: the provider's prior is entered as two
    pseudo-measurements, each new measurement fades older ones by ``decay``, so one odd
    sample can't swing the estimate but a slower machine is picked up within a few
    chunks.
    """

    prior: tuple[float, float]
    decay: float = 0.8
    _sums: list[float] = field(default_factory=lambda: [0.0] * 5)  # w, x, y, xx, xy

    def __post_init__(self) -> None:
        a, b = self.prior
        for chars in (20.0, 120.0):
            self._add(chars, a + b * chars)

    def update(self, chars: int, seconds: float) -> None:
        self._sums = [s * self.decay for s in self._sums]
        self._add(float(chars), seconds)

    def estimate(self) -> tuple[float, float]:
        w, x, y, xx, xy = self._sums
        var = w * xx - x * x
        if var <= 1e-9:
            return self.prior
        b = (w * xy - x * y) / var
        a = (y - b * x) / w
        return min(max(a, 0.05), 5.0), min(max(b, 0.001), 0.2)

    def _add(self, x: float, y: float) -> None:
        w, sx, sy, sxx, sxy = self._sums
        self._sums = [w + 1, sx + x, sy + y, sxx + x * x, sxy + x * y]


@dataclass
class SpeechTiming:
    """A provider's synthesis latency model and speaking rate (seconds per char)."""

    latency: LatencyModel
    audio_per_char: float = AUDIO_PER_CHAR_PRIOR

    @classmethod
    def for_provider(cls, provider: object, speed: float = 1.0) -> SpeechTiming:
        prior = getattr(provider, "latency_prior", DEFAULT_LATENCY_PRIOR)
        return cls(LatencyModel(prior), AUDIO_PER_CHAR_PRIOR / max(speed, 0.25))

    def record_audio(self, chars: int, seconds: float) -> None:
        """Fold in how long a chunk of ``chars`` characters actually played."""
        if chars > 0:
            self.audio_per_char = 0.7 * self.audio_per_char + 0.3 * (seconds / chars)
