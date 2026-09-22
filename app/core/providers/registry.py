"""Provider registry: builds the map of provider-name -> instance from settings.

Only providers with configured credentials are registered, so an unconfigured
provider surfaces a clear error at request time rather than at import time.
"""

from __future__ import annotations

from app.config import Settings

from .base import TTSProvider
from .kokoro import KokoroProvider
from .kokoro import is_installed as kokoro_installed
from .openai import OpenAIProvider


def build_registry(settings: Settings) -> dict[str, TTSProvider]:
    providers: dict[str, TTSProvider] = {}

    # Local Kokoro (CPU) — registered only when the package is installed so the
    # provider list reflects what can actually run.
    if settings.kokoro_enabled and kokoro_installed():
        providers["kokoro"] = KokoroProvider(default_voice=settings.kokoro_voice)

    if settings.openai_api_key:
        providers["openai"] = OpenAIProvider(
            api_key=settings.openai_api_key,
            default_model=settings.openai_tts_model,
        )

    return providers
