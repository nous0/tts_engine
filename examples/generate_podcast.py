"""Render a multi-speaker script into one audio file.

Runs the podcast pipeline directly (no HTTP server needed):

    uv run python examples/generate_podcast.py examples/script.json
    uv run python examples/generate_podcast.py examples/script.json --format wav

The script may be JSON (a list of ``{"speaker": ..., "text": ...}`` objects) or
plain text with ``Alice: ...`` lines. Speakers without an explicit voice are
cast automatically by gender (guessed from the name, or set with --gender).
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from app.config import get_settings
from app.core import podcast as podcast_core
from app.core.cache import FileCache
from app.core.engine import TTSEngine
from app.core.podcast import PodcastSpec
from app.core.providers.base import SynthOpts
from app.core.providers.registry import build_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a podcast script to audio.")
    parser.add_argument("script", type=Path, help="Path to a .json or .txt script.")
    parser.add_argument("--provider", default=None, help="Provider name (default: config).")
    parser.add_argument("--format", default="wav", help="Output format (default: wav).")
    parser.add_argument("--pause-ms", type=int, default=600, help="Silence between turns.")
    parser.add_argument("--out", type=Path, default=None, help="Output file path.")
    parser.add_argument(
        "--voice",
        action="append",
        default=[],
        metavar="SPEAKER=VOICE",
        help="Pin a speaker to a voice; repeatable.",
    )
    parser.add_argument(
        "--gender",
        action="append",
        default=[],
        metavar="SPEAKER=female|male",
        help="Set a speaker's gender when the name can't be guessed; repeatable.",
    )
    return parser.parse_args()


def parse_voice_flags(pairs: list[str]) -> dict[str, str]:
    voices: dict[str, str] = {}
    for pair in pairs:
        speaker, _, voice = pair.partition("=")
        if not speaker or not voice:
            raise SystemExit(f"Invalid --voice '{pair}'; expected SPEAKER=VOICE.")
        voices[speaker] = voice
    return voices


def parse_gender_flags(pairs: list[str]) -> dict[str, str]:
    genders: dict[str, str] = {}
    for pair in pairs:
        speaker, _, gender = pair.partition("=")
        if not speaker or gender not in ("female", "male"):
            raise SystemExit(f"Invalid --gender '{pair}'; expected SPEAKER=female|male.")
        genders[speaker] = gender
    return genders


async def main() -> None:
    args = parse_args()
    settings = get_settings()
    providers = build_registry(settings)
    if not providers:
        raise SystemExit(
            "No providers configured. Install Kokoro (pip install -e \".[local]\") "
            "or set OPENAI_API_KEY in .env."
        )

    # Same on-disk cache as the server: re-rendering a script skips turns already done.
    cache = (
        FileCache(settings.cache_dir, settings.cache_max_mb * 1024 * 1024)
        if settings.cache_enabled
        else None
    )
    engine = TTSEngine(
        providers=providers,
        default_provider=settings.default_provider,
        default_voice=settings.default_voice,
        cache=cache,
    )

    turns = podcast_core.parse_script(args.script.read_text(encoding="utf-8"))
    catalog = await engine.list_voices(args.provider)
    voices = podcast_core.assign_voices(
        turns, parse_voice_flags(args.voice), catalog, parse_gender_flags(args.gender)
    )

    print(f"Script: {len(turns)} turns, {len(voices)} speakers")
    gender_of = {v.id: v.gender for v in catalog}
    for speaker, voice in voices.items():
        print(f"  {speaker:<12} -> {voice} ({gender_of.get(voice) or 'neutral'})")

    spec = PodcastSpec(
        turns=turns,
        voices=voices,
        provider=args.provider,
        format=args.format,
        pause_ms=args.pause_ms,
    )

    started = time.perf_counter()

    def on_progress(done: int, total: int) -> None:
        elapsed = time.perf_counter() - started
        print(f"  [{done}/{total}] rendered (t={elapsed:.1f}s)")

    async def synthesize(text: str, voice: str, provider: str | None, opts: SynthOpts):
        return await engine.synthesize(text, voice=voice, provider=provider, opts=opts)

    print("\nRendering...")
    data = await podcast_core.render(spec, synthesize, on_progress)

    out_path = args.out or Path(settings.output_dir) / f"{args.script.stem}.{args.format}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)

    elapsed = time.perf_counter() - started
    print(f"\nDone in {elapsed:.1f}s -> {out_path} ({len(data) / 1024:.0f} KB)")


if __name__ == "__main__":
    asyncio.run(main())
