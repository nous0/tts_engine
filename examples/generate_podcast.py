"""Render a multi-speaker script into one audio file, or write one from a topic first.

Runs the podcast pipeline directly (no HTTP server needed):

    uv run python examples/generate_podcast.py examples/script.json
    uv run python examples/generate_podcast.py examples/script_trio.txt --format mp3 --title "Trio"

From a topic, with a review step (the LLM writes the script, you edit it, then render):

    uv run python examples/generate_podcast.py --topic "Why local TTS matters" --minutes 2 \\
        --script-only          # -> output/why-local-tts-matters.script.json
    uv run python examples/generate_podcast.py output/why-local-tts-matters.script.json --format mp3

or in one go with ``--topic "..." --format mp3``. ``--writer fake`` writes a fixed
dialogue without calling OpenAI.

The script may be JSON (a list of ``{"speaker": ..., "text": ...}`` objects, or an
object with ``turns`` plus optional ``genders`` and ``title``, as --topic writes it) or
plain text with ``Alice: ...`` lines. Speakers without an explicit voice are cast
automatically by gender (from the script, guessed from the name, or set with --gender).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from app.config import Settings, get_settings
from app.core import audio
from app.core import podcast as podcast_core
from app.core.cache import FileCache
from app.core.engine import TTSEngine
from app.core.podcast import PodcastSpec
from app.core.providers.base import SynthOpts
from app.core.providers.registry import build_registry
from app.core.scriptwriter import Host, TopicBrief, build_writer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a podcast script to audio.")
    parser.add_argument(
        "script", type=Path, nargs="?", help="Path to a .json or .txt script (or use --topic)."
    )
    parser.add_argument("--topic", default=None, help="Have the LLM write the script first.")
    parser.add_argument("--minutes", type=float, default=3.0, help="Target length for --topic.")
    parser.add_argument(
        "--host",
        action="append",
        default=[],
        metavar="NAME[=female|male]",
        help="A host for --topic; repeatable (default: the writer picks two).",
    )
    parser.add_argument("--style", default=None, help="Tone hints for --topic.")
    parser.add_argument(
        "--writer",
        choices=["openai", "fake"],
        default=None,
        help="Script writer for --topic (default: SCRIPT_WRITER from config).",
    )
    parser.add_argument(
        "--script-only",
        action="store_true",
        help="With --topic: save the script for review and stop before rendering.",
    )
    parser.add_argument("--provider", default=None, help="Provider name (default: config).")
    parser.add_argument("--format", default="wav", help="Output format (default: wav).")
    parser.add_argument("--pause-ms", type=int, default=600, help="Silence between turns.")
    parser.add_argument("--out", type=Path, default=None, help="Output file path.")
    parser.add_argument("--title", default=None, help="Episode title (mp3 ID3 tag).")
    parser.add_argument(
        "--loudness",
        choices=["lufs", "peak", "off"],
        default=None,
        help="lufs (EBU R128, needs ffmpeg), peak, or off (default: lufs if ffmpeg is found).",
    )
    parser.add_argument(
        "--target-lufs", type=float, default=None, help="Loudness target (default: config)."
    )
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


def parse_host_flags(pairs: list[str]) -> list[Host]:
    hosts = []
    for pair in pairs:
        name, _, gender = pair.partition("=")
        if not name or gender not in ("", "female", "male"):
            raise SystemExit(f"Invalid --host '{pair}'; expected NAME or NAME=female|male.")
        hosts.append(Host(name, gender or None))
    return hosts


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


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "episode"


async def write_script(args: argparse.Namespace, settings: Settings) -> tuple[dict, Path]:
    """Step 1: the LLM writes a script for ``--topic``; it is saved for review."""
    if args.writer:
        settings = settings.model_copy(update={"script_writer": args.writer})
    writer = build_writer(settings)
    if writer is None:
        raise SystemExit("No script writer: set OPENAI_API_KEY, or pass --writer fake.")
    brief = TopicBrief(args.topic, args.minutes, parse_host_flags(args.host), args.style)
    print(f"Writing a {args.minutes:g}-minute script with {writer.label}...")
    started = time.perf_counter()
    script = await writer(brief)
    body = script.to_request()
    path = Path(settings.output_dir) / f"{slug(args.topic)}.script.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"  {script.title!r}: {len(script.turns)} turns, about {script.estimated_minutes} min, "
        f"written in {time.perf_counter() - started:.1f}s -> {path}"
    )
    for turn in script.turns[:4]:
        print(f"  {turn.speaker}: {turn.text}")
    return body, path


def read_script(path: Path) -> dict:
    """A script file as ``{"turns": ..., "genders"?, "title"?, "normalize_text"?}``."""
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw) if raw.lstrip()[:1] in ("[", "{") else raw
    return data if isinstance(data, dict) else {"turns": data}


async def main() -> None:
    args = parse_args()
    if bool(args.script) == bool(args.topic):
        raise SystemExit("Pass either a script file or --topic.")
    settings = get_settings()

    if args.topic:
        data, script_path = await write_script(args, settings)
        if args.script_only:
            print(f"\nReview and edit {script_path}, then render it with:")
            print(f"  uv run python examples/generate_podcast.py {script_path} --format mp3")
            return
        name = script_path.name.removesuffix(".script.json")
    else:
        data = read_script(args.script)
        name = args.script.name.removesuffix(".script.json").removesuffix(args.script.suffix)

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

    turns = podcast_core.parse_script(data)
    if data.get("normalize_text"):
        turns = podcast_core.normalize_turns(turns)
    genders = {**(data.get("genders") or {}), **parse_gender_flags(args.gender)}
    title = args.title or data.get("title")
    catalog = await engine.list_voices(args.provider)
    voices = podcast_core.assign_voices(turns, parse_voice_flags(args.voice), catalog, genders)

    print(f"\nScript: {len(turns)} turns, {len(voices)} speakers")
    gender_of = {v.id: v.gender for v in catalog}
    for speaker, voice in voices.items():
        print(f"  {speaker:<12} -> {voice} ({gender_of.get(voice) or 'neutral'})")

    loudness = podcast_core.pick_loudness(args.loudness, True, audio.HAS_FFMPEG)
    mp3 = args.format == "mp3"
    spec = PodcastSpec(
        turns=turns,
        voices=voices,
        provider=args.provider,
        format=args.format,
        pause_ms=args.pause_ms,
        loudness=loudness,
        target_lufs=(
            args.target_lufs if args.target_lufs is not None else settings.podcast_target_lufs
        ),
        title=title,
        sample_rate=settings.podcast_mp3_sample_rate if mp3 else None,
        bitrate=settings.podcast_mp3_bitrate if mp3 else None,
    )

    started = time.perf_counter()

    def on_progress(done: int, total: int) -> None:
        elapsed = time.perf_counter() - started
        print(f"  [{done}/{total}] rendered (t={elapsed:.1f}s)")

    async def synthesize(text: str, voice: str, provider: str | None, opts: SynthOpts):
        return await engine.synthesize(text, voice=voice, provider=provider, opts=opts)

    print("\nRendering...")
    result = await podcast_core.render(spec, synthesize, on_progress)

    out_path = args.out or Path(settings.output_dir) / f"{name}.{args.format}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(result.audio)

    if result.loudness:
        stats = result.loudness
        print(
            f"Loudness: {stats.get('input_i')} -> {stats.get('output_i')} LUFS, "
            f"true peak {stats.get('output_tp')} dBTP ({stats.get('normalization_type')})"
        )
    else:
        print(f"Loudness: {loudness}")

    elapsed = time.perf_counter() - started
    print(
        f"\nDone in {elapsed:.1f}s -> {out_path} "
        f"({len(result.audio) / 1024:.0f} KB, {result.duration_s:.1f}s of audio)"
    )


if __name__ == "__main__":
    asyncio.run(main())
