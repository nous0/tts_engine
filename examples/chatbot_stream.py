"""Chatbot streaming demo: LLM tokens -> engine stream -> audio.

Run with local Kokoro installed (the "local" extra) or OPENAI_API_KEY in .env:

    python examples/chatbot_stream.py

It simulates an LLM emitting tokens one at a time with a small delay, feeds
them through ``SentenceChunker``, and streams each completed sentence through
the engine. Audio bytes are written to ``output/chatbot_stream.wav`` as they
arrive so you can hear the first sentence before the LLM has finished talking.

No extra playback dependency: we just save the file and print timing. Open
``output/chatbot_stream.wav`` in any player to hear the result.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from pathlib import Path

from app.config import get_settings
from app.core.chunker import SentenceChunker
from app.core.engine import TTSEngine
from app.core.providers.base import SynthOpts
from app.core.providers.registry import build_registry

# A pretend LLM reply, emitted token-by-token.
LLM_REPLY = (
    "Hello! I'm glad you asked. "
    "Text-to-speech streaming lets a chatbot start talking before the whole "
    "reply is generated. "
    "Each sentence is synthesized as soon as it is complete, so latency stays "
    "low even for long answers. "
    "Thanks for listening!"
)

# Simulate token boundaries (roughly word-level).
LLM_TOKENS = LLM_REPLY.split(" ")


async def fake_llm() -> AsyncIterator[str]:
    """Yield tokens with a small delay, like a real LLM would."""
    for token in LLM_TOKENS:
        await asyncio.sleep(0.05)
        yield token + " "


async def main() -> None:
    settings = get_settings()
    providers = build_registry(settings)
    if not providers:
        raise SystemExit(
            "No providers configured. Install Kokoro (uv sync --extra local) "
            "or set OPENAI_API_KEY in .env."
        )
    engine = TTSEngine(
        providers=providers,
        default_provider=settings.default_provider,
        default_voice=settings.default_voice,
    )

    out_dir = Path(settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chatbot_stream.wav"

    opts = SynthOpts(format="wav")
    chunker = SentenceChunker(max_length=200)

    started = time.perf_counter()
    first_byte_at: float | None = None
    total_bytes = 0

    with out_path.open("wb") as f:
        async for token in fake_llm():
            for sentence in chunker.feed(token):
                print(f"  [sentence] {sentence.strip()!r}")
                async for chunk in engine.synthesize_stream(
                    sentence, opts=opts
                ):
                    if first_byte_at is None:
                        first_byte_at = time.perf_counter()
                    f.write(chunk)
                    total_bytes += len(chunk)
                print(
                    f"    -> streamed {total_bytes} bytes total "
                    f"(t={time.perf_counter() - started:.2f}s)"
                )

        tail = chunker.flush()
        if tail:
            print(f"  [sentence] {tail.strip()!r}")
            async for chunk in engine.synthesize_stream(tail, opts=opts):
                if first_byte_at is None:
                    first_byte_at = time.perf_counter()
                f.write(chunk)
                total_bytes += len(chunk)

    elapsed = time.perf_counter() - started
    first_byte_latency = (first_byte_at - started) if first_byte_at else float("nan")
    print(
        f"\nDone: {total_bytes} bytes -> {out_path}\n"
        f"First audio byte at {first_byte_latency:.2f}s, "
        f"total elapsed {elapsed:.2f}s."
    )


if __name__ == "__main__":
    asyncio.run(main())
