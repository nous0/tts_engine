# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Read `CONTEXT.md` first.** It holds the current working state, what's done, and the
prioritized to-do list. This file covers how the code works; roadmap and status belong in
CONTEXT.md.

## What this is

A Python 3.12 FastAPI service that puts several TTS providers behind one API, for two
consumers:
- **Chatbot streaming**: low-latency, sentence-by-sentence audio over chunked HTTP (`POST /v1/speech/stream`).
- **Podcast generation**: multi-speaker scripts rendered as a background job and stitched into one file (`POST /v1/podcast` → `GET /v1/jobs/{id}` → `GET /v1/jobs/{id}/audio`).

Providers: **Kokoro** (local, CPU, no key, the configured default) and **OpenAI**.
Gemini was removed on 2026-09-22 by decision; ElevenLabs is out of scope.
`GET /v1/voices` is not built yet; `TTSEngine.list_voices()` already exists for it.

## Commands

Uses `uv` (there's a `uv.lock`). Extras: `dev` (pytest, ruff), `ui` (streamlit), `local` (kokoro + soundfile, which pulls in CPU torch).

```bash
uv sync --extra dev --extra local --extra ui     # install
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000   # API; Swagger at /docs
uv run streamlit run streamlit_app.py            # visual tester for /v1/speech + /stream

uv run pytest                                    # all tests
uv run pytest tests/test_podcast.py              # one file
uv run pytest tests/test_chunker.py::test_name   # one test
uv run pytest -k kokoro                          # by keyword

uv run ruff check .                              # lint (E,F,I,UP,B; line length 100)
uv run ruff format .

uv run python examples/chatbot_stream.py                      # streaming demo
uv run python examples/generate_podcast.py examples/script.json  # end-to-end podcast
```

Config comes from `.env` through pydantic-settings (`app/config.py`; see `.env.example`).
Compressed formats (mp3/opus/aac/flac) need `ffmpeg` on PATH. Without it, those formats
return 400. `wav` and `pcm` are always available.

## Architecture

Request flow: **route → `TTSEngine` → provider → (podcast only) audio pipeline → job store.**

- **`app/main.py` lifespan** builds everything once and stores it on `app.state`: `engine`
  (from `build_registry(settings)`), `output_dir`, and `jobs` (the `JobStore`). Routes get
  these from `request.app.state` through small helpers (`_engine(request)`, and so on), not
  FastAPI `Depends`.
- **Provider registry** (`app/core/providers/registry.py`): a provider is registered only
  if it can actually run. OpenAI needs an API key. Kokoro needs `kokoro_enabled`
  and the `kokoro` package importable (`kokoro.is_installed()`). If you request an
  unregistered provider, you get `ProviderNotConfigured`, which lists the available ones.
  Adding a provider means implementing the `TTSProvider` protocol in `providers/base.py`
  (`synthesize`, `synthesize_stream`, `list_voices`) and adding a gated entry in
  `build_registry`.
- **Canonical audio format: 24 kHz, 16-bit, mono PCM** (constants in
  `providers/_audio.py`). Kokoro produces it natively; OpenAI is asked for
  `response_format="pcm"`. `app/core/audio.py` (concat, silence, normalize, encode) builds
  on the same primitives, so provider output and stitched output always match. Any new
  provider must produce this format when `opts.format == "pcm"`.
- **Streaming**: `TTSEngine.synthesize_stream` splits the full request text with
  `split_sentences` and streams each sentence through the provider's `synthesize_stream`.
  For WAV, providers yield a streaming header first (`wav_stream_header()`).
  `SentenceChunker` (for incremental text such as LLM tokens) is used on the client side,
  in `examples/chatbot_stream.py`, not by the server.
- **Kokoro is synchronous**, so its calls are pushed off the event loop (`asyncio.to_thread`
  for one-shot calls, and a thread plus queue bridge for streaming). Keep that pattern for
  any blocking backend.
- **Podcast** (`app/core/podcast.py`): `parse_script` accepts a `turns` list, a JSON string,
  or a plain `Speaker: text` transcript. `assign_voices` fills in unmapped speakers from the
  provider's voice catalog. `render` synthesizes turns **one after another** as PCM, then
  concatenates them with `pause_ms` of silence, peak-normalizes, and encodes once. You can
  override provider, voice, and instructions per turn.
- **Jobs** (`app/core/jobs.py`): a SQLite job store that uses the stdlib `sqlite3` from
  worker threads (not aiosqlite). `JobStore.spawn` runs the render as an in-process asyncio
  task, and results are written to `output_dir`. Jobs don't survive a process restart
  mid-render.

## Testing notes

- `tests/conftest.py` has an autouse fixture that points `OUTPUT_DIR`/`JOBS_DB` at a tmp
  dir and clears the `get_settings()` lru_cache. If a test changes settings through env
  vars, call `get_settings.cache_clear()` so the change takes effect.
- `asyncio_mode = "auto"`, so async tests need no decorator.
- Provider network calls are mocked. Tests shouldn't need real keys or the Kokoro model.

## Repo gotchas

- `kokoro-tts/` is a vendored upstream checkout (its own git repo) used for reference. It's
  gitignored and excluded from ruff, and the app doesn't import it; the app uses the
  `kokoro` PyPI package.
- `DEFAULT_PROVIDER`/`DEFAULT_VOICE` in `.env` override the code default (`kokoro`). Check
  `.env` before assuming which provider a request with no provider will hit.
- The first Kokoro synthesis downloads the model and spaCy data, which takes minutes. After
  that, it's several seconds per sentence on CPU.
- The ruff `target-version` is `py311`, although the project pins Python 3.12.

## Working agreement

- Update `CONTEXT.md` at the end of each work session with what changed and what's next.
- Keep `README.md` in sync with the real feature and phase status.
- Commit working code promptly. Phase 3 and Kokoro once sat uncommitted for about two
  months.
