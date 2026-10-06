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
`GET /v1/voices` lists voices with a `gender` (`Voice.gender` in `providers/base.py`), which podcast casting relies on.

## Commands

Uses `uv` (there's a `uv.lock`). Extras: `dev` (pytest, ruff), `ui` (streamlit), `local` (kokoro + soundfile, which pulls in CPU torch), `demo` (gradio + websockets for `examples/voice_chat_ui.py`; needs ffmpeg).

```bash
uv sync --extra dev --extra local --extra ui     # install
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000   # API; Swagger at /docs
uv run streamlit run streamlit_app.py            # visual tester for /v1/speech + /stream

uv run pytest                                    # all tests
uv run pytest tests/test_podcast.py              # one file
uv run pytest tests/test_chunker.py::test_name   # one test
uv run pytest -k kokoro                          # by keyword

uv run python scripts/smoke_test.py              # E2E against a running server (real Kokoro), 32 checks incl. WebSocket
uv run python scripts/bench_latency.py           # streaming TTFA / RTF / stalls against a server

uv run ruff check .                              # lint (E,F,I,UP,B; line length 100)
uv run ruff format .

uv run python examples/ws_client.py                           # WebSocket client: fake LLM tokens -> audio
uv run python examples/voice_chat_ui.py                       # Gradio voice chat demo (extra: demo)
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
  For `wav` the engine asks providers for **PCM** and emits a single `wav_stream_header()`
  itself, just before the first audio chunk. Never let providers emit a header per
  sentence. `flac` can't be streamed (400). The route awaits the first chunk before
  returning the 200, so early failures still get a real status. After that, errors are
  logged and the connection is dropped; never write error text into the audio stream.
- **Real-time sessions** (`app/core/session.py`, route `app/api/routes/realtime.py`):
  `SpeechSession` is transport-neutral. `feed()` pushes text through a server-side
  `SentenceChunker`, whose complete sentences go on a queue for one speaker task. That
  task synthesizes them in order via `engine.open_sentence` (so sentences are cached) and
  emits `SentenceStarted`/`Audio`/`TurnDone`/`TurnFailed`/`Cancelled`.
  - `cancel()` bumps a generation counter, drops queued work and unsent audio, and
    cancels the in-flight sentence task. `aclosing` then closes the provider stream.
  - Every turn ends with exactly one of done/failed/cancelled.
  - The WS route only translates messages. Both the receiver and the event forwarder
    send, so they share `send_lock`.
  - Error payloads come from `app/api/errors.py:error_payload`, shared with the HTTP
    handlers.
  - Browser `Origin`s are refused unless listed in `WS_ALLOWED_ORIGINS`.
- **Kokoro is synchronous and CPU-bound.** It runs on its own `ThreadPoolExecutor`
  (`KOKORO_MAX_CONCURRENCY`, default 1), not the default executor that the job store's
  SQLite calls use. `KPipeline` creation is behind a lock. `_aiter_blocking` bridges the
  blocking generator to async with `call_soon_threadsafe` and a stop flag, so client
  disconnects don't leak threads. Keep this pattern for any blocking backend.
  `warm_up()` preloads the model at startup (`KOKORO_PRELOAD`, off in tests via
  conftest). `state` (`idle|loading|ready|error`) feeds `/health` through
  `TTSEngine.provider_states()`, and `engine.close()` shuts the pool down.
- **Errors**: providers raise `ProviderError(kind=timeout|rate_limited|auth|bad_request|upstream)`.
  Exception handlers in `app/main.py` map it to 504/503/502/400, and map
  `ProviderNotConfigured`, `UnsupportedFormat` and `InvalidRequest` (incl. `InvalidVoice`)
  to 400. Routes don't need their own try/except for these. OpenAI retries come from the
  SDK (`OPENAI_MAX_RETRIES`); the timeout is set explicitly because the SDK default is 600 s.
- **Voices**: `TTSEngine.resolve_voice` validates an explicit voice against the provider's
  catalog. This matters for security: Kokoro would otherwise download unknown names from
  HF or `torch.load` any path ending in `.pt`. It also picks a per-provider default:
  `DEFAULT_VOICE` only for the default provider, otherwise `provider.default_voice`.
- **Cache** (`app/core/cache.py`): `TTSEngine.synthesize` goes through `synthesize_cached`.
  The key includes the resolved voice and the provider's `cache_tag(opts)` (real model or
  package version). Bump `CACHE_VERSION` if the audio pipeline's output changes. Writes
  are atomic, and every `OSError` is a miss or a skipped write. Streaming caches **per
  sentence** through `TTSEngine.open_sentence` (pcm/wav only). A miss passes the
  provider stream through and stores it only if the sentence completes, so an abandoned
  sentence is never cached. The key matches `synthesize_cached` with `format="pcm"`, so
  stream sentences and podcast turns share entries. `use_cache=False` (HTTP
  `Cache-Control: no-cache`) skips the lookup but still stores. Wrap provider streams in
  `contextlib.aclosing` so closing early reaches the provider (Kokoro's stop flag).
- **Chunking** (`app/core/chunker.py`):
  - A terminator ends a sentence only when whitespace follows it, so `3.5`,
    `example.com` and `$4.99` stay whole.
  - It never ends one after an abbreviation in `_ABBREVIATIONS` (`Dr.`, `e.g.`), after
    initials (`U.S.`, `J.`), or after a list number at the start of a chunk.
  - `...` and `?!` count as one terminator, and closers (quotes, brackets, `*_\``) stay
    on the chunk.
  - When a terminator or clause break is the last character received, it **waits** for
    the next one, so streaming token by token gives exactly the same chunks as the whole
    text. `test_streaming_token_by_token_matches_the_whole_text` enforces this; keep it
    green when changing rules.
  - No chunk exceeds `max_length`, whichever way it ends.
  - `fast_start=True` lets a turn's first chunk end at `,;:` (followed by whitespace)
    once it's 24+ chars. `flush()`/`reset()` start a new turn.
  - Pull mode (`append()` + `take(max_chars, final, min_chars)`) is for the adaptive
    scheduler. It takes whole sentences (merging several if they fit), waits while it
    can, and cuts inside a sentence only when the buffer exceeds `max_chars`. The cut
    goes at a clause break, else before a conjunction or preposition (`_CONJUNCTIONS`),
    else at any word. It never splits a word, and it merges a tiny leading sentence
    ("Hi!") instead of speaking it alone.
- **Adaptive scheduling** (`SpeechSession(chunking="adaptive")`, the WS default):
  - The speaker pulls the next chunk only when ready, sized by `_budget()`. The first
    chunk of a turn is sized for latency (`FIRST_CHUNK_TARGET_S`, ≤
    `FIRST_CHUNK_MAX_CHARS`).
  - Later chunks must be synthesized before the queued audio (playback clock) runs out.
    The limit is `(SAFETY * ahead - a) / b`, and `None` (whole sentences) once there is
    enough audio queued.
  - No chunk goes below the break-even length `a / (d - b)`, below which a chunk takes
    longer to make than to play. Without this, a busy CPU spirals into tiny chunks.
  - `a`, `b` (latency: `timing.LatencyModel`, EWLS seeded by the provider's
    `latency_prior`) and `d` (speech seconds per char) are learned per provider and
    speed on the engine (`TTSEngine.timing`), so they carry over between sessions.
  - Kokoro on this laptop CPU (i7-1360P: 4 P-cores + 8 E-cores, *Balanced* power plan)
    varies up to ~2.5x in speed run to run. Benchmarks must interleave modes, and the
    scheduler must stay robust to that.
  - `KOKORO_TORCH_THREADS=4` is ~20% faster than torch's default 12. It is applied in
    the Kokoro worker thread when the pipeline loads.
- **Joining streamed chunks** (`engine.shape_chunk`, `audio.EdgeTrimmer`): every Kokoro
  chunk comes padded with ~0.3 s of silence in front and ~0.5 s behind. With
  `shape_audio` (engine flag, set from `STREAM_SHAPE_AUDIO` in the app, off by default in
  unit tests), each streamed pcm/wav chunk has its edge silence trimmed. The trimmer
  keeps 30 ms with a 5 ms fade, and holds back only the quiet tail, so speech is never
  delayed. A pause by boundary follows (`audio.PAUSE_AFTER_MS`: sentence 300 ms, clause
  150, line 450, word 40, end 0). The cache keeps the raw audio, and podcasts don't use
  this yet. The chunker reports the boundary (`Chunk.boundary`, from
  `feed_chunks`/`flush_chunk`/`split_chunks`), and the WS `sentence` event carries it
  along with the spoken `text`.
- **Text normalization** (`app/core/textnorm.py`): `normalize_for_speech` strips
  markdown, turns URLs into "link" and drops emoji. It runs on each chunk *after*
  chunking (whole text, so it is safe when streaming), and an empty result is skipped.
  It is on by default for WebSocket sessions and opt-in (`normalize`) on the HTTP
  stream.
- **Logging** (`app/logging_setup.py`): the `app.*` and `openai` loggers, with a request id
  from a contextvar set by the middleware in `app/main.py`. Log text lengths, never text.
- **Podcast** (`app/core/podcast.py`): `parse_script` accepts a `turns` list, a JSON string,
  or a plain `Speaker: text` transcript. `assign_voices` casts unmapped speakers by gender:
  explicit `genders`, then `app/core/names.py:guess_gender` (a lookup table, which returns
  `None` when unsure), then whichever gender balances the cast. Each speaker gets the first
  *unused* catalog voice of that gender, so catalogs must list their best voices first and
  tag `Voice.gender`. `render` synthesizes turns **one after another** as PCM, then
  concatenates them with `pause_ms` of silence, peak-normalizes, and encodes once. You can
  override provider, voice, and instructions per turn.
- **Jobs** (`app/core/jobs.py`): a SQLite job store that uses the stdlib `sqlite3` from
  worker threads (not aiosqlite). `JobStore.spawn` queues the render as an in-process
  asyncio task behind a semaphore (`MAX_CONCURRENT_JOBS`), and results are written to
  `output_dir`.
  - `cancel` records `cancelled` for a user cancel and `error` for a shutdown.
  - `recover_stale` (at startup) fails jobs left `queued`/`running` by a previous process.
  - `start_maintenance` deletes old finished jobs and their files hourly.
  - **Single-process assumption:** don't run uvicorn with `--workers N`.
  - Every SQLite call goes through `_in_thread`, which keeps the lock until the thread
    finishes even if the caller is cancelled, and `close()` takes the same lock. Closing the
    connection under a running query crashes the process.

- **Voice chat demo** (`examples/voice_chat_ui.py`): the Gradio backend plays the
  chatbot. It runs the LLM (OpenAI or the Fake LLM), sends deltas to `/v1/speech/ws`,
  and yields WAV chunks to `gr.Audio(streaming=True)`. Gradio decodes those with the
  stdlib and keeps one ffmpeg AAC encoder per stream, so it's gapless. That player adds
  ~0.8 s before playback (measured with headless Chromium); the user accepted this over
  a custom Web Audio page. Barge-in works through a per-session token: Stop or a new
  question supersedes the running answer, which sends `cancel`.
  `tests/test_voice_chat_ui.py` covers its non-UI logic and is skipped without gradio.

## Testing notes

- `tests/conftest.py` has an autouse fixture that points `OUTPUT_DIR`/`JOBS_DB`/`CACHE_DIR`
  at a tmp dir and clears the `get_settings()` lru_cache. If a test changes settings through env
  vars, call `get_settings.cache_clear()` so the change takes effect.
- `asyncio_mode = "auto"`, so async tests need no decorator.
- conftest also sets `KOKORO_PRELOAD=false` and `KOKORO_TORCH_THREADS=0`, so app
  startup in tests never loads the model or imports torch. A slow background torch
  import made the preload tests flaky.
- Timing-based tests (`test_session.py`'s `TimedProvider`) use small, fast delays and
  generous thresholds; keep them that way.
- Provider network calls are mocked. Tests shouldn't need real keys or the Kokoro model.
  `tests/test_openai.py` runs the real OpenAI SDK against `httpx.MockTransport` (error
  responses carry `retry-after-ms: 1` to keep retries fast). Kokoro tests inject fake
  pipelines into `provider._pipelines`, or a fake `kokoro` module via `sys.modules`.
- A "Windows fatal exception: access violation" thread dump in a test run is a real bug,
  not noise. The one seen so far was the job store closing SQLite under a query still
  running in a thread (fixed in `JobStore._in_thread`). Run the suite several times after
  touching threading code.

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
