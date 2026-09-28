# TTS Engine

A Python/FastAPI backend that wraps TTS providers behind one API. It targets two
consumers: **chatbot** (low-latency streaming speech) and **podcast** (long-form,
multi-speaker audio files rendered as background jobs).

**Status:** Phases 1–5 done: Kokoro + OpenAI providers, `GET /v1/voices`, gender-aware
podcast casting, and the Phase 5 hardening for local use (timeouts and clear errors, a
real job queue with cancel and restart recovery, an on-disk audio cache, and
request-scoped logging). See `CONTEXT.md` for what's next.

## Providers

| Provider | Kind | Enabled when |
|---|---|---|
| `kokoro` (code default) | Local Kokoro-82M on CPU, no key | `local` extra installed and `KOKORO_ENABLED=true` |
| `openai` | OpenAI cloud | `OPENAI_API_KEY` set |

Only usable providers are registered. Requesting any other returns an error listing the
available ones. `DEFAULT_PROVIDER` / `DEFAULT_VOICE` in `.env` override the code default.

## Setup

```bash
uv sync --extra dev --extra local --extra ui   # drop "local" to skip Kokoro (pulls CPU torch)
copy .env.example .env                          # then add any API keys you want
```

The first Kokoro call downloads the model from Hugging Face, which takes a few minutes once.
Compressed output formats (mp3/opus/aac/flac) need **ffmpeg** on PATH. `wav` and `pcm`
always work.

## Run

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
uv run streamlit run streamlit_app.py      # optional browser test UI
```

- Swagger UI: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

Run **one** server process (no `--workers N`): background jobs live in the process's
memory, so several workers can't see or cancel each other's jobs.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/speech` | One-shot synthesis, returns an audio file |
| POST | `/v1/speech/stream` | Chunked streaming, sentence-by-sentence |
| POST | `/v1/podcast` | Submit a multi-speaker script, returns `job_id` |
| GET | `/v1/jobs` | List recent jobs |
| GET | `/v1/jobs/{id}` | Job status and progress (`audio_url` when done) |
| GET | `/v1/jobs/{id}/audio` | Download the finished podcast |
| POST | `/v1/jobs/{id}/cancel` | Cancel a queued or running job (409 if already finished) |
| GET | `/v1/voices` | List voices (`?provider=`, `?gender=female\|male`) |
| GET | `/health` | Liveness |

### Speech

```bash
curl -X POST http://127.0.0.1:8000/v1/speech \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Hello world\",\"provider\":\"kokoro\",\"voice\":\"af_heart\"}" \
  --output hello.wav
```

Fields: `text` (required), `voice`, `provider`, `format` (default `wav`), `speed`,
`instructions` (style hint, where the provider supports it).

### Streaming (chatbot)

Same body as `/v1/speech`, plus `max_sentence_length` (default 200). The text is split
into sentences and each one is streamed as soon as it's synthesized, so playback can start
early.

```bash
curl -X POST http://127.0.0.1:8000/v1/speech/stream \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Hello world. How are you?\"}" --output stream.wav
```

### Voices

```bash
curl "http://127.0.0.1:8000/v1/voices?provider=kokoro&gender=male"
# -> {"default_provider": "kokoro", "providers": ["kokoro", "openai"],
#     "voices": [{"id": "am_michael", "name": "am_michael", "provider": "kokoro",
#                 "language": "en-US", "gender": "male"}, ...]}
```

Kokoro's genders come from the voice id (`af_`/`bf_` female, `am_`/`bm_` male). OpenAI
publishes no gender metadata, so its genders are perceived ones, and `alloy` is `null`
(neutral).

### Podcast

Send either `turns` (`[{"speaker": "Alice", "text": "..."}]`) or a plain-text `script`
(`Alice: ...` lines). `voices` (a speaker → voice map) is optional.

Speakers you don't map are **cast automatically by gender**:
- The gender is guessed from the name, using common English and Vietnamese names,
  Vietnamese middle names such as `Thị` and `Văn`, and titles such as `Mr` and `Chị`.
  Examples: Alice → female, Bob → male, `Nguyễn Thị Lan` → female.
- Each speaker gets a different voice of that gender, so two women and one man get three
  distinct voices. A speaker keeps the same voice for the whole episode.
- If a name gives no clue (`Host`, `Guest`), the speaker gets whichever gender keeps the
  cast balanced.
- You can set a gender yourself with `genders`, for example
  `{"Host": "female"}`. `voices` still wins over everything.
- The chosen map is returned as `voices` in `GET /v1/jobs/{id}`.

```bash
curl -X POST http://127.0.0.1:8000/v1/podcast \
  -H "Content-Type: application/json" \
  -d "{\"script\":\"Alice: Hi Bob.\nBob: Hi Alice!\",\"provider\":\"kokoro\"}"
# -> {"job_id": "...", "status": "queued", ...}
curl http://127.0.0.1:8000/v1/jobs/<job_id>
curl http://127.0.0.1:8000/v1/jobs/<job_id>/audio --output podcast.wav
```

### Errors

Every error is JSON with a `detail` message. Provider failures also have a `kind`:

| Status | When |
|---|---|
| 400 | Bad input: unknown provider or voice (`GET /v1/voices` lists valid ones), unsupported format, `flac` on the stream endpoint, or the provider rejected the text (`kind: bad_request`) |
| 502 | The provider failed (`upstream`) or rejected the server's API key (`auth`) |
| 503 | The provider is rate limiting (`rate_limited`); `Retry-After` says when to retry |
| 504 | The provider timed out (`timeout`) |

OpenAI calls time out after `OPENAI_TIMEOUT` seconds (default 30) per attempt. The SDK
retries network errors, 429 and 5xx up to `OPENAI_MAX_RETRIES` times (default 2) with
backoff, so the worst case is about 95 s. Bad input and bad keys are not retried.

On the stream endpoint the server waits for the first audio chunk before answering, so
most failures still come back as a proper error status. If a later sentence fails, the
connection is dropped (the client sees an incomplete response); error text is never
mixed into the audio.

## Reliability notes

- **Kokoro runs one synthesis at a time** (`KOKORO_MAX_CONCURRENCY`, default 1), since
  each one already uses every CPU core. Extra requests wait their turn.
- **Podcast jobs queue:** `MAX_CONCURRENT_JOBS` (default 1) render at once, and the rest
  stay `queued`. Cancelling a queued job is immediate. A running job stops after the
  sentence it is synthesizing.
- **Restarts:** jobs a crashed or killed server left `queued`/`running` are marked
  `error` ("Interrupted by a server restart") on the next start.
- **Cleanup:** finished jobs and their audio are deleted after `JOBS_RETENTION_DAYS`
  (default 7; 0 keeps everything), at startup and hourly. Only job files inside
  `OUTPUT_DIR` are removed.
- **Cache:** one-shot speech and podcast turns are cached on disk in `CACHE_DIR` (default
  `./output/cache`, up to `CACHE_MAX_MB`, default 512, least recently used evicted), so
  repeats are nearly instant and survive restarts. `POST /v1/speech` reports
  `X-Cache: hit|miss`. Streaming is not cached. Set `CACHE_ENABLED=false` to turn it off.
- **Logging:** one line per request plus provider, retry, cache and job events. Each line
  carries the request id, which is also returned as `X-Request-ID` (send your own to trace
  a call). Text is never logged, only its length. `LOG_LEVEL=DEBUG` adds cache hits and
  stream shutdowns.

All settings are listed in `.env.example`.

## Examples

```bash
uv run python examples/chatbot_stream.py                                   # simulated LLM tokens -> streaming audio
uv run python examples/generate_podcast.py examples/script_trio.txt --provider kokoro   # 3 speakers (2 female, 1 male)
uv run python examples/generate_podcast.py examples/script.json --provider kokoro   # -> output/script.wav
```

## Test

```bash
uv run pytest        # hermetic: providers are mocked, no API keys needed
uv run ruff check .
```
