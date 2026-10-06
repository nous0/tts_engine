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
- For real-time speech with Kokoro, set Windows' power mode to **Best performance**:
  "Best power efficiency" can make synthesis ~2x slower, which is slower than real time.
- Health: http://127.0.0.1:8000/health returns readiness per provider, e.g.
  `{"status": "ok", "default_provider": "kokoro", "providers": {"kokoro": "ready", "openai": "ready"}}`.
  Kokoro loads in the background at startup (`KOKORO_PRELOAD=true`), so it shows
  `loading` for about 10 s, then `ready`. Other states are `idle` (not loaded yet) and
  `error`.

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
| WS | `/v1/speech/ws` | Real-time chatbot voice: send LLM tokens, get audio per sentence |
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

`fast_start: true` lets the first chunk end at a comma once it's at least 24 characters
long, so a long first sentence starts playing sooner. It's off by default because the
first sentence may sound slightly split. Measured on Kokoro/CPU, time to first audio
went from 3.5 to 2.0 s for a 79-character first sentence, and from 5.2 to 3.0 s for a
171-character one.

### Real-time chatbot voice (WebSocket)

`WS /v1/speech/ws` is for a chatbot that talks **while the LLM is still writing**. The
client forwards LLM token deltas as they arrive. The server cuts them into sentences
and sends back audio for each sentence right away, and the client can cancel speech at
any time (barge-in).

| Direction | Message |
|---|---|
| client → server | `{"type": "start", "provider"?, "voice"?, "speed"?, "chunking"?: "adaptive", "cache"?: true, "normalize"?: true}` (must be first) · `{"type": "text", "text": "..."}` (any size) · `{"type": "flush"}` (end of turn) · `{"type": "cancel"}` (stop speaking now) |
| server → client | `{"type": "ready", "sample_rate": 24000, "channels": 1, "sample_width": 2, "format": "pcm", ...}` · `{"type": "sentence", "index", "turn", "chars", "cached", "text", "boundary"}` · **binary frames: raw PCM** (s16le, mono) · `{"type": "done", "turn"}` · `{"type": "cancelled"}` · `{"type": "error", "detail", "status", "kind"?, "turn"?}` |

- Each turn ends with exactly one of `done`, `error` (with `turn`), or `cancelled`.
  `cancelled` ends every pending turn.
- A bad `start` (unknown provider or voice, malformed message) gets an `error` and the
  connection closes with code 1008. Other bad messages get an `error`, and the session
  keeps going.
- `chunking` decides how text is cut into chunks to synthesize:
  - **`adaptive`** (default): the first chunk of a turn is short, so audio starts fast.
    Each next chunk is sized so that it is synthesized before the audio already queued
    runs out (no stalls), and grows as that queue builds. Once enough is queued, whole
    sentences are used again.
  - `fast_start`: whole sentences, but the first one may end at a comma.
  - `sentence`: whole sentences.

  Adaptive learns the provider's speed as it goes (shared across sessions), so it adapts
  to a slower or busier CPU. Chunks are cached like `/v1/speech/stream` (`"cache": false`
  to skip).
- `normalize` is on by default here: markdown, links and emoji in LLM output are removed
  before speaking (`**bold**` → "bold", a URL → "link"). The HTTP stream has the same
  `normalize` field, off by default.
- Chunks are joined smoothly: each chunk's leading and trailing silence (about 0.8 s
  with Kokoro) is trimmed, and a natural pause is inserted for how it ended (sentence
  300 ms, comma 150 ms, line 450 ms). The same applies to pcm/wav on the HTTP stream;
  turn it off with `STREAM_SHAPE_AUDIO=false`. The `sentence` event also carries the
  spoken `text` and its `boundary` (`sentence`, `clause`, `line`, `word` or `end`).
- Sentence splitting understands decimals (`3.5`), prices (`$4.99`), domains
  (`example.com`), abbreviations (`Dr.`, `e.g.`), initials (`U.S.`) and ellipses, so
  none of them break a sentence in two.
- Browser clients (which send an `Origin` header) are refused unless the origin is listed
  in `WS_ALLOWED_ORIGINS`. Server-side clients are always allowed. A turn is capped at
  `WS_MAX_TURN_CHARS` (default 10 000).

Try it with the reference client, which sends a demo answer word by word, like an LLM:

```bash
uv run python examples/ws_client.py                      # prints a timeline, saves output/ws_client.wav
uv run python examples/ws_client.py --cancel-after 3     # barge-in
uv run python examples/ws_client.py --chunking sentence --no-cache
```

Measured with `scripts/bench_latency.py --mode ws` (Kokoro/CPU, a fake LLM at 15
words/s, 8 chatbot-style answers). *Hear* is the time from the first sentence being
complete to the first audible sound:

| chunking | hear (median / max) | total stalls | dead air ≥ 0.4 s |
|---|---|---|---|
| sentence | 2.47 / 7.93 s | 5.82 s | 2 |
| fast_start | 1.08 / 3.62 s | 4.78 s | 0 |
| **adaptive** | **1.17 / 1.95 s** | **0.13 s** | **0** |

Kokoro needs a CPU that synthesizes faster than real time. These numbers are from a
laptop i7-1360P when it ran at ~27 ms of synthesis per character (speech is ~62 ms per
character). In Windows' "Best power efficiency" power mode, the same CPU dropped to
55–62 ms/char: then synthesis is barely real-time, and every mode stalls. Use
**Best performance** for real-time speech, or OpenAI. Cached answers start in about
10 ms.

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
- **Cache:** one-shot speech, podcast turns, and **each sentence of a pcm/wav stream** are
  cached on disk in `CACHE_DIR` (default `./output/cache`, up to `CACHE_MAX_MB`, default
  512, least recently used evicted). Repeats are nearly instant and survive restarts: a
  fully cached stream starts in about 10 ms. Streamed sentences and podcast turns share
  entries. `POST /v1/speech` reports `X-Cache: hit|miss`. Send `Cache-Control: no-cache`
  to skip cached audio for one request (the fresh result is still stored). Set
  `CACHE_ENABLED=false` to turn the cache off. Compressed stream formats (mp3/opus/aac)
  aren't cached.
- **Logging:** one line per request plus provider, retry, cache and job events. Each line
  carries the request id, which is also returned as `X-Request-ID` (send your own to trace
  a call). Text is never logged, only its length. `LOG_LEVEL=DEBUG` adds cache hits and
  stream shutdowns.

All settings are listed in `.env.example`.

## Voice chat demo

A chatbot that speaks while it is still writing, in the browser (Gradio). It needs the
`demo` extra and **ffmpeg on PATH**, because Gradio's player streams AAC.

```bash
uv sync --extra demo
uv run uvicorn app.main:app --port 8000      # terminal 1: the TTS server
uv run python examples/voice_chat_ui.py      # terminal 2: open http://127.0.0.1:7860
```

- Type a question. The answer streams into the chat and is spoken as it arrives.
  **Stop**, or a new question, cancels the speech right away (barge-in).
- **Fake LLM** (canned answers, no key) is on automatically when the OpenAI key doesn't
  work; with a valid `OPENAI_API_KEY` it chats for real (`DEMO_CHAT_MODEL`, default
  `gpt-4o-mini`). The demo is local only (`127.0.0.1`, no public share link).
- Gradio's streaming player adds about **0.8 s** before playback, because it re-encodes
  the stream for the browser. Measured: the TTS server's first audio came 0.45 s after
  Enter, and playback 1.3 s after Enter. The first question after startup takes about
  2 s, while ffmpeg starts.

## Examples

```bash
uv run python examples/ws_client.py                                         # fake LLM tokens -> WebSocket -> output/ws_client.wav
uv run python examples/voice_chat_ui.py                                     # voice chat in the browser (demo extra)
uv run python examples/generate_podcast.py examples/script_trio.txt --provider kokoro   # 3 speakers (2 female, 1 male)
uv run python examples/generate_podcast.py examples/script.json --provider kokoro   # -> output/script.wav
```

## Test

```bash
uv run pytest        # hermetic: providers are mocked, no API keys needed
uv run ruff check .
```

End-to-end smoke test against a running server, with real Kokoro and no OpenAI calls.
It exits 1 if any check fails:

```bash
uv run uvicorn app.main:app --port 8000          # terminal 1
uv run python scripts/smoke_test.py              # terminal 2: 22 checks, about 1 minute
uv run python scripts/smoke_test.py --skip-jobs  # without the job queue/cancel checks
uv run python scripts/bench_latency.py           # streaming latency: TTFA, RTF, stalls
uv run python scripts/bench_latency.py --mode ws --chunking all --wps 5,15,40   # chatbot view

# Restart recovery (two steps, with a server restart in between):
uv run python scripts/smoke_test.py --leave-running-job
uv run python scripts/smoke_test.py --check-interrupted <job_id>
```

It checks health, request ids, cache miss/hit, a single WAV header when streaming, a
client hanging up mid-stream, error statuses, concurrency, the job queue and cancel. It
also saves `output/smoke_stream.wav`: listen to it to confirm there are no clicks between
sentences.
```
