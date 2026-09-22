# TTS Engine

A Python/FastAPI backend that wraps TTS providers behind one API. It targets two
consumers: **chatbot** (low-latency streaming speech) and **podcast** (long-form,
multi-speaker audio files rendered as background jobs).

**Status:** Phases 1–3 done, Phase 4 partial (Kokoro + OpenAI providers;
`GET /v1/voices` not built yet), Phase 5 polish not started. See `CONTEXT.md` for the
current to-do list.

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

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/speech` | One-shot synthesis, returns an audio file |
| POST | `/v1/speech/stream` | Chunked streaming, sentence-by-sentence |
| POST | `/v1/podcast` | Submit a multi-speaker script, returns `job_id` |
| GET | `/v1/jobs` | List recent jobs |
| GET | `/v1/jobs/{id}` | Job status and progress (`audio_url` when done) |
| GET | `/v1/jobs/{id}/audio` | Download the finished podcast |
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

### Podcast

Send either `turns` (`[{"speaker": "Alice", "text": "..."}]`) or a plain-text `script`
(`Alice: ...` lines). `voices` (a speaker → voice map) is optional. Speakers you don't map
get voices from the provider's catalog automatically.

```bash
curl -X POST http://127.0.0.1:8000/v1/podcast \
  -H "Content-Type: application/json" \
  -d "{\"script\":\"Alice: Hi Bob.\nBob: Hi Alice!\",\"provider\":\"kokoro\"}"
# -> {"job_id": "...", "status": "queued", ...}
curl http://127.0.0.1:8000/v1/jobs/<job_id>
curl http://127.0.0.1:8000/v1/jobs/<job_id>/audio --output podcast.wav
```

## Examples

```bash
uv run python examples/chatbot_stream.py                                   # simulated LLM tokens -> streaming audio
uv run python examples/generate_podcast.py examples/script.json --provider kokoro   # -> output/script.wav
```

## Test

```bash
uv run pytest        # hermetic: providers are mocked, no API keys needed
uv run ruff check .
```
