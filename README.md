# TTS Engine

A Python/FastAPI backend that wraps cloud TTS providers behind one API. It targets
two consumers: **chatbot** (low-latency streaming speech) and **podcast** (long-form,
multi-speaker audio files). See `CLAUDE.md` for the full design and roadmap.

**Status:** Phase 2 — scaffold + OpenAI provider + one-shot `POST /v1/speech` and
chunked streaming `POST /v1/speech/stream` (sentence-by-sentence).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell:  .venv\Scripts\Activate.ps1
pip install -e ".[dev]"

copy .env.example .env             # then edit .env and add OPENAI_API_KEY
```

## Run

```bash
uvicorn app.main:app --reload
```

- Swagger UI: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## Synthesize speech

```bash
curl -X POST http://127.0.0.1:8000/v1/speech \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Hello world\",\"voice\":\"alloy\"}" \
  --output hello.mp3
```

Request fields: `text` (required), `voice`, `provider`, `format` (mp3/wav/opus/aac/flac/pcm),
`speed` (0.25–4.0, tts-1 models only), `instructions` (style hint).

## Stream speech (chatbot)

`POST /v1/speech/stream` splits the text into sentences and streams audio chunks
back so the client can start playing before the whole clip is synthesized.

```bash
curl -X POST http://127.0.0.1:8000/v1/speech/stream \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Hello world. How are you?\",\"voice\":\"alloy\"}" \
  --output stream.mp3
```

Same fields as `/v1/speech`, plus `max_sentence_length` (default 200, 20–2000).
The response is a chunked HTTP stream of raw audio bytes in the requested format.

For an end-to-end demo (simulated LLM tokens -> chunker -> engine -> mp3), run:

```bash
uv run python examples/chatbot_stream.py
```

## Test

```bash
pytest        # hermetic; uses a fake provider, no API key needed
ruff check .
```

## Roadmap

Phase 3 podcast pipeline · Phase 4 ElevenLabs + voices · Phase 5 polish.
