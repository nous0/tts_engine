"""Streamlit test UI for the TTS engine (speech + streaming, all providers).

Talks to the running FastAPI server over HTTP so it exercises the real API:

- One-shot synthesis  -> POST /v1/speech
- Low-latency stream  -> POST /v1/speech/stream  (measures time-to-first-byte)

Run (with the API server already up in another terminal):

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import time

import httpx
import streamlit as st

# Known voices per provider (imported for the picker; free-text also allowed).
try:
    from app.core.providers.openai import _OPENAI_VOICES as OPENAI_VOICES
except Exception:  # pragma: no cover
    OPENAI_VOICES = [
        "alloy", "ash", "ballad", "coral", "echo",
        "fable", "nova", "onyx", "sage", "shimmer",
    ]

try:
    from app.core.providers.kokoro import KOKORO_VOICES
except Exception:  # pragma: no cover
    KOKORO_VOICES = ["af_heart", "af_bella", "am_adam", "am_michael", "bf_emma", "bm_george"]

VOICES: dict[str, list[str]] = {
    "kokoro": list(KOKORO_VOICES),
    "openai": list(OPENAI_VOICES),
}
DEFAULT_VOICES = {"kokoro": "af_heart", "openai": "alloy"}
FORMATS = ["wav", "mp3", "opus", "aac", "flac", "pcm"]
MEDIA = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "pcm": "audio/L16",
}

st.set_page_config(page_title="TTS Engine Tester", page_icon="🔊", layout="centered")
st.title("🔊 TTS Engine — Test UI")
st.caption("One-shot synthesis and low-latency streaming · kokoro / openai")

# --------------------------------------------------------------------------- #
# Sidebar: connection + synthesis options
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Config")
    base_url = st.text_input("API base URL", "http://127.0.0.1:8000").rstrip("/")

    provider_label = st.selectbox(
        "Provider", ["(server default)", "kokoro", "openai"],
        help="Server default comes from DEFAULT_PROVIDER in .env.",
    )
    provider = None if provider_label.startswith("(") else provider_label

    if provider is None:
        # Let the server pick its default voice too: a voice from another provider's
        # catalog would be rejected.
        voice = None
        st.caption("Voice: server default (DEFAULT_VOICE)")
    else:
        voice_opts = VOICES[provider]
        default_voice = DEFAULT_VOICES.get(provider, voice_opts[0])
        index = voice_opts.index(default_voice) if default_voice in voice_opts else 0
        voice = st.selectbox("Voice", voice_opts, index=index)

    fmt = st.selectbox(
        "Format", FORMATS,
        help="wav/pcm work without ffmpeg; mp3/opus/aac/flac need ffmpeg for Kokoro.",
    )
    speed = st.slider(
        "Speed", 0.25, 4.0, 1.0, 0.05, help="Used by Kokoro and OpenAI tts-1."
    )
    instructions = st.text_input("Instructions (style)", "", placeholder="e.g. Say cheerfully")

    st.divider()
    if st.button("Check /health", use_container_width=True):
        try:
            r = httpx.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                st.success(f"health → {r.json()}")
            else:
                st.error(f"{r.status_code}: {r.text}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Cannot reach API: {exc}")


def build_payload(text: str, extra: dict | None = None) -> dict:
    """Assemble a request body from the current sidebar options."""
    body: dict = {"text": text, "format": fmt, "speed": speed}
    if provider:
        body["provider"] = provider
    if voice:
        body["voice"] = voice
    if instructions.strip():
        body["instructions"] = instructions.strip()
    if extra:
        body.update(extra)
    return body


def offer_playback_and_download(data: bytes, filename: str) -> None:
    if fmt == "pcm":
        st.info("PCM is raw audio — not playable in the browser. Use Download, or pick 'wav'.")
    else:
        st.audio(data, format=MEDIA.get(fmt, "audio/wav"))
    st.download_button("⬇️ Download", data, file_name=filename, mime=MEDIA.get(fmt))


tab_oneshot, tab_stream = st.tabs(["🗣️ One-shot  /v1/speech", "⚡ Streaming  /v1/speech/stream"])

# --------------------------------------------------------------------------- #
# One-shot synthesis
# --------------------------------------------------------------------------- #
with tab_oneshot:
    text1 = st.text_area("Text", "Hello world. This is a one-shot synthesis test.", height=120)
    if st.button("Synthesize", type="primary", key="btn_oneshot"):
        if not text1.strip():
            st.warning("Please enter some text.")
        else:
            try:
                t0 = time.perf_counter()
                resp = httpx.post(f"{base_url}/v1/speech", json=build_payload(text1), timeout=120)
                elapsed = time.perf_counter() - t0
            except Exception as exc:  # noqa: BLE001
                st.error(f"Request failed: {exc}")
            else:
                if resp.status_code == 200:
                    st.success(f"OK — {len(resp.content):,} bytes in {elapsed:.2f}s")
                    offer_playback_and_download(resp.content, f"speech.{fmt}")
                else:
                    st.error(f"{resp.status_code}: {resp.text}")

# --------------------------------------------------------------------------- #
# Streaming synthesis (time-to-first-byte)
# --------------------------------------------------------------------------- #
with tab_stream:
    st.caption(
        "Text is split into sentences server-side; audio chunks stream back. "
        "The **first byte** should arrive well before the whole clip is done."
    )
    text2 = st.text_area(
        "Text (multiple sentences show streaming best)",
        "Hello! This is the first sentence. Here comes a second one. "
        "And a third sentence so you can watch chunks arrive over time.",
        height=140,
    )
    max_len = st.slider("max_sentence_length", 20, 2000, 200, 10)

    if st.button("Stream", type="primary", key="btn_stream"):
        if not text2.strip():
            st.warning("Please enter some text.")
        else:
            buf = bytearray()
            first_byte: float | None = None
            n_chunks = 0
            live = st.empty()
            try:
                t0 = time.perf_counter()
                with httpx.stream(
                    "POST",
                    f"{base_url}/v1/speech/stream",
                    json=build_payload(text2, {"max_sentence_length": max_len}),
                    timeout=120,
                ) as resp:
                    if resp.status_code != 200:
                        st.error(f"{resp.status_code}: {resp.read().decode(errors='ignore')}")
                    else:
                        for chunk in resp.iter_bytes():
                            if not chunk:
                                continue
                            if first_byte is None:
                                first_byte = time.perf_counter() - t0
                            buf.extend(chunk)
                            n_chunks += 1
                            live.write(
                                f"chunk {n_chunks} · {len(buf):,} bytes · "
                                f"first byte @ {first_byte:.2f}s · "
                                f"elapsed {time.perf_counter() - t0:.2f}s"
                            )
                        total = time.perf_counter() - t0

                        col1, col2, col3 = st.columns(3)
                        col1.metric("First byte", f"{first_byte:.2f}s" if first_byte else "—")
                        col2.metric("Total time", f"{total:.2f}s")
                        col3.metric("Chunks", str(n_chunks))

                        if first_byte is not None and (total - first_byte) > 0.05:
                            st.success(
                                "✅ Audio started arriving before the stream finished — "
                                "streaming works."
                            )
                        if buf:
                            offer_playback_and_download(bytes(buf), f"stream.{fmt}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Stream failed: {exc}")
