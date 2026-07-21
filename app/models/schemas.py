"""Request/response models for the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AudioFormat = Literal["mp3", "wav", "opus", "aac", "flac", "pcm"]


class SpeechRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    voice: str | None = Field(None, description="Provider voice id; falls back to server default.")
    provider: str | None = Field(None, description="Provider name; falls back to server default.")
    format: AudioFormat = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)
    instructions: str | None = Field(
        None, description="Optional style/tone hint (supported by some providers)."
    )


class SpeechStreamRequest(BaseModel):
    """Request body for the streaming speech endpoint.

    The full text is sent up-front; the server splits it into sentences and
    streams audio chunks back so the client can start playing before the whole
    clip is synthesized.
    """

    text: str = Field(..., min_length=1, max_length=50000)
    voice: str | None = Field(None, description="Provider voice id; falls back to server default.")
    provider: str | None = Field(None, description="Provider name; falls back to server default.")
    format: AudioFormat = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)
    instructions: str | None = Field(
        None, description="Optional style/tone hint (supported by some providers)."
    )
    max_sentence_length: int = Field(
        200, ge=20, le=2000, description="Max chars per synthesized sentence chunk."
    )
