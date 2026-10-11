"""Shared test fixtures.

Every test runs against a throwaway output directory and job database so the
suite never writes into the repo's ``./output`` or reuses state between runs.
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.core import audio


@pytest.fixture(autouse=True)
def isolated_output(tmp_path, monkeypatch, request):
    out_dir = tmp_path / "output"
    monkeypatch.setenv("OUTPUT_DIR", str(out_dir))
    monkeypatch.setenv("JOBS_DB", str(out_dir / "jobs.db"))
    monkeypatch.setenv("CACHE_DIR", str(out_dir / "cache"))
    # Never load the real Kokoro model at app startup in tests.
    monkeypatch.setenv("KOKORO_PRELOAD", "false")
    # Don't import the real torch to set its thread count when a fake pipeline loads.
    monkeypatch.setenv("KOKORO_TORCH_THREADS", "0")
    # Podcast loudness/encoding must not depend on whether this machine has ffmpeg:
    # the pipeline acts as if it's missing unless a test turns it on (or uses the
    # `real_ffmpeg` marker, which needs the binary).
    if "real_ffmpeg" not in request.keywords:
        monkeypatch.setattr(audio, "HAS_FFMPEG", False)
    get_settings.cache_clear()
    yield out_dir
    get_settings.cache_clear()
