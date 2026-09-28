"""Shared test fixtures.

Every test runs against a throwaway output directory and job database so the
suite never writes into the repo's ``./output`` or reuses state between runs.
"""

from __future__ import annotations

import pytest

from app.config import get_settings


@pytest.fixture(autouse=True)
def isolated_output(tmp_path, monkeypatch):
    out_dir = tmp_path / "output"
    monkeypatch.setenv("OUTPUT_DIR", str(out_dir))
    monkeypatch.setenv("JOBS_DB", str(out_dir / "jobs.db"))
    monkeypatch.setenv("CACHE_DIR", str(out_dir / "cache"))
    get_settings.cache_clear()
    yield out_dir
    get_settings.cache_clear()
