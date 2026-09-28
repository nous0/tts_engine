"""On-disk cache of synthesized audio, keyed by everything that shapes the output.

Local Kokoro takes seconds per sentence on CPU, so repeating a phrase, re-rendering
a podcast, or re-trying a voice shouldn't pay that again. Entries are plain files
(``<sha256>.<format>``) in ``cache_dir``, so the cache survives restarts.

Rules that keep it safe:

- **Keys use effective values**: the resolved voice, and a provider ``cache_tag``
  that names the real model (OpenAI's configured model, Kokoro's package version).
  Changing the model or upgrading Kokoro therefore misses instead of replaying old
  audio. Bump ``CACHE_VERSION`` when the audio pipeline itself changes.
- **Writes are atomic** (temp file + ``os.replace``), so a crash never leaves a
  truncated file that would then be served forever.
- **The cache can't break a request**: any ``OSError`` (full disk, read-only dir,
  or Windows refusing to replace/delete a file another request is reading) just
  means a miss or a skipped write.
- **Bounded size**: least-recently-used files (by mtime, refreshed on every hit)
  are deleted once the total exceeds ``max_bytes``.

All file work runs in a thread so the event loop never blocks on disk.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path

log = logging.getLogger("app.cache")

CACHE_VERSION = 1

# Temp files older than this are leftovers from a crash and get deleted by prune().
_STALE_TMP_S = 3600.0


class FileCache:
    def __init__(self, directory: str | Path, max_bytes: int) -> None:
        self._dir = Path(directory)
        self._max_bytes = max(0, max_bytes)
        self._approx_bytes = 0  # exact after prune(), then grows with each put

    @staticmethod
    def make_key(**parts: object) -> str:
        payload = json.dumps(
            {"cache_version": CACHE_VERSION, **parts}, sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def get(self, key: str, fmt: str) -> bytes | None:
        return await asyncio.to_thread(self._read, self._path(key, fmt))

    async def put(self, key: str, fmt: str, data: bytes) -> None:
        if not data:
            return
        written = await asyncio.to_thread(self._write, key, fmt, data)
        if written:
            self._approx_bytes += len(data)
            if self._approx_bytes > self._max_bytes:
                await self.prune()

    async def prune(self) -> int:
        """Shrink the cache to ``max_bytes``; returns the size left, in bytes."""
        self._approx_bytes = await asyncio.to_thread(self._prune)
        return self._approx_bytes

    def _path(self, key: str, fmt: str) -> Path:
        return self._dir / f"{key}.{fmt}"

    def _read(self, path: Path) -> bytes | None:
        try:
            data = path.read_bytes()
        except OSError:
            return None
        try:
            os.utime(path)  # mark as recently used
        except OSError:
            pass
        return data

    def _write(self, key: str, fmt: str, data: bytes) -> bool:
        tmp = self._dir / f"{key}.{fmt}.{uuid.uuid4().hex}.tmp"
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(data)
            os.replace(tmp, self._path(key, fmt))
            return True
        except OSError as exc:
            log.debug("cache write skipped: %s", exc)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return False

    def _prune(self) -> int:
        try:
            entries = list(os.scandir(self._dir))
        except OSError:
            return 0
        now = time.time()
        files: list[tuple[float, int, str]] = []
        for entry in entries:
            try:
                stat = entry.stat()
            except OSError:
                continue
            if entry.name.endswith(".tmp"):
                if now - stat.st_mtime > _STALE_TMP_S:
                    _unlink(entry.path)
                continue
            if entry.is_file():
                files.append((stat.st_mtime, stat.st_size, entry.path))

        total = sum(size for _, size, _ in files)
        removed = 0
        for _, size, path in sorted(files):  # oldest first
            if total <= self._max_bytes:
                break
            if _unlink(path):
                total -= size
                removed += 1
        if removed:
            log.info("cache pruned %d file(s); %.1f MB left", removed, total / 1e6)
        return total


def _unlink(path: str) -> bool:
    try:
        os.unlink(path)
        return True
    except OSError:  # e.g. Windows: another request is reading it; try next time
        return False
