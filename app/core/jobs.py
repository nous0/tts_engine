"""Background job store for long-running renders (SQLite + in-process worker).

Podcast renders take minutes, so ``POST /v1/podcast`` returns a job id and the
work continues on an asyncio task. State lives in SQLite so it survives a
reload of the client (and can be inspected while running); the stdlib
``sqlite3`` driver is used from a worker thread, which keeps the dependency
list unchanged.

Scale-out path: swap this module for Celery/RQ + object storage. Nothing
outside it knows how jobs are stored.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    status       TEXT NOT NULL,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    progress     INTEGER NOT NULL DEFAULT 0,
    total        INTEGER NOT NULL DEFAULT 0,
    result_path  TEXT,
    error        TEXT,
    meta         TEXT
);
"""


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass(slots=True)
class Job:
    id: str
    kind: str
    status: JobStatus
    created_at: float
    updated_at: float
    progress: int
    total: int
    result_path: str | None
    error: str | None
    meta: dict[str, Any]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Job:
        return cls(
            id=row["id"],
            kind=row["kind"],
            status=JobStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            progress=row["progress"],
            total=row["total"],
            result_path=row["result_path"],
            error=row["error"],
            meta=json.loads(row["meta"]) if row["meta"] else {},
        )


class JobStore:
    """SQLite-backed job records plus an in-process asyncio worker."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()

    async def connect(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # The connection is shared across worker threads but every access is
        # serialized by ``self._lock``.
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        conn.commit()
        self._conn = conn

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    async def create(self, kind: str, total: int = 0, meta: dict | None = None) -> Job:
        now = time.time()
        job = Job(
            id=uuid.uuid4().hex,
            kind=kind,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            progress=0,
            total=total,
            result_path=None,
            error=None,
            meta=meta or {},
        )
        await self._execute(
            "INSERT INTO jobs (id, kind, status, created_at, updated_at, progress,"
            " total, result_path, error, meta) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                job.id, job.kind, job.status.value, job.created_at, job.updated_at,
                job.progress, job.total, None, None, json.dumps(job.meta),
            ),
        )
        return job

    async def get(self, job_id: str) -> Job | None:
        rows = await self._query("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return Job.from_row(rows[0]) if rows else None

    async def list(self, limit: int = 50) -> list[Job]:
        rows = await self._query(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [Job.from_row(r) for r in rows]

    async def update(self, job_id: str, **fields: Any) -> None:
        """Patch a job row. Unknown columns are rejected to catch typos early."""
        allowed = {"status", "progress", "total", "result_path", "error", "meta"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Unknown job fields: {sorted(unknown)}")
        if not fields:
            return
        if "status" in fields:
            fields["status"] = JobStatus(fields["status"]).value
        if "meta" in fields:
            fields["meta"] = json.dumps(fields["meta"])
        fields["updated_at"] = time.time()
        assignments = ", ".join(f"{k} = ?" for k in fields)
        await self._execute(
            f"UPDATE jobs SET {assignments} WHERE id = ?", (*fields.values(), job_id)
        )

    def spawn(self, job_id: str, work: Callable[[], Awaitable[None]]) -> None:
        """Run ``work`` in the background, recording failures on the job."""

        async def runner() -> None:
            await self.update(job_id, status=JobStatus.RUNNING)
            try:
                await work()
            except asyncio.CancelledError:
                await self.update(
                    job_id, status=JobStatus.ERROR, error="Job cancelled (server shutdown)."
                )
                raise
            except Exception as exc:  # noqa: BLE001 - recorded on the job for the client
                await self.update(job_id, status=JobStatus.ERROR, error=str(exc))

        task = asyncio.create_task(runner())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def wait(self, job_id: str, timeout: float = 60.0) -> Job | None:
        """Block until a job leaves the queued/running states (used by tests/CLIs)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = await self.get(job_id)
            if job is None or job.status in (JobStatus.DONE, JobStatus.ERROR):
                return job
            await asyncio.sleep(0.05)
        return await self.get(job_id)

    async def _execute(self, sql: str, params: tuple) -> None:
        conn = self._require_conn()

        def run() -> None:
            with conn:
                conn.execute(sql, params)

        async with self._lock:
            await asyncio.to_thread(run)

    async def _query(self, sql: str, params: tuple) -> list[sqlite3.Row]:
        conn = self._require_conn()

        def run() -> list[sqlite3.Row]:
            return conn.execute(sql, params).fetchall()

        async with self._lock:
            return await asyncio.to_thread(run)

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("JobStore is not connected; call connect() first.")
        return self._conn
