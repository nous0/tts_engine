"""Background job store for long-running renders (SQLite + in-process worker).

Podcast renders take minutes, so ``POST /v1/podcast`` returns a job id and the
work continues on an asyncio task. State lives in SQLite so it survives a
reload of the client (and can be inspected while running); the stdlib
``sqlite3`` driver is used from a worker thread, which keeps the dependency
list unchanged.

Jobs really queue: at most ``max_concurrent`` run at once (default 1, since a local
Kokoro render already uses every core); the rest stay ``queued`` until a slot frees.
On startup, jobs left ``queued``/``running`` by a previous process are marked as
errors, and finished jobs older than the retention period are deleted with their
files, at startup and then hourly.

Assumes a single server process: tasks live in this process's memory, so with
several workers one could neither cancel nor see another's jobs, and startup
recovery would fail jobs that another worker is still running.

Scale-out path: swap this module for Celery/RQ + object storage. Nothing
outside it knows how jobs are stored.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
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


log = logging.getLogger("app.jobs")

CLEANUP_INTERVAL_S = 3600.0


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


FINISHED = frozenset({JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED})


class JobNotFound(LookupError):
    """No job with that id."""


class JobNotActive(Exception):
    """The job already finished, so it can't be cancelled."""

    def __init__(self, job: Job) -> None:
        self.job = job
        super().__init__(f"Job '{job.id}' is already {job.status.value}.")


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

    def __init__(self, db_path: str | Path, max_concurrent: int = 1) -> None:
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(max(1, max_concurrent))
        self._tasks: dict[str, asyncio.Task] = {}
        self._user_cancelled: set[str] = set()
        self._maintenance: asyncio.Task | None = None

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
        if self._maintenance is not None:
            self._maintenance.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._maintenance
            self._maintenance = None
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
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
            log.info("job %s -> %s", job_id, fields["status"])
        if "meta" in fields:
            fields["meta"] = json.dumps(fields["meta"])
        fields["updated_at"] = time.time()
        assignments = ", ".join(f"{k} = ?" for k in fields)
        await self._execute(
            f"UPDATE jobs SET {assignments} WHERE id = ?", (*fields.values(), job_id)
        )

    def spawn(self, job_id: str, work: Callable[[], Awaitable[None]]) -> None:
        """Queue ``work`` to run in the background, recording the outcome on the job.

        The job stays ``queued`` until one of the ``max_concurrent`` slots frees up.
        """

        async def runner() -> None:
            try:
                async with self._slots:
                    await self.update(job_id, status=JobStatus.RUNNING)
                    await work()
            except asyncio.CancelledError:
                await self._record_cancellation(job_id)
                raise
            except Exception as exc:  # noqa: BLE001 - recorded on the job for the client
                log.exception("job %s failed", job_id)
                await self.update(job_id, status=JobStatus.ERROR, error=str(exc))
            finally:
                self._user_cancelled.discard(job_id)
                self._tasks.pop(job_id, None)

        self._tasks[job_id] = asyncio.create_task(runner())

    async def cancel(self, job_id: str, wait: float = 5.0) -> Job:
        """Cancel a queued or running job and return its updated record.

        A queued job stops at once. A running job stops at its next await; work
        already handed to a thread (the Kokoro sentence being synthesized) finishes
        in the background and its result is discarded.
        """
        job = await self.get(job_id)
        if job is None:
            raise JobNotFound(job_id)
        if job.status in FINISHED:
            raise JobNotActive(job)
        task = self._tasks.get(job_id)
        if task is None:
            # Active in the database but not running here (e.g. left over): just mark it.
            await self.update(job_id, status=JobStatus.CANCELLED)
        else:
            self._user_cancelled.add(job_id)
            task.cancel()
            await asyncio.wait({task}, timeout=wait)
        updated = await self.get(job_id)
        assert updated is not None
        return updated

    async def recover_stale(self) -> int:
        """Mark jobs a previous process left queued/running as errors; return how many."""
        conn = self._require_conn()
        now = time.time()

        def run() -> int:
            with conn:
                cur = conn.execute(
                    "UPDATE jobs SET status = ?, error = ?, updated_at = ?"
                    " WHERE status IN (?, ?)",
                    (
                        JobStatus.ERROR.value,
                        "Interrupted by a server restart.",
                        now,
                        JobStatus.QUEUED.value,
                        JobStatus.RUNNING.value,
                    ),
                )
                return cur.rowcount

        async with self._lock:
            count = await asyncio.to_thread(run)
        if count:
            log.warning("marked %d interrupted job(s) as error after restart", count)
        return count

    async def cleanup(self, retention_days: float, output_dir: str | Path) -> int:
        """Delete finished jobs last updated before the retention window, with files.

        Only files inside ``output_dir`` are ever deleted. A row whose file can't be
        removed right now (e.g. locked on Windows) is kept and retried next time.
        """
        if retention_days <= 0:
            return 0
        cutoff = time.time() - retention_days * 86400
        finished = tuple(status.value for status in FINISHED)
        rows = await self._query(
            "SELECT id, result_path FROM jobs WHERE updated_at < ? AND status IN (?, ?, ?)",
            (cutoff, *finished),
        )
        root = Path(output_dir).resolve()
        removable: list[str] = []
        for row in rows:
            if row["result_path"] and not _remove_file(row["result_path"], root):
                continue
            removable.append(row["id"])
        for job_id in removable:
            await self._execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        if removable:
            log.info(
                "cleaned up %d finished job(s) older than %s day(s)",
                len(removable),
                retention_days,
            )
        return len(removable)

    def start_maintenance(
        self,
        retention_days: float,
        output_dir: str | Path,
        interval: float = CLEANUP_INTERVAL_S,
    ) -> None:
        """Run ``cleanup`` now and then every ``interval`` seconds until ``close``."""

        async def loop() -> None:
            while True:
                try:
                    await self.cleanup(retention_days, output_dir)
                except Exception:  # noqa: BLE001 - maintenance must never die
                    log.exception("job cleanup failed")
                await asyncio.sleep(interval)

        self._maintenance = asyncio.create_task(loop())

    async def wait(self, job_id: str, timeout: float = 60.0) -> Job | None:
        """Block until a job finishes (used by tests/CLIs)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = await self.get(job_id)
            if job is None or job.status in FINISHED:
                return job
            await asyncio.sleep(0.05)
        return await self.get(job_id)

    async def _record_cancellation(self, job_id: str) -> None:
        job = await self.get(job_id)
        if job is None or job.status in FINISHED:
            return  # finished just before the cancel landed; keep that outcome
        if job_id in self._user_cancelled:
            await self.update(job_id, status=JobStatus.CANCELLED)
        else:
            await self.update(
                job_id, status=JobStatus.ERROR, error="Job cancelled (server shutdown)."
            )

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


def _remove_file(path: str, root: Path) -> bool:
    """Delete ``path`` if it lies inside ``root``; True when the row can go too."""
    try:
        resolved = Path(path).resolve()
    except OSError:
        return True
    if not resolved.is_relative_to(root):
        log.warning("not deleting %s: outside the output directory", resolved)
        return True  # drop the row, leave the foreign file alone
    try:
        resolved.unlink(missing_ok=True)
    except OSError as exc:
        log.warning("could not delete %s yet: %s", resolved, exc)
        return False
    return True
