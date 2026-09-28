"""Tests for the SQLite-backed job store and its in-process worker."""

from __future__ import annotations

import asyncio
import time

import pytest

from app.core.jobs import JobNotActive, JobNotFound, JobStatus, JobStore


@pytest.fixture
async def store(tmp_path):
    s = JobStore(tmp_path / "jobs.db")
    await s.connect()
    try:
        yield s
    finally:
        await s.close()


async def test_create_and_get_round_trip(store):
    job = await store.create("podcast", total=3, meta={"voices": {"A": "v1"}})
    fetched = await store.get(job.id)
    assert fetched is not None
    assert fetched.status is JobStatus.QUEUED
    assert fetched.total == 3
    assert fetched.meta == {"voices": {"A": "v1"}}


async def test_get_unknown_job_returns_none(store):
    assert await store.get("does-not-exist") is None


async def test_update_patches_fields(store):
    job = await store.create("podcast")
    await store.update(job.id, status=JobStatus.RUNNING, progress=2)
    updated = await store.get(job.id)
    assert updated.status is JobStatus.RUNNING
    assert updated.progress == 2
    assert updated.updated_at >= job.updated_at


async def test_update_rejects_unknown_field(store):
    job = await store.create("podcast")
    with pytest.raises(ValueError, match="Unknown job fields"):
        await store.update(job.id, bogus=1)


async def test_list_returns_newest_first(store):
    first = await store.create("podcast")
    await asyncio.sleep(0.01)
    second = await store.create("podcast")
    jobs = await store.list()
    assert [j.id for j in jobs][:2] == [second.id, first.id]


async def test_spawn_marks_job_done(store):
    job = await store.create("podcast", total=1)

    async def work() -> None:
        await store.update(job.id, progress=1, status=JobStatus.DONE)

    store.spawn(job.id, work)
    finished = await store.wait(job.id, timeout=5)
    assert finished.status is JobStatus.DONE
    assert finished.progress == 1


async def test_spawn_records_failure_on_the_job(store):
    job = await store.create("podcast")

    async def work() -> None:
        raise RuntimeError("provider exploded")

    store.spawn(job.id, work)
    finished = await store.wait(job.id, timeout=5)
    assert finished.status is JobStatus.ERROR
    assert "provider exploded" in finished.error


async def test_spawn_sets_running_before_work_completes(store):
    job = await store.create("podcast")
    gate = asyncio.Event()
    observed: list[JobStatus] = []

    async def work() -> None:
        observed.append((await store.get(job.id)).status)
        gate.set()

    store.spawn(job.id, work)
    await asyncio.wait_for(gate.wait(), timeout=5)
    assert observed == [JobStatus.RUNNING]


async def test_close_cancels_running_work(tmp_path):
    s = JobStore(tmp_path / "jobs.db")
    await s.connect()
    job = await s.create("podcast")
    started = asyncio.Event()

    async def work() -> None:
        started.set()
        await asyncio.sleep(30)

    s.spawn(job.id, work)
    await asyncio.wait_for(started.wait(), timeout=5)
    await s.close()
    # The connection is closed, so re-open to inspect the recorded state.
    reopened = JobStore(tmp_path / "jobs.db")
    await reopened.connect()
    try:
        assert (await reopened.get(job.id)).status is JobStatus.ERROR
    finally:
        await reopened.close()


async def test_operations_before_connect_raise(tmp_path):
    s = JobStore(tmp_path / "jobs.db")
    with pytest.raises(RuntimeError, match="not connected"):
        await s.get("anything")


def _blocking_work(started: asyncio.Event, release: asyncio.Event):
    async def work() -> None:
        started.set()
        await release.wait()

    return work


async def test_jobs_queue_behind_the_concurrency_limit(tmp_path):
    s = JobStore(tmp_path / "jobs.db", max_concurrent=1)
    await s.connect()
    try:
        first, second = await s.create("podcast"), await s.create("podcast")
        started1, started2, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        s.spawn(first.id, _blocking_work(started1, release))
        s.spawn(second.id, _blocking_work(started2, release))
        await asyncio.wait_for(started1.wait(), timeout=5)
        await asyncio.sleep(0.1)

        assert (await s.get(first.id)).status is JobStatus.RUNNING
        assert (await s.get(second.id)).status is JobStatus.QUEUED
        assert not started2.is_set()

        release.set()
        await asyncio.wait_for(started2.wait(), timeout=5)
    finally:
        await s.close()


async def test_cancel_running_job(store):
    job = await store.create("podcast")
    started, release = asyncio.Event(), asyncio.Event()
    store.spawn(job.id, _blocking_work(started, release))
    await asyncio.wait_for(started.wait(), timeout=5)

    cancelled = await store.cancel(job.id)
    assert cancelled.status is JobStatus.CANCELLED


async def test_cancel_queued_job(tmp_path):
    s = JobStore(tmp_path / "jobs.db", max_concurrent=1)
    await s.connect()
    try:
        first, second = await s.create("podcast"), await s.create("podcast")
        started1, started2, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        s.spawn(first.id, _blocking_work(started1, release))
        s.spawn(second.id, _blocking_work(started2, release))
        await asyncio.wait_for(started1.wait(), timeout=5)

        assert (await s.cancel(second.id)).status is JobStatus.CANCELLED
        assert (await s.get(first.id)).status is JobStatus.RUNNING
        assert not started2.is_set()
    finally:
        await s.close()


async def test_cancel_finished_job_is_refused(store):
    job = await store.create("podcast")
    await store.update(job.id, status=JobStatus.DONE)
    with pytest.raises(JobNotActive):
        await store.cancel(job.id)


async def test_cancel_unknown_job_raises(store):
    with pytest.raises(JobNotFound):
        await store.cancel("missing")


async def test_job_finishing_as_cancel_lands_keeps_its_outcome(store):
    job = await store.create("podcast")

    async def work() -> None:
        await store.update(job.id, status=JobStatus.DONE)
        await asyncio.sleep(30)  # still "running" as a task, but already done

    store.spawn(job.id, work)
    finished = await store.wait(job.id, timeout=5)
    assert finished.status is JobStatus.DONE
    store._tasks[job.id].cancel()
    await asyncio.sleep(0.1)
    assert (await store.get(job.id)).status is JobStatus.DONE


async def test_recover_stale_fails_jobs_left_by_a_previous_process(store):
    queued = await store.create("podcast")
    running = await store.create("podcast")
    done = await store.create("podcast")
    await store.update(running.id, status=JobStatus.RUNNING)
    await store.update(done.id, status=JobStatus.DONE)

    assert await store.recover_stale() == 2
    for job_id in (queued.id, running.id):
        job = await store.get(job_id)
        assert job.status is JobStatus.ERROR
        assert "restart" in job.error
    assert (await store.get(done.id)).status is JobStatus.DONE


async def _aged(store, status: JobStatus, days_old: float, result_path=None):
    job = await store.create("podcast")
    await store.update(job.id, status=status, result_path=result_path)
    old = time.time() - days_old * 86400
    await store._execute("UPDATE jobs SET updated_at = ? WHERE id = ?", (old, job.id))
    return job


async def test_cleanup_deletes_old_finished_jobs_and_their_files(store, tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    old_file, new_file = out / "old.wav", out / "new.wav"
    old_file.write_bytes(b"x")
    new_file.write_bytes(b"x")
    old = await _aged(store, JobStatus.DONE, 10, str(old_file))
    new = await _aged(store, JobStatus.DONE, 1, str(new_file))
    old_error = await _aged(store, JobStatus.ERROR, 10)
    old_running = await _aged(store, JobStatus.RUNNING, 10)

    assert await store.cleanup(7, out) == 2
    assert await store.get(old.id) is None
    assert await store.get(old_error.id) is None
    assert not old_file.exists()
    assert await store.get(new.id) is not None and new_file.exists()
    assert await store.get(old_running.id) is not None  # active jobs are never cleaned


async def test_cleanup_never_deletes_files_outside_the_output_dir(store, tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    foreign = tmp_path / "precious.wav"
    foreign.write_bytes(b"keep me")
    job = await _aged(store, JobStatus.DONE, 10, str(foreign))

    await store.cleanup(7, out)
    assert foreign.exists()
    assert await store.get(job.id) is None


async def test_cleanup_tolerates_missing_files_and_can_be_disabled(store, tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    job = await _aged(store, JobStatus.DONE, 10, str(out / "gone.wav"))
    assert await store.cleanup(0, out) == 0
    assert await store.get(job.id) is not None
    assert await store.cleanup(7, out) == 1
