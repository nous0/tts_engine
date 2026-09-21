"""Tests for the SQLite-backed job store and its in-process worker."""

from __future__ import annotations

import asyncio

import pytest

from app.core.jobs import JobStatus, JobStore


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
