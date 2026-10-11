"""End-to-end smoke test against a running server (real Kokoro, no mocks).

Start the server first, in another terminal:

    uv run uvicorn app.main:app --port 8000

Then run:

    uv run python scripts/smoke_test.py                  # all checks
    uv run python scripts/smoke_test.py --skip-jobs      # skip the ~1 min job checks
    uv run python scripts/smoke_test.py --base-url http://127.0.0.1:8765

Restart recovery needs a server restart in the middle, so it is split in two steps:

    uv run python scripts/smoke_test.py --leave-running-job   # prints a job id
    # stop the server (Ctrl+C, or kill it), start it again, then:
    uv run python scripts/smoke_test.py --check-interrupted <job_id>

The topic -> script -> podcast checks need a server started with SCRIPT_WRITER=fake
(they are skipped otherwise); the mp3/loudness checks need ffmpeg on the server.

Exits with status 1 if any check fails. OpenAI is never called, so no quota is used.
The first request after a server start can take ~40 s while Kokoro loads.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

LONG_SCRIPT = "\n".join(
    f"Alice: This is line number {i} of a long test episode."
    if i % 2 == 0
    else f"Bob: And this is reply number {i}, which keeps the show going."
    for i in range(20)
)


class Checks:
    def __init__(self) -> None:
        self.failed = 0
        self.passed = 0

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        return ok

    def skip(self, name: str, why: str) -> None:
        print(f"  [SKIP] {name}  ({why})")


def section(title: str) -> None:
    print(f"\n== {title}")


async def wait_for_status(
    client: httpx.AsyncClient, job_id: str, wanted: set[str], timeout: float = 120
) -> str:
    deadline = time.monotonic() + timeout
    status = "?"
    while time.monotonic() < deadline:
        status = (await client.get(f"/v1/jobs/{job_id}")).json()["status"]
        if status in wanted:
            return status
        await asyncio.sleep(0.3)
    return status


async def run_checks(client: httpx.AsyncClient, provider: str, skip_jobs: bool) -> Checks:
    c = Checks()

    section("Server")
    r = await client.get("/health", headers={"X-Request-ID": "smoke-123"})
    c.check("GET /health", r.status_code == 200)
    c.check("X-Request-ID is echoed back", r.headers.get("x-request-id") == "smoke-123")
    voices = (await client.get("/v1/voices")).json()
    c.check(
        f"provider '{provider}' is registered",
        provider in voices["providers"],
        f"registered: {voices['providers']}",
    )
    if provider not in voices["providers"]:
        return c

    print("  ... warming up the model (first request can take ~40 s)")
    # Streaming is never cached, so this really loads the model.
    await client.post("/v1/speech/stream", json={"text": "Warm up.", "provider": provider})

    section("Cache")
    text = f"Smoke test cache check {time.time():.0f}."
    started = time.perf_counter()
    first = await client.post("/v1/speech", json={"text": text, "provider": provider})
    miss_s = time.perf_counter() - started
    started = time.perf_counter()
    second = await client.post("/v1/speech", json={"text": text, "provider": provider})
    hit_s = time.perf_counter() - started
    c.check(
        "first call is a miss, second a hit",
        (first.headers.get("x-cache"), second.headers.get("x-cache")) == ("miss", "hit"),
        f"miss {miss_s:.2f}s, hit {hit_s:.3f}s",
    )
    c.check("cached audio is identical", first.content == second.content)

    section("Streaming")
    r = await client.post(
        "/v1/speech/stream",
        json={
            "text": "First sentence here. Second one follows. And a third one.",
            "provider": provider,
            "format": "wav",
        },
    )
    headers = r.content.count(b"RIFF")
    c.check("multi-sentence WAV stream has one header", r.status_code == 200 and headers == 1,
            f"status {r.status_code}, RIFF headers: {headers}")
    out = Path("output") / "smoke_stream.wav"
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(r.content)
    print(f"  -> saved {out}: listen to it, there should be no click between sentences")

    long_text = " ".join(f"This is streamed sentence number {i}." for i in range(30))
    try:
        async with client.stream(
            "POST", "/v1/speech/stream",
            json={"text": long_text, "provider": provider, "format": "pcm"},
        ) as s:
            async for _ in s.aiter_bytes():
                break  # hang up after the first chunk
    except httpx.HTTPError:
        pass
    r = await client.post("/v1/speech", json={"text": "Still alive?", "provider": provider})
    c.check("server keeps working after a client hangs up mid-stream", r.status_code == 200,
            "with LOG_LEVEL=DEBUG the log shows 'producer stopped'")

    section("Errors")
    cases = [
        ("unknown voice -> 400", "/v1/speech",
         {"text": "Hi", "provider": provider, "voice": "no-such-voice"}, 400),
        ("voice that looks like a file path -> 400", "/v1/speech",
         {"text": "Hi", "provider": provider, "voice": "C:/x/voice.pt"}, 400),
        ("unknown voice on the stream endpoint -> 400", "/v1/speech/stream",
         {"text": "Hi", "provider": provider, "voice": "no-such-voice"}, 400),
        ("flac on the stream endpoint -> 400", "/v1/speech/stream",
         {"text": "Hi", "provider": provider, "format": "flac"}, 400),
        ("unregistered provider -> 400", "/v1/speech",
         {"text": "Hi", "provider": "gemini"}, 400),
        ("podcast with an unknown voice -> 400 at submit", "/v1/podcast",
         {"script": "Alice: Hi.", "provider": provider, "voices": {"Alice": "nope"}}, 400),
        ("podcast with an invalid gender -> 422", "/v1/podcast",
         {"script": "Host: Hi.", "provider": provider, "genders": {"Host": "robot"}}, 422),
    ]
    for name, path, body, expected in cases:
        r = await client.post(path, json=body)
        c.check(name, r.status_code == expected, f"got {r.status_code}")

    section("Concurrency")
    texts = [f"Parallel request {i} at {time.time():.3f}." for i in range(3)]
    tasks = [
        asyncio.create_task(client.post("/v1/speech", json={"text": t, "provider": provider}))
        for t in texts
    ]
    await asyncio.sleep(0.3)
    started = time.perf_counter()
    r = await client.get("/v1/jobs")
    poll_ms = (time.perf_counter() - started) * 1000
    results = await asyncio.gather(*tasks)
    c.check("3 parallel syntheses all succeed", all(x.status_code == 200 for x in results))
    c.check("GET /v1/jobs stays responsive meanwhile", r.status_code == 200 and poll_ms < 1000,
            f"{poll_ms:.0f} ms")

    await run_ws_checks(c, str(client.base_url), provider)

    section("Jobs")
    if skip_jobs:
        c.skip("job queue and cancel", "--skip-jobs")
        return c
    r = await client.post("/v1/podcast", json={"script": LONG_SCRIPT, "provider": provider})
    job1 = r.json()["job_id"]
    status1 = await wait_for_status(client, job1, {"running", "done", "error"})
    r = await client.post(
        "/v1/podcast", json={"script": LONG_SCRIPT + "\nAlice: Bye.", "provider": provider}
    )
    job2 = r.json()["job_id"]
    await asyncio.sleep(0.5)
    status2 = (await client.get(f"/v1/jobs/{job2}")).json()["status"]
    if status2 == "running":
        c.skip("second job waits as queued", "MAX_CONCURRENT_JOBS > 1?")
    else:
        c.check("second job waits as queued", (status1, status2) == ("running", "queued"),
                f"job1={status1}, job2={status2}")

    r = await client.post(f"/v1/jobs/{job2}/cancel")
    c.check("cancel a queued job", r.status_code == 200 and r.json()["status"] == "cancelled")
    r = await client.post(f"/v1/jobs/{job2}/cancel")
    c.check("cancelling it again -> 409", r.status_code == 409)
    started = time.perf_counter()
    r = await client.post(f"/v1/jobs/{job1}/cancel")
    c.check("cancel a running job", r.status_code == 200 and r.json()["status"] == "cancelled",
            f"took {time.perf_counter() - started:.2f}s")
    r = await client.get(f"/v1/jobs/{job1}/audio")
    c.check("audio of a cancelled job -> 409", r.status_code == 409)
    r = await client.post("/v1/jobs/does-not-exist/cancel")
    c.check("cancel an unknown job -> 404", r.status_code == 404)

    await run_podcast_checks(c, client, provider, cancelled_job=job2)
    return c


SHORT_SCRIPT = "Alice: Welcome to the smoke test.\nBob: Thanks, it's short."


async def run_podcast_checks(
    c: Checks, client: httpx.AsyncClient, provider: str, cancelled_job: str
) -> None:
    """Phase 7: retry, mp3 + loudness, and topic -> script -> podcast."""
    section("Retry")
    r = await client.post(f"/v1/jobs/{cancelled_job}/retry")
    c.check("retry a cancelled job -> 202", r.status_code == 202, f"got {r.status_code}")
    if r.status_code == 202:
        retried = r.json()["job_id"]
        # The long script from the job checks; mostly cached by now.
        status = await wait_for_status(client, retried, {"done", "error", "cancelled"}, 300)
        job = (await client.get(f"/v1/jobs/{retried}")).json()
        c.check("the retried job finishes", status == "done", f"{status}: {job['error']}")
        c.check("it points back to the original", job["retry_of"] == cancelled_job)
        r = await client.post(f"/v1/jobs/{retried}/retry")
        c.check("retrying a done job -> 409", r.status_code == 409)
    r = await client.post("/v1/jobs/does-not-exist/retry")
    c.check("retry an unknown job -> 404", r.status_code == 404)

    section("Podcast mp3 + loudness")
    r = await client.post("/v1/podcast", json={
        "script": SHORT_SCRIPT, "provider": provider, "format": "mp3",
        "loudness": "lufs", "target_lufs": -19, "title": "Smoke test",
    })
    if r.status_code == 400 and "ffmpeg" in r.text:
        c.skip("mp3 at -19 LUFS", "ffmpeg is not installed on the server")
    else:
        c.check("submit an mp3 podcast with loudness lufs -> 202", r.status_code == 202,
                f"got {r.status_code}: {r.text[:200]}")
        job_id = r.json()["job_id"]
        status = await wait_for_status(client, job_id, {"done", "error", "cancelled"}, 300)
        job = (await client.get(f"/v1/jobs/{job_id}")).json()
        c.check("the mp3 job finishes", status == "done", f"{status}: {job['error']}")
        loudness = job.get("loudness") or {}
        output_i = loudness.get("output_i")
        c.check("loudness is -19 +/- 1 LUFS", output_i is not None and abs(output_i + 19) <= 1,
                f"{loudness}")
        c.check("true peak stays under -1 dBTP", (loudness.get("output_tp") or 0) <= -1.0)
        c.check("title and duration are reported",
                job["title"] == "Smoke test" and (job["duration_s"] or 0) > 1)
        r = await client.get(f"/v1/jobs/{job_id}/audio")
        c.check("the audio is an mp3 with an ID3 tag",
                r.headers.get("content-type") == "audio/mpeg" and r.content[:3] == b"ID3")

    section("Topic -> script -> podcast")
    writer = (await client.get("/health")).json().get("script_writer")
    if writer != "fake":
        c.skip("topic -> script -> podcast",
               f"script writer is {writer!r}; start the server with SCRIPT_WRITER=fake "
               "(this test never calls OpenAI)")
        return
    r = await client.post("/v1/podcast/script", json={
        "topic": "the smoke test", "duration_min": 1,
        "hosts": [{"name": "Lan", "gender": "female"}, {"name": "Minh", "gender": "male"}],
    })
    c.check("ask for a script -> 202", r.status_code == 202)
    script_job = r.json()["job_id"]
    status = await wait_for_status(client, script_job, {"done", "error"}, 60)
    c.check("the script job finishes", status == "done")
    r = await client.get(f"/v1/jobs/{script_job}/script")
    script = r.json()
    c.check("the script has turns and the hosts' genders",
            r.status_code == 200 and script["genders"] == {"Lan": "female", "Minh": "male"}
            and len(script["turns"]) > 2)
    r = await client.get(f"/v1/jobs/{script_job}/audio")
    c.check("a script job has no audio -> 404", r.status_code == 404)

    script["turns"] = script["turns"][:3]  # the review step: keep the render short
    r = await client.post("/v1/podcast", json={**script, "provider": provider, "format": "wav"})
    c.check("the script posts to /v1/podcast as is -> 202", r.status_code == 202,
            f"got {r.status_code}: {r.text[:200]}")
    job_id = r.json()["job_id"]
    status = await wait_for_status(client, job_id, {"done", "error", "cancelled"}, 300)
    job = (await client.get(f"/v1/jobs/{job_id}")).json()
    c.check("the episode renders", status == "done", f"{status}: {job['error']}")
    voices = (await client.get(f"/v1/voices?provider={provider}")).json()["voices"]
    gender_of = {v["id"]: v["gender"] for v in voices}
    c.check("hosts are cast by the script's genders",
            gender_of.get(job["voices"].get("Lan")) == "female"
            and gender_of.get(job["voices"].get("Minh")) == "male", f"{job['voices']}")


async def _turn(ws, text: str, timeout: float = 90) -> dict:
    """Send one turn and collect what comes back until it ends."""
    await ws.send(json.dumps({"type": "text", "text": text}))
    await ws.send(json.dumps({"type": "flush"}))
    out: dict = {"sentences": [], "audio": 0, "end": None}

    async def collect() -> None:
        async for raw in ws:
            if isinstance(raw, bytes):
                out["audio"] += len(raw)
                continue
            event = json.loads(raw)
            if event["type"] == "sentence":
                out["sentences"].append(event)
            elif event["type"] in ("done", "cancelled") or "turn" in event:
                out["end"] = event
                return

    await asyncio.wait_for(collect(), timeout)
    return out


async def _first_audio(ws, timeout: float = 60) -> list[dict]:
    """Read until the first audio frame; return the events seen before it."""
    seen: list[dict] = []

    async def read() -> None:
        async for raw in ws:
            if isinstance(raw, bytes):
                return
            seen.append(json.loads(raw))

    await asyncio.wait_for(read(), timeout)
    return seen


async def run_ws_checks(c: Checks, base_url: str, provider: str) -> None:
    section("WebSocket")
    url = base_url.replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
    url += "/v1/speech/ws"
    start = {"type": "start", "provider": provider}

    async with connect(url, max_size=None) as ws:
        await ws.send(json.dumps(start))  # chunking defaults to adaptive
        ready = json.loads(await ws.recv())
        c.check("start -> ready", ready.get("type") == "ready"
                and (ready.get("sample_rate"), ready.get("format")) == (24000, "pcm"),
                f"voice={ready.get('voice')}")
        await ws.send(json.dumps({"type": "text", "text": (
            "This first sentence is long enough to be spoken before the turn ends, "
            "while the rest of the answer is still being written. And then")}))
        seen = await _first_audio(ws)
        c.check("audio arrives before the turn is flushed", True)
        sentence = next((e for e in seen if e.get("type") == "sentence"), {})
        c.check("sentence events carry text and boundary",
                bool(sentence.get("text")) and sentence.get("boundary") in
                ("sentence", "clause", "line", "word", "end"),
                f"{sentence.get('boundary')}: {sentence.get('text', '')[:40]!r}")
        await ws.send(json.dumps({"type": "text", "text": " a second sentence."}))
        await ws.send(json.dumps({"type": "flush"}))
        end = None
        async for raw in ws:
            if not isinstance(raw, bytes) and json.loads(raw)["type"] in ("done", "error"):
                end = json.loads(raw)
                break
        c.check("flush -> done", end is not None and end["type"] == "done")

        # Barge-in on the same session, then a normal turn again.
        await ws.send(json.dumps({"type": "text", "text": (
            "Here is a long answer that will be interrupted part of the way through. "
            "It keeps going with more sentences. And even more after that one.")}))
        await ws.send(json.dumps({"type": "flush"}))
        await _first_audio(ws)
        await ws.send(json.dumps({"type": "cancel"}))
        cancelled = False
        async for raw in ws:
            if not isinstance(raw, bytes) and json.loads(raw)["type"] == "cancelled":
                cancelled = True
                break
        c.check("cancel -> cancelled", cancelled)
        after = await _turn(ws, "Back again after the cancel, speaking normally now.")
        c.check("the session works after a cancel", (after["end"] or {}).get("type") == "done")

    async with connect(url, max_size=None) as ws:
        await ws.send(json.dumps({**start, "chunking": "sentence"}))
        await ws.recv()
        tricky = await _turn(ws, "It takes 3.5 seconds, e.g. on Dr. Smith's laptop. Done.")
        texts = [s["text"] for s in tricky["sentences"]]
        c.check("decimals and abbreviations aren't split",
                texts == ["It takes 3.5 seconds, e.g. on Dr. Smith's laptop.", "Done."],
                f"{texts}")
        again = await _turn(ws, "It takes 3.5 seconds, e.g. on Dr. Smith's laptop. Done.")
        c.check("a repeated turn comes from the cache",
                all(s["cached"] for s in again["sentences"]) and again["sentences"])

    async with connect(url, max_size=None) as ws:
        await ws.send(json.dumps({**start, "voice": "no-such-voice"}))
        error = json.loads(await ws.recv())
        try:
            await asyncio.wait_for(ws.recv(), 5)
            code = None
        except ConnectionClosed as exc:
            code = exc.rcvd.code if exc.rcvd else None
        c.check("a bad start gets an error and close 1008",
                error.get("type") == "error" and code == 1008, f"close code {code}")

    try:
        async with connect(url, origin="https://evil.example"):
            refused = False
    except InvalidStatus:
        refused = True
    c.check("a browser page from an unknown origin is refused", refused)


async def leave_running_job(client: httpx.AsyncClient, provider: str) -> int:
    r = await client.post("/v1/podcast", json={"script": LONG_SCRIPT, "provider": provider})
    job_id = r.json()["job_id"]
    status = await wait_for_status(client, job_id, {"running", "done", "error"})
    print(f"job {job_id} is {status}.")
    print("Now stop the server (Ctrl+C or kill it), start it again, then run:")
    print(f"  uv run python scripts/smoke_test.py --check-interrupted {job_id}")
    return 0 if status == "running" else 1


async def check_interrupted(client: httpx.AsyncClient, job_id: str) -> int:
    r = await client.get(f"/v1/jobs/{job_id}")
    if r.status_code != 200:
        print(f"  [FAIL] job {job_id} not found ({r.status_code})")
        return 1
    job = r.json()
    ok = job["status"] == "error" and (
        "restart" in (job["error"] or "") or "shutdown" in (job["error"] or "")
    )
    print(f"  [{'PASS' if ok else 'FAIL'}] job after restart: {job['status']} - {job['error']}")
    if not ok:
        return 1
    r = await client.post(f"/v1/jobs/{job_id}/retry")
    if r.status_code != 202:
        print(f"  [FAIL] retry the interrupted job ({r.status_code}: {r.text[:200]})")
        return 1
    retried = r.json()["job_id"]
    status = await wait_for_status(client, retried, {"done", "error", "cancelled"}, 300)
    ok = status == "done"
    print(f"  [{'PASS' if ok else 'FAIL'}] retry the interrupted job: {status}")
    return 0 if ok else 1


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--provider", default="kokoro", help="provider to exercise")
    parser.add_argument("--skip-jobs", action="store_true", help="skip job queue/cancel checks")
    parser.add_argument("--leave-running-job", action="store_true",
                        help="start a long job for the restart check and exit")
    parser.add_argument("--check-interrupted", metavar="JOB_ID",
                        help="after a restart, check that JOB_ID was marked as interrupted")
    args = parser.parse_args()

    async with httpx.AsyncClient(base_url=args.base_url, timeout=600) as client:
        try:
            await client.get("/health")
        except httpx.HTTPError:
            print(f"Can't reach {args.base_url}. Start the server first:")
            print("  uv run uvicorn app.main:app --port 8000")
            return 2

        if args.leave_running_job:
            return await leave_running_job(client, args.provider)
        if args.check_interrupted:
            return await check_interrupted(client, args.check_interrupted)

        started = time.perf_counter()
        checks = await run_checks(client, args.provider, args.skip_jobs)

    print(f"\n{checks.passed} passed, {checks.failed} failed "
          f"in {time.perf_counter() - started:.0f}s")
    return 1 if checks.failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
