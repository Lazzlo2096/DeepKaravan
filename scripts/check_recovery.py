"""Exercise queue persistence and crash recovery on a disposable CPU Compose stack.

This check stops the CPU worker and restarts Redis. Use it without other clients.
Run the regular smoke test first so the model is already downloaded.
"""

import argparse
import subprocess
import time
from pathlib import Path

from smoke import request, wait_result

ROOT = Path(__file__).resolve().parents[1]
RECOVERY_CONTAINER = "deepkaravan-recovery-check"


def compose(*args):
    return subprocess.check_output(
        ["docker", "compose", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    payload = {"input": ["Recovery check"], "embedding_types": ["dense"]}
    existing = subprocess.run(
        ["docker", "container", "inspect", RECOVERY_CONTAINER],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if existing.returncode == 0:
        raise RuntimeError(f"Check container already exists: {RECOVERY_CONTAINER}")
    recovery_started = False

    try:
        compose("stop", "--timeout", "10", "celery-cpu")
        code, first = request(base, "/v1/embeddings/async", payload)
        assert code == 202, (code, first)
        code, second = request(base, "/v1/embeddings/async", payload)
        assert code == 202, (code, second)

        compose("restart", "redis")
        for task in (first, second):
            code, current = request(base, task["status_url"])
            assert code == 200 and current["status"] == "queued", (code, current)
        assert int(compose("exec", "-T", "redis", "redis-cli", "LLEN", "embeddings")) == 2
        print("PASS queued tasks and status metadata survive Redis restart", flush=True)

        compose("start", "celery-cpu")
        for task in (first, second):
            result = wait_result(base, task["result_url"], 180)
            assert len(result["data"][0]["dense"]) == 1024
            for prefix in ("task", "result"):
                ttl = int(
                    compose(
                        "exec",
                        "-T",
                        "redis",
                        "redis-cli",
                        "-n",
                        "1",
                        "TTL",
                        f"deepkaravan:{prefix}:{task['task_id']}",
                    )
                )
                assert 86200 <= ttl <= 86400, ttl
        first_status = request(base, first["status_url"])[1]
        second_status = request(base, second["status_url"])[1]
        assert first_status["finished_at"] <= second_status["started_at"]
        print("PASS FIFO completion after restart; result and status TTL are 24 hours", flush=True)

        code, interrupted = request(
            base,
            "/v1/embeddings/async",
            {"input": ["recovery " * 256] * 32, "embedding_types": ["dense"]},
        )
        assert code == 202
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            _, current = request(base, interrupted["status_url"])
            if current["status"] == "running":
                break
            time.sleep(0.02)
        else:
            raise TimeoutError("Worker did not start the crash-check task")
        original_start = current["started_at"]
        # Zero-second stop forces SIGKILL, without the normal warm-shutdown requeue.
        compose("stop", "--timeout", "0", "celery-cpu")
        _, current = request(base, interrupted["status_url"])
        assert current["status"] == "running", current
        print("PASS worker interrupted during real inference", flush=True)

        # Shorten only this check's recovery delay; leave production .env unchanged.
        compose(
            "run",
            "-d",
            "--rm",
            "--no-deps",
            "--name",
            RECOVERY_CONTAINER,
            "-e",
            "VISIBILITY_TIMEOUT_SECONDS=3",
            "celery-cpu",
        )
        recovery_started = True
        result = wait_result(base, interrupted["result_url"], 240)
        assert len(result["data"]) == 32
        _, recovered = request(base, interrupted["status_url"])
        assert recovered["started_at"] != original_start
        print("PASS interrupted task redelivered and recomputed with the same task_id", flush=True)
    finally:
        if recovery_started:
            subprocess.run(
                ["docker", "stop", "--time", "10", RECOVERY_CONTAINER],
                cwd=ROOT,
                capture_output=True,
                check=False,
            )
        compose("start", "celery-cpu")


if __name__ == "__main__":
    main()
