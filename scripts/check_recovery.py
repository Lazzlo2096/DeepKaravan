"""Opt-in restart check for the single-container in-memory runtime (no client traffic)."""

import argparse
import subprocess
import time
from pathlib import Path

from smoke import request, wait_result

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    code, task = request(
        base, "/api/v1/embeddings/async", {"model": "baai/bge-m3", "input": ["Recovery check"]}
    )
    assert code == 202, (code, task)
    wait_result(base, task["result_url"], args.timeout)
    subprocess.run(["docker", "compose", "restart", "api"], cwd=ROOT, check=True)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        try:
            if request(base, "/health/ready")[0] == 200:
                break
        except OSError:
            pass
        time.sleep(1)
    else:
        raise TimeoutError("API did not become ready after restart")
    assert request(base, task["status_url"])[0] == 404
    assert request(base, task["result_url"])[0] == 404
    code, result = request(
        base,
        "/api/v1/embeddings",
        {"model": "baai/bge-m3", "input": ["After restart"]},
        timeout=args.timeout,
    )
    assert code == 200 and result["data"][0]["embedding"], (code, result)
    print("PASS restart: old jobs lost, model reloaded, new inference succeeds")


if __name__ == "__main__":
    main()
