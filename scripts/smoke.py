"""Smoke-test a running API and real BGE-M3 worker using only the Python stdlib."""

import argparse
import json
import time
import urllib.error
import urllib.request


def request(base_url, path, payload=None, *, timeout=90):
    data = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
    req = urllib.request.Request(
        base_url + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def wait_result(base_url, result_url, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code, result = request(base_url, result_url)
        if code == 200:
            return result
        if code != 202:
            raise RuntimeError(f"Unexpected result response {code}: {result}")
        time.sleep(0.5)
    raise TimeoutError(f"Task did not finish within {timeout}s: {result_url}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    assert request(base, "/health/ready")[0] == 200

    code, accepted = request(
        base,
        "/v1/embeddings/async",
        {
            "input": ["Привет, мир", "Search across documents"],
            "embedding_types": ["dense", "sparse", "colbert"],
        },
    )
    assert code == 202, (code, accepted)
    result = wait_result(base, accepted["result_url"], args.timeout)
    assert result["embedding_types"] == ["dense", "sparse", "colbert"]
    assert len(result["data"]) == 2
    for i, item in enumerate(result["data"]):
        assert item["index"] == i
        assert len(item["dense"]) == 1024
        assert abs(sum(x * x for x in item["dense"]) - 1) < 0.03
        assert item["sparse"] and all(key.isdigit() for key in item["sparse"])
        assert item["colbert"] and len(item["colbert"][0]) == 1024
    print("PASS async inference: dense + sparse + ColBERT, two texts, original order")

    code, result = request(
        base,
        "/v1/embeddings",
        {"input": ["Проверка"]},
        timeout=args.timeout,
    )
    assert code == 200, (code, result)
    assert set(result["data"][0]) == {"index", "dense"}
    print("PASS synchronous inference and selection of embedding types")

    code, result = request(base, "/v1/embeddings", {"input": []}, timeout=args.timeout)
    assert code == 200 and result["data"] == [], (code, result)
    print("PASS empty batch")

    code, error = request(base, "/v1/embeddings/async", {"input": ["hello " * 9000]})
    assert code == 422 and error["detail"]["code"] == "text_too_long", (code, error)
    print("PASS long text rejected before enqueue")


if __name__ == "__main__":
    main()
